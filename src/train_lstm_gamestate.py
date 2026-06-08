import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils import (
    DATA_DIR,
    FLOAT_DIM,
    N_BITS,
    _canonical_action_id,
    _clean_input,
    _mirror_input,
    build_action_id_vocab,
    build_input_vocab,
    extract_action_idx,
    frame_features,
    split_episodes,
)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_DIM      = 8  # embedding dims per action ID
FRAME_DIM      = FLOAT_DIM + EMBED_DIM * 2
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
SEQ_LEN        = 128 # number of past frames to remember
STRIDE         = 16 # increment step of the sliding window of memory frames
BATCH_SIZE      = 64
N_EPOCHS        = 50
LOOKAHEAD       = 40   # frames to look ahead for an upcoming attack
PRE_ATTACK_MULT = 3.0  # loss multiplier for frames inside the lookahead window
LR             = 2e-4
SAVE_PATH      = "models/lstm_gamestate/"
SAVE_FILE      = os.path.join(SAVE_PATH, "lstm_policy.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(SAVE_PATH, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMPolicy(nn.Module):
    def __init__(self, n_classes: int, n_action_ids: int):
        super().__init__()
        self.n_classes       = n_classes
        self.input_ints      = []
        self.action_id_vocab = {}
        self.action_embed    = nn.Embedding(n_action_ids, EMBED_DIM)
        self.lstm            = nn.LSTM(FRAME_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.head            = nn.Linear(HIDDEN_DIM, n_classes)

    def _build_input(self, x_float, x_ids):
        """x_float: (B, S, FLOAT_DIM), x_ids: (B, S, 2) → (B, S, FRAME_DIM)"""
        p_emb   = self.action_embed(x_ids[..., 0])  # (B, S, EMBED_DIM)
        opp_emb = self.action_embed(x_ids[..., 1])  # (B, S, EMBED_DIM)
        return torch.cat([x_float, p_emb, opp_emb], dim=-1)

    def forward(self, x_float, x_ids, hidden=None):
        x = self._build_input(x_float, x_ids)
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden

    def predict(self, float_features: np.ndarray, action_ids: list,
                hidden=None, mirror: bool = False):
        """Single-frame inference. Returns (bits: np.ndarray (N_BITS,), new hidden)."""
        mapped = [self.action_id_vocab.get(_canonical_action_id(a), 0) for a in action_ids]
        x_float = torch.tensor(float_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        x_ids   = torch.tensor(mapped,         dtype=torch.long   ).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, hidden = self.forward(x_float, x_ids, hidden)
        class_idx = logits.squeeze().argmax().item()
        val = _clean_input(self.input_ints[class_idx])
        if mirror:
            val = _mirror_input(val)
        bits = np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int)
        return bits, hidden


# ── Dataset ────────────────────────────────────────────────────────────────────

class EpisodeDataset(Dataset):
    def __init__(self, episodes: list[list[dict]], vocab: dict, action_id_vocab: dict, input_ints: list):
        self.chunks = []
        _ATTACK_MASK = 0b1111110000

        for ep in episodes:
            if len(ep) < 2:
                continue

            splits  = [frame_features(f) for f in ep[:-1]]
            floats  = np.array([s[0] for s in splits], dtype=np.float32)
            ids     = np.array([[action_id_vocab.get(a, 0) for a in s[1]] for s in splits], dtype=np.int64)
            actions = np.array([extract_action_idx(f, vocab) for f in ep[:-1]], dtype=np.int64)

            # compute per-frame loss weights
            n  = len(actions)
            fw = np.ones(n, dtype=np.float32)
            for i in range(n):
                if input_ints[actions[i]] == 0:
                    # keep if part of a motion (non-zero input coming within LOOKAHEAD), else ignore
                    has_motion = any(input_ints[actions[j]] != 0 for j in range(i, min(i + LOOKAHEAD, n)))
                    fw[i] = 1.0 if has_motion else 0.0
                    continue
                for j in range(i, min(i + LOOKAHEAD, n)):
                    if input_ints[actions[j]] & _ATTACK_MASK:
                        fw[i] = PRE_ATTACK_MULT
                        break

            for start in range(0, len(floats), STRIDE):
                xf = floats[start : start + SEQ_LEN]
                xi = ids[start : start + SEQ_LEN]
                y  = actions[start : start + SEQ_LEN]
                w  = fw[start : start + SEQ_LEN]
                if len(y) < 1:
                    continue

                actual_len = len(y)
                if actual_len < SEQ_LEN:
                    pad = SEQ_LEN - actual_len
                    xf = np.vstack([xf, np.zeros((pad, FLOAT_DIM), dtype=np.float32)])
                    xi = np.vstack([xi, np.zeros((pad, 2), dtype=np.int64)])
                    y  = np.concatenate([y, np.zeros(pad, dtype=np.int64)])
                    w  = np.concatenate([w, np.ones(pad, dtype=np.float32)])

                mask = np.zeros(SEQ_LEN, dtype=np.float32)
                mask[:actual_len] = 1.0

                self.chunks.append((
                    torch.tensor(xf, dtype=torch.float32),
                    torch.tensor(xi, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long),
                    torch.tensor(mask, dtype=torch.float32),
                    torch.tensor(w,    dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    frames = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            data = json.load(f)
        frames += data["frame_data"]
        print(f"Loaded {filename}: {len(data['frame_data'])} frames")
    print(f"Total frames: {len(frames)}")

    vocab, input_ints           = build_input_vocab(frames)
    action_id_vocab, n_action_ids = build_action_id_vocab(frames)
    N_CLASSES = len(input_ints)
    print(f"Input vocabulary:     {N_CLASSES} distinct inputs")
    print(f"Action ID vocabulary: {n_action_ids} distinct action IDs")

    episodes = split_episodes(frames)
    print(f"Built {len(episodes)} episodes")

    dataset = EpisodeDataset(episodes, vocab, action_id_vocab, input_ints)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Total chunks: {len(dataset)}")

    all_labels = torch.cat([y for _, _, y, _, _ in dataset], dim=0)
    counts     = torch.bincount(all_labels, minlength=N_CLASSES).float()

    _ATTACK_MASK = 0b1111110000  # bits 4-9
    class_weight = torch.zeros(N_CLASSES)
    for i, val in enumerate(input_ints):
        c = counts[i].item()
        if c == 0:
            class_weight[i] = 0.0
        elif val & _ATTACK_MASK:
            class_weight[i] = 1.0 / (c ** 0.3)   # attacks: upweight, but not so much that distance is ignored
        elif val == 0:
            class_weight[i] = 1.0 / (c ** 0.8)  # idle neutral: still discouraged but not zeroed
        else:
            class_weight[i] = 1.0 / (c ** 0.5)   # movement-only: moderate downweight

    class_weight = (class_weight / class_weight.sum() * N_CLASSES).to(DEVICE)

    model     = LSTMPolicy(n_classes=N_CLASSES, n_action_ids=n_action_ids).to(DEVICE)
    model.input_ints      = input_ints
    model.action_id_vocab = action_id_vocab
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weight, reduction="none")

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for xf, xi, y, mask, fw in loader:
            xf, xi, y, mask, fw = xf.to(DEVICE), xi.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), fw.to(DEVICE)

            logits, _ = model(xf, xi)
            B, S, C   = logits.shape
            loss = loss_fn(logits.view(B * S, C), y.view(B * S))
            loss = loss.view(B, S)
            eff  = mask * fw
            loss = (loss * eff).sum() / eff.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1:>2}/{N_EPOCHS}  loss={avg_loss:.4f}")

    torch.save({
        "state_dict":      model.state_dict(),
        "input_ints":      input_ints,
        "action_id_vocab": action_id_vocab,
        "n_action_ids":    n_action_ids,
        "n_classes":       N_CLASSES,
    }, SAVE_FILE)
    print(f"Saved to {SAVE_FILE}")


if __name__ == "__main__":
    main()
