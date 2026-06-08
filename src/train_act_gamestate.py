import json
import math
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
EMBED_DIM       = 8
D_MODEL         = 64
NHEAD           = 2
NUM_LAYERS      = 2
DIM_FF          = 256
DROPOUT         = 0.1
LATENT_DIM      = 32    # CVAE latent size — captures which combo/plan to commit to
KL_WEIGHT       = 1e-4  # beta — ramped up slowly to prevent posterior collapse
KL_WARMUP       = 0.3   # fraction of epochs before KL loss reaches full weight
LOOKAHEAD       = 20    # frames to look ahead when masking isolated neutral frames
SEQ_LEN         = 64    # observation window (past frames fed to the encoder)
CHUNK_LEN       = 16    # frames to predict per forward pass (matches replan interval)
STRIDE          = 16    # sliding window step for dataset construction
BATCH_SIZE      = 64
N_EPOCHS        = 40
LR              = 1e-4
SAVE_PATH       = "models/act_gamestate/"
SAVE_FILE       = os.path.join(SAVE_PATH, "act_policy.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(SAVE_PATH, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────────

class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CVAEEncoder(nn.Module):
    """Encodes the target action chunk into latent z during training.

    At inference this is unused — z is set to zeros (prior mean), giving
    deterministic behaviour. The latent captures which combo/plan the
    demonstration chose, so the decoder can commit to one coherent sequence.
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.action_embed = nn.Embedding(n_classes, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=1, enable_nested_tensor=False)
        self.fc_mean   = nn.Linear(D_MODEL, LATENT_DIM)
        self.fc_logvar = nn.Linear(D_MODEL, LATENT_DIM)

    def forward(self, y: torch.Tensor, obs_summary: torch.Tensor):
        """
        y           : (B, CHUNK_LEN)  target action class indices
        obs_summary : (B, D_MODEL)    pooled observation from main encoder
        returns (z_mean, z_logvar) each (B, LATENT_DIM)
        """
        act_emb = self.action_embed(y)                  # (B, CHUNK_LEN, D_MODEL)
        cls     = obs_summary.unsqueeze(1)              # (B, 1, D_MODEL)
        seq     = torch.cat([cls, act_emb], dim=1)      # (B, CHUNK_LEN+1, D_MODEL)
        out     = self.encoder(seq)[:, 0]               # (B, D_MODEL) — CLS token output
        return self.fc_mean(out), self.fc_logvar(out)


class ACTPolicy(nn.Module):
    def __init__(self, n_classes: int, n_action_ids: int):
        super().__init__()
        self.n_classes       = n_classes
        self.input_ints      = []
        self.action_id_vocab = {}

        self.action_embed = nn.Embedding(n_action_ids, EMBED_DIM)
        self.input_proj   = nn.Linear(FLOAT_DIM + 2 * EMBED_DIM, D_MODEL)
        self.pos_enc      = SinusoidalPosEnc(D_MODEL, max_len=SEQ_LEN + 1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder      = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS, enable_nested_tensor=False)
        self.cvae_encoder = CVAEEncoder(n_classes)
        self.head         = nn.Linear(D_MODEL + LATENT_DIM, CHUNK_LEN * n_classes)

    def _embed(self, x_float: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        p_emb   = self.action_embed(x_ids[..., 0])
        opp_emb = self.action_embed(x_ids[..., 1])
        return self.input_proj(torch.cat([x_float, p_emb, opp_emb], dim=-1))

    def _encode_obs(self, x_float: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(x_float, x_ids)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x.mean(dim=1)                            # (B, D_MODEL)

    def forward(self, x_float: torch.Tensor, x_ids: torch.Tensor, y: torch.Tensor = None):
        """
        x_float : (B, SEQ_LEN, FLOAT_DIM)
        x_ids   : (B, SEQ_LEN, 2)
        y       : (B, CHUNK_LEN) target actions — provided during training, None at inference
        returns   (logits (B, CHUNK_LEN, N_CLASSES), z_mean, z_logvar)
                  z_mean/z_logvar are None when y is None
        """
        obs = self._encode_obs(x_float, x_ids)          # (B, D_MODEL)

        if y is not None:
            z_mean, z_logvar = self.cvae_encoder(y, obs)
            z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        else:
            z        = torch.zeros(obs.shape[0], LATENT_DIM, device=obs.device)
            z_mean   = None
            z_logvar = None

        logits = self.head(torch.cat([obs, z], dim=-1))
        return logits.view(-1, CHUNK_LEN, self.n_classes), z_mean, z_logvar

    def predict_chunk(self, obs_floats: np.ndarray, obs_ids_raw: np.ndarray) -> list:
        """
        obs_floats   : (SEQ_LEN, FLOAT_DIM)
        obs_ids_raw  : (SEQ_LEN, 2)  raw action IDs (vocab-mapped internally)
        Returns CHUNK_LEN bit arrays in Cammy-relative coordinates.
        """
        mapped = np.array(
            [[self.action_id_vocab.get(_canonical_action_id(int(a)), 0) for a in row]
             for row in obs_ids_raw],
            dtype=np.int64,
        )
        xf = torch.tensor(obs_floats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        xi = torch.tensor(mapped,     dtype=torch.long   ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = self.forward(xf, xi, y=None)    # z=0, deterministic
        class_idxs = logits.squeeze(0).argmax(dim=-1).tolist()
        result = []
        for idx in class_idxs:
            val  = _clean_input(self.input_ints[idx])
            bits = np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int)
            result.append(bits)
        return result


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    def __init__(self, episodes: list, vocab: dict, action_id_vocab: dict, input_ints: list):
        self.chunks = []

        for ep in episodes:
            if len(ep) < CHUNK_LEN + 1:
                continue

            splits  = [frame_features(f) for f in ep]
            floats  = np.array([s[0] for s in splits], dtype=np.float32)
            ids     = np.array([[action_id_vocab.get(a, 0) for a in s[1]] for s in splits], dtype=np.int64)
            actions = np.array([extract_action_idx(f, vocab) for f in ep], dtype=np.int64)
            n       = len(actions)

            for t in range(0, n - 1, STRIDE):
                obs_start  = max(0, t - SEQ_LEN + 1)
                xf         = floats[obs_start : t + 1]
                xi         = ids[obs_start : t + 1]
                actual_obs = len(xf)

                tgt_end    = min(n, t + CHUNK_LEN)
                y          = actions[t : tgt_end]
                actual_tgt = len(y)

                # zero out isolated neutral frames — neutral not followed by any action in LOOKAHEAD frames
                fw = np.ones(actual_tgt, dtype=np.float32)
                for k in range(actual_tgt):
                    if input_ints[y[k]] == 0:
                        look_end   = min(n, t + k + 1 + LOOKAHEAD)
                        has_action = any(input_ints[actions[j]] != 0 for j in range(t + k + 1, look_end))
                        if not has_action:
                            fw[k] = 0.0

                if actual_obs < SEQ_LEN:
                    pad = SEQ_LEN - actual_obs
                    xf  = np.vstack([np.zeros((pad, FLOAT_DIM), dtype=np.float32), xf])
                    xi  = np.vstack([np.zeros((pad, 2),         dtype=np.int64),   xi])

                tgt_mask = np.zeros(CHUNK_LEN, dtype=np.float32)
                tgt_mask[:actual_tgt] = 1.0
                fw_full  = np.zeros(CHUNK_LEN, dtype=np.float32)
                fw_full[:actual_tgt] = fw
                if actual_tgt < CHUNK_LEN:
                    y = np.concatenate([y, np.zeros(CHUNK_LEN - actual_tgt, dtype=np.int64)])

                self.chunks.append((
                    torch.tensor(xf,       dtype=torch.float32),
                    torch.tensor(xi,       dtype=torch.long),
                    torch.tensor(y,        dtype=torch.long),
                    torch.tensor(tgt_mask, dtype=torch.float32),
                    torch.tensor(fw_full,  dtype=torch.float32),
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

    vocab, input_ints             = build_input_vocab(frames)
    action_id_vocab, n_action_ids = build_action_id_vocab(frames)
    N_CLASSES = len(input_ints)
    print(f"Input vocabulary:     {N_CLASSES} distinct inputs")
    print(f"Action ID vocabulary: {n_action_ids} distinct action IDs")

    episodes = split_episodes(frames)
    print(f"Built {len(episodes)} episodes")

    dataset = ChunkDataset(episodes, vocab, action_id_vocab, input_ints)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Total chunks: {len(dataset)}")

    all_labels   = torch.cat([y for _, _, y, _, _ in dataset], dim=0)
    counts       = torch.bincount(all_labels, minlength=N_CLASSES).float()
    _ATTACK_MASK = 0b1111110000
    class_weight = torch.zeros(N_CLASSES)
    for i, val in enumerate(input_ints):
        c = counts[i].item()
        if c == 0:
            class_weight[i] = 0.0
        elif val == 0:
            class_weight[i] = 1.0 / (c ** 0.9)   # neutral — strongly downweight
        elif val & _ATTACK_MASK:
            class_weight[i] = 1.0 / (c ** 0.4)   # attacks — upweight
        else:
            class_weight[i] = 1.0 / (c ** 0.5)   # movement — moderate
    class_weight = (class_weight / class_weight.sum() * N_CLASSES).to(DEVICE)

    model = ACTPolicy(n_classes=N_CLASSES, n_action_ids=n_action_ids).to(DEVICE)
    model.input_ints      = input_ints
    model.action_id_vocab = action_id_vocab

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weight, reduction="none")

    for epoch in range(N_EPOCHS):
        # ramp KL from 0 → full weight over first KL_WARMUP fraction of training
        kl_anneal  = min(1.0, max(0.0, epoch / (N_EPOCHS * KL_WARMUP)))
        current_kl = KL_WEIGHT * kl_anneal
        total_loss = 0.0
        for xf, xi, y, mask, fw in loader:
            xf, xi, y, mask, fw = (
                xf.to(DEVICE), xi.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), fw.to(DEVICE),
            )

            logits, z_mean, z_logvar = model(xf, xi, y=y)
            B, C_LEN, CLS = logits.shape

            recon_loss = loss_fn(logits.view(B * C_LEN, CLS), y.view(B * C_LEN))
            eff_mask   = mask * fw
            recon_loss = (recon_loss.view(B, C_LEN) * eff_mask).sum() / eff_mask.sum().clamp(min=1.0)

            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(dim=-1).mean()

            loss = recon_loss + current_kl * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        lr_now   = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1:>2}/{N_EPOCHS}  loss={avg_loss:.4f}  kl_w={current_kl:.1e}  lr={lr_now:.2e}")

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
