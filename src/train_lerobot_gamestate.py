import json
import os

import numpy as np
import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader, Dataset

from utils import (
    DATA_DIR,
    FLOAT_DIM,
    MAX_ACTION_ID,
    build_input_vocab,
    extract_input_index,
    frame_features,
    split_episodes,
)

# ── Config ─────────────────────────────────────────────────────────────────────
N_OBS_STEPS = 1
STATE_DIM   = FLOAT_DIM + 2        # 18 floats + 2 normalised action IDs
FLAT_DIM    = N_OBS_STEPS * STATE_DIM
CHUNK_SIZE  = 1
STRIDE      = 1
BATCH_SIZE  = 256
N_EPOCHS    = 40
LR          = 1e-4
KL_WEIGHT   = 0.001

SAVE_PATH = "models/lerobot_act/"
SAVE_FILE = os.path.join(SAVE_PATH, "act_policy.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(SAVE_PATH, exist_ok=True)


# ── Feature helpers (used by inference scripts) ────────────────────────────────

def _build_obs(history: list[tuple]) -> np.ndarray:
    """Return a FLAT_DIM vector from the last N_OBS_STEPS (float_feats, action_ids) tuples."""
    while len(history) < N_OBS_STEPS:
        history = [history[0]] + history
    history = history[-N_OBS_STEPS:]
    vecs = []
    for floats, action_ids in history:
        norm_ids = [a / MAX_ACTION_ID for a in action_ids]
        vecs.append(np.array(floats + norm_ids, dtype=np.float32))
    return np.concatenate(vecs)


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    """Each sample: observation state → chunk of one-hot action vectors.

    Actions are encoded as one-hot over the input vocabulary rather than raw
    bits. This means every target has exactly one 1, making L1 loss well-posed:
    the model is pushed to predict 1 for the correct class and 0 for all others.
    """

    def __init__(self, episodes: list, vocab: dict, vocab_size: int):
        obs_list    = []
        action_list = []
        pad_list    = []

        for ep in episodes:
            if len(ep) < CHUNK_SIZE + 1:
                continue

            feats = [frame_features(f) for f in ep]
            state = np.array(
                [np.concatenate([fl, [a / MAX_ACTION_ID for a in ids]])
                 for fl, ids in feats],
                dtype=np.float32,
            )                                              # (n, STATE_DIM)
            class_ids = np.array(
                [extract_input_index(f, vocab) for f in ep], dtype=np.int32
            )                                              # (n,)
            n = len(ep)

            for t in range(0, n - 1, STRIDE):
                if class_ids[t] == 0:
                    continue  # skip neutral starting frames

                tgt_end    = min(n, t + CHUNK_SIZE)
                actual_tgt = tgt_end - t

                # one-hot encode each frame in the chunk
                action_chunk = np.zeros((CHUNK_SIZE, vocab_size), dtype=np.float32)
                for i, cls in enumerate(class_ids[t:tgt_end]):
                    action_chunk[i, cls] = 1.0

                pad_mask              = np.zeros(CHUNK_SIZE, dtype=bool)
                pad_mask[actual_tgt:] = True

                obs_list.append(state[t])
                action_list.append(action_chunk)
                pad_list.append(pad_mask)

        print(f"Chunks: {len(obs_list)}")

        self.obs    = torch.from_numpy(np.stack(obs_list))
        self.action = torch.from_numpy(np.stack(action_list))
        self.pad    = torch.from_numpy(np.stack(pad_list))

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return {
            "observation.state": self.obs[idx],
            "action":            self.action[idx],
            "action_is_pad":     self.pad[idx],
        }


# ── Policy ─────────────────────────────────────────────────────────────────────

def build_policy(vocab_size: int) -> ACTPolicy:
    cfg = ACTConfig(
        input_features={
            "observation.state":             PolicyFeature(type=FeatureType.STATE, shape=(FLAT_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV,   shape=(1,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(vocab_size,)),
        },
        n_obs_steps=1,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        normalization_mapping={
            "VISUAL":  NormalizationMode.MEAN_STD,
            "STATE":   NormalizationMode.IDENTITY,
            "ENV":     NormalizationMode.IDENTITY,
            "ACTION":  NormalizationMode.IDENTITY,
        },
        use_vae=True,
        dim_model=256,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=4,
        n_vae_encoder_layers=2,
        latent_dim=8,
        dropout=0.1,
        kl_weight=KL_WEIGHT,
    )
    return ACTPolicy(cfg)


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    all_frames = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            data = json.load(f)
        file_frames = data["frame_data"]
        eps = split_episodes(file_frames)
        ep_frames = sum(len(e) for e in eps)
        print(f"{filename}: {len(file_frames)} raw -> {ep_frames} cleaned -> {len(eps)} episodes")
        all_frames += file_frames

    vocab, input_ints = build_input_vocab(all_frames)
    vocab_size = len(input_ints)
    print(f"\nVocab size: {vocab_size} unique inputs")

    episodes = split_episodes(all_frames)
    print(f"Episodes: {len(episodes)}")

    dataset = ChunkDataset(episodes, vocab, vocab_size)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    policy   = build_policy(vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    for epoch in range(N_EPOCHS):
        policy.train()
        total_loss = total_l1 = total_kld = 0.0

        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            B = batch["observation.state"].shape[0]
            batch["observation.environment_state"] = torch.zeros(B, 1, device=DEVICE)
            loss, info = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l1   += info.get("l1_loss",  0.0)
            total_kld  += info.get("kld_loss", 0.0)

        scheduler.step()
        n = len(loader)
        print(f"Epoch {epoch+1:>2}/{N_EPOCHS}  "
              f"loss={total_loss/n:.4f}  "
              f"l1={total_l1/n:.4f}  "
              f"kld={total_kld/n:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save({
        "policy_state_dict": policy.state_dict(),
        "config":            policy.config,
        "n_obs_steps":       N_OBS_STEPS,
        "flat_dim":          FLAT_DIM,
        "vocab_size":        vocab_size,
        "vocab":             vocab,
        "input_ints":        input_ints,
        "chunk_size":        CHUNK_SIZE,
    }, SAVE_FILE)
    print(f"\nSaved to {SAVE_FILE}")


if __name__ == "__main__":
    main()
