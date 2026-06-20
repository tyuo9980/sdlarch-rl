import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils import (
    DATA_DIR,
    FLOAT_DIM,
    N_BITS,
    _canonical_action_id,
    _clean_input,
    _get_player_and_opponent,
    _mirror_input,
    build_action_id_vocab,
    build_input_vocab,
    canonicalize_action_ids,
    extract_input_index,
    frame_features,
    split_episodes,
)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_DIM        = 16
D_MODEL          = 128
NHEAD            = 4
NUM_ENC_LAYERS   = 1
NUM_DEC_LAYERS   = 4
DIM_FF           = 512
DROPOUT          = 0.1
LATENT_DIM  = 16      # CVAE latent dimensionality
KL_WEIGHT   = 0.01   # fixed KL penalty — small enough to prevent collapse
SEQ_LEN     = 1
CHUNK_LEN   = 32      # frames to predict per forward pass
STRIDE      = 1
MIN_ACTIVE  = 0       # minimum non-neutral frames in a chunk — filters out sparse bursts
ANIM_LEAD_IN = 20    # only start chunks in the last N frames of Cammy's animation
BATCH_SIZE  = 128
N_EPOCHS    = 30
LR          = 1e-4
SAVE_PATH   = "models/act_gamestate/"
SAVE_FILE   = os.path.join(SAVE_PATH, "act_policy.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(SAVE_PATH, exist_ok=True)


# ── Positional encoding ────────────────────────────────────────────────────────

class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── CVAE encoder ───────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """Encodes the target action chunk into a latent distribution (mu, log_var).

    Only used during training. At inference the prior N(0,1) is sampled instead,
    allowing the same observation to produce different action sequences.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.embed   = nn.Embedding(n_classes, D_MODEL)
        self.pos_enc = SinusoidalPosEnc(D_MODEL, max_len=CHUNK_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.mu_head = nn.Linear(D_MODEL, LATENT_DIM)
        self.lv_head = nn.Linear(D_MODEL, LATENT_DIM)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """y: (B, CHUNK_LEN) class indices → (mu, log_var) each (B, LATENT_DIM)"""
        x = self.pos_enc(self.embed(y))   # (B, CHUNK_LEN, D_MODEL)
        x = self.encoder(x).mean(dim=1)   # (B, D_MODEL)
        return self.mu_head(x), self.lv_head(x)


# ── Policy ─────────────────────────────────────────────────────────────────────

class ACTPolicy(nn.Module):
    def __init__(self, n_classes: int, n_action_ids: int):
        super().__init__()
        self.n_classes       = n_classes
        self.input_ints      = []
        self.action_id_vocab = {}

        # 15 scalar features → one token each via shared direction+bias params
        # scalars: indices 0:4 and 7:18 (everything except stance at 4:7)
        self.scalar_dir   = nn.Parameter(torch.randn(15, D_MODEL) * 0.02)
        self.scalar_bias  = nn.Parameter(torch.zeros(15, D_MODEL))
        # stance stays grouped (one-hot, only one active per frame)
        self.proj_stance  = nn.Linear(3, D_MODEL)
        self.action_embed = nn.Embedding(n_action_ids, D_MODEL)   # player & opp action IDs

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=NUM_ENC_LAYERS, enable_nested_tensor=False)

        # CVAE
        self.cvae_enc = CVAEEncoder(n_classes)
        self.z_proj   = nn.Linear(D_MODEL + LATENT_DIM, D_MODEL)

        # autoregressive decoder — target sequence is embedded and fed back as
        # input; position i sees tokens 0..i-1 via causal self-attention
        self.bos_idx  = n_classes                              # special start-of-chunk token
        self.tgt_embed = nn.Embedding(n_classes + 1, D_MODEL)  # +1 for BOS
        self.tgt_pos   = SinusoidalPosEnc(D_MODEL, max_len=CHUNK_LEN + 1)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=NUM_DEC_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, n_classes)

    def _encode_obs(self, x_float: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        # 15 scalar tokens: indices 0:4 and 7:18 (skip stance at 4:7)
        scalars = torch.cat([x_float[..., 0:4], x_float[..., 7:18]], dim=-1)  # (B, S, 15)
        scalar_tokens = scalars.unsqueeze(-1) * self.scalar_dir + self.scalar_bias  # (B, S, 15, D_MODEL)

        # stance token + 2 action_id tokens → (B, S, 3, D_MODEL)
        extra = torch.stack([
            self.proj_stance (x_float[..., 4:7]),
            self.action_embed(x_ids[..., 0]),
            self.action_embed(x_ids[..., 1]),
        ], dim=-2)

        tokens = torch.cat([scalar_tokens, extra], dim=-2)  # (B, S, 18, D_MODEL)
        B, S, G, D = tokens.shape
        tokens = tokens.view(B, S * G, D)                   # (B, SEQ_LEN*18, D_MODEL)
        return self.encoder(tokens).mean(dim=1)              # (B, D_MODEL)

    def _build_memory(self, x_float: torch.Tensor, x_ids: torch.Tensor,
                      z: torch.Tensor) -> torch.Tensor:
        obs    = self._encode_obs(x_float, x_ids)
        return self.z_proj(torch.cat([obs, z], dim=-1)).unsqueeze(1)   # (B, 1, D_MODEL)

    def forward(
        self,
        x_float: torch.Tensor,
        x_ids:   torch.Tensor,
        y:       torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training only — uses teacher forcing.
        x_float : (B, SEQ_LEN, FLOAT_DIM)
        x_ids   : (B, SEQ_LEN, 2)
        y       : (B, CHUNK_LEN) ground-truth class indices
        Returns  : (logits (B, CHUNK_LEN, N_CLASSES), kld scalar)
        """
        obs = self._encode_obs(x_float, x_ids)                              # (B, D_MODEL)
        mu, log_var = self.cvae_enc(y)
        z   = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        memory = self.z_proj(torch.cat([obs, z], dim=-1)).unsqueeze(1)      # (B, 1, D_MODEL)

        # teacher forcing: input is [BOS, y[0], y[1], ..., y[CHUNK_LEN-2]]
        # so position i predicts y[i] having seen y[0..i-1]
        B   = y.shape[0]
        bos = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=y.device)
        tgt_in  = torch.cat([bos, y[:, :-1]], dim=1)                        # (B, CHUNK_LEN)
        tgt_emb = self.tgt_pos(self.tgt_embed(tgt_in))                      # (B, CHUNK_LEN, D_MODEL)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(CHUNK_LEN, device=y.device)
        decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask, tgt_is_causal=True)
        logits  = self.out_proj(decoded)                                     # (B, CHUNK_LEN, N_CLASSES)
        return logits, kld

    def predict_chunk(
        self,
        obs_floats: np.ndarray,
        obs_ids_raw: np.ndarray,
        temperature: float = 0,
        z: "torch.Tensor | None" = None,
    ) -> list:
        """
        Inference — autoregressive with KV-cache.
        Each step only processes one new token instead of the full growing sequence,
        reducing decoder work from O(n²) to O(n) across the chunk.
        obs_floats  : (SEQ_LEN, FLOAT_DIM)
        obs_ids_raw : (SEQ_LEN, 2)  raw action IDs
        temperature : 0 = greedy argmax, >0 = sample from softmax
        z           : optional latent to reuse (carry across replans mid-combo)
        """
        mapped = np.array(
            [[self.action_id_vocab.get(_canonical_action_id(int(a)), 0) for a in row]
             for row in obs_ids_raw],
            dtype=np.int64,
        )
        device = next(self.parameters()).device
        xf = torch.tensor(obs_floats, dtype=torch.float32).unsqueeze(0).to(device)
        xi = torch.tensor(mapped,     dtype=torch.long   ).unsqueeze(0).to(device)

        H  = NHEAD
        DH = D_MODEL // H

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B, S, _ = t.shape
            return t.view(B, S, H, DH).transpose(1, 2)   # (B, H, S, DH)

        def merge_heads(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous().view(t.shape[0], -1, D_MODEL)

        with torch.no_grad():
            if z is None:
                z = torch.randn(1, LATENT_DIM, device=device)
            self.last_z = z
            memory = self._build_memory(xf, xi, z)           # (1, 1, D_MODEL)

            # Pre-project cross-attention K, V from memory once per layer (memory never changes)
            cross_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer in self.decoder.layers:
                W = layer.multihead_attn.in_proj_weight
                b = layer.multihead_attn.in_proj_bias
                ck = split_heads(F.linear(memory, W[D_MODEL:2*D_MODEL], b[D_MODEL:2*D_MODEL] if b is not None else None))
                cv = split_heads(F.linear(memory, W[2*D_MODEL:],        b[2*D_MODEL:]        if b is not None else None))
                cross_kv.append((ck, cv))

            # Self-attention KV caches — one (k, v) pair per decoder layer, grows each step
            self_kv: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [(None, None)] * len(self.decoder.layers)

            tgt_buf   = torch.full((1, CHUNK_LEN + 1), self.bos_idx, dtype=torch.long, device=device)
            all_probs = torch.zeros(CHUNK_LEN, self.n_classes, device=device)

            for step in range(CHUNK_LEN):
                # Embed the single token for this step and add its positional encoding
                x = self.tgt_embed(tgt_buf[:, step]).unsqueeze(1)   # (1, 1, D_MODEL)
                x = x + self.tgt_pos.pe[:, step:step + 1]           # position `step`

                new_self_kv = []
                for i, layer in enumerate(self.decoder.layers):
                    # ── Self-attention (norm_first) ───────────────────────────
                    residual = x
                    x = layer.norm1(x)
                    W = layer.self_attn.in_proj_weight
                    b = layer.self_attn.in_proj_bias
                    q  = split_heads(F.linear(x, W[:D_MODEL],           b[:D_MODEL]           if b is not None else None))
                    k  = split_heads(F.linear(x, W[D_MODEL:2*D_MODEL],  b[D_MODEL:2*D_MODEL]  if b is not None else None))
                    v  = split_heads(F.linear(x, W[2*D_MODEL:],         b[2*D_MODEL:]         if b is not None else None))
                    pk, pv = self_kv[i]
                    if pk is not None:
                        k = torch.cat([pk, k], dim=2)
                        v = torch.cat([pv, v], dim=2)
                    new_self_kv.append((k, v))
                    # no mask needed — cache only holds past tokens, so query sees 0..step
                    attn_out = F.scaled_dot_product_attention(q, k, v)
                    x = residual + F.linear(merge_heads(attn_out),
                                            layer.self_attn.out_proj.weight,
                                            layer.self_attn.out_proj.bias)

                    # ── Cross-attention ───────────────────────────────────────
                    residual = x
                    x = layer.norm2(x)
                    W = layer.multihead_attn.in_proj_weight
                    b = layer.multihead_attn.in_proj_bias
                    q = split_heads(F.linear(x, W[:D_MODEL], b[:D_MODEL] if b is not None else None))
                    ck, cv = cross_kv[i]
                    attn_out = F.scaled_dot_product_attention(q, ck, cv)
                    x = residual + F.linear(merge_heads(attn_out),
                                            layer.multihead_attn.out_proj.weight,
                                            layer.multihead_attn.out_proj.bias)

                    # ── FFN ───────────────────────────────────────────────────
                    residual = x
                    x = layer.norm3(x)
                    x = layer.linear2(layer.activation(layer.linear1(x)))
                    x = residual + x

                self_kv = new_self_kv

                logit = self.out_proj(x[:, 0, :])         # (1, N_CLASSES)
                probs = torch.softmax(logit, dim=-1)
                all_probs[step] = probs[0]

                if temperature <= 0:
                    next_cls = logit.argmax(dim=-1)
                else:
                    next_cls = torch.multinomial(probs / probs.sum(), num_samples=1).squeeze(-1)
                tgt_buf[0, step + 1] = next_cls

        class_idxs = tgt_buf[0, 1:].tolist()
        self.last_probs = all_probs[range(CHUNK_LEN), class_idxs].tolist()
        result = []
        for idx in class_idxs:
            val = _clean_input(self.input_ints[idx])
            result.append(np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int))
        return result


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    def __init__(self, episodes: list, vocab: dict, action_id_vocab: dict):
        self.chunks = []

        for ep in episodes:
            if len(ep) < CHUNK_LEN + 1:
                continue

            splits  = [frame_features(f) for f in ep]
            floats  = np.array([s[0] for s in splits], dtype=np.float32)
            ids     = np.array([[action_id_vocab.get(a, 0) for a in s[1]] for s in splits], dtype=np.int64)
            actions = np.array([extract_input_index(f, vocab) for f in ep], dtype=np.int64)
            n       = len(actions)

            event_frames = [0]
            last_player, last_opp = _get_player_and_opponent(ep[0])
            for t in range(1, n - 1):
                player, opponent = _get_player_and_opponent(ep[t])
                #input_changed = actions[t] != actions[t - 1]
                p_in_hitstun = player["hitstun"][0] == player["hitstun"][1] and player["hitstun"][1] != 0
                p_in_blockstun = player["blockstun"][0] == player["blockstun"][1] and player["blockstun"][1] != 0
                op_in_hitstun = opponent["hitstun"][0] == opponent["hitstun"][1] and opponent["hitstun"][1] != 0
                op_in_blockstun = opponent["blockstun"][0] == opponent["blockstun"][1] and opponent["blockstun"][1] != 0
                player_id_changed = player["action"][0] != last_player["action"][0]
                opp_id_changed = opponent["action"][0] != last_opp["action"][0]
                if player_id_changed or opp_id_changed or p_in_hitstun or p_in_blockstun or op_in_hitstun or op_in_blockstun:
                    event_frames.append(t)

                last_player, last_opp = player, opponent

            for t in event_frames:
                if actions[t] == 0:
                    continue   # skip neutral starting frames

                player, _ = _get_player_and_opponent(ep[t])
                action_id, anim_frame, anim_total = player["action"]
                if action_id > 1024 and anim_total > 240 and anim_total - anim_frame > ANIM_LEAD_IN:
                    continue   # skip early frames of long/looping animations only

                tgt_end    = min(n, t + CHUNK_LEN)
                y          = actions[t : tgt_end]
                actual_tgt = len(y)

                if np.count_nonzero(y) < MIN_ACTIVE:
                    continue   # skip sparse chunks to avoid oscillation

                obs_start  = max(0, t - SEQ_LEN + 1)
                xf         = floats[obs_start : t + 1]
                xi         = ids[obs_start : t + 1]
                actual_obs = len(xf)

                if actual_obs < SEQ_LEN:
                    pad = SEQ_LEN - actual_obs
                    xf  = np.vstack([np.zeros((pad, FLOAT_DIM), dtype=np.float32), xf])
                    xi  = np.vstack([np.zeros((pad, 2),         dtype=np.int64),   xi])

                tgt_mask = np.zeros(CHUNK_LEN, dtype=np.float32)
                tgt_mask[:actual_tgt] = 1.0
                if actual_tgt < CHUNK_LEN:
                    y = np.concatenate([y, np.zeros(CHUNK_LEN - actual_tgt, dtype=np.int64)])

                self.chunks.append((
                    torch.tensor(xf,       dtype=torch.float32),
                    torch.tensor(xi,       dtype=torch.long),
                    torch.tensor(y,        dtype=torch.long),
                    torch.tensor(tgt_mask, dtype=torch.float32),
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
    canonicalize_action_ids(frames)
    print(f"Total frames: {len(frames)}")

    vocab, input_ints             = build_input_vocab(frames)
    action_id_vocab, n_action_ids = build_action_id_vocab(frames)
    N_CLASSES = len(input_ints)
    print(f"Input vocabulary:     {N_CLASSES} distinct inputs")
    print(f"Action ID vocabulary: {n_action_ids} distinct action IDs")

    episodes = split_episodes(frames)
    print(f"Built {len(episodes)} episodes")

    dataset = ChunkDataset(episodes, vocab, action_id_vocab)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Total chunks: {len(dataset)}")

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    model = ACTPolicy(n_classes=N_CLASSES, n_action_ids=n_action_ids).to(DEVICE)
    model.input_ints      = input_ints
    model.action_id_vocab = action_id_vocab
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = total_ce = total_kld = 0.0

        for xf, xi, y, mask in loader:
            xf, xi, y, mask = xf.to(DEVICE), xi.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

            logits, kld = model(xf, xi, y=y)
            B, C_LEN, OUT = logits.shape

            ce   = loss_fn(logits.view(B * C_LEN, OUT), y.view(B * C_LEN))
            ce   = (ce.view(B, C_LEN) * mask).sum() / mask.sum().clamp(min=1.0)
            loss = ce + KL_WEIGHT * kld

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += ce.item()
            total_kld  += kld.item()

        scheduler.step()
        n = len(loader)
        print(f"Epoch {epoch+1:>2}/{N_EPOCHS}  "
              f"loss={total_loss/n:.4f}  "
              f"ce={total_ce/n:.4f}  "
              f"kld={total_kld/n:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

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
