import argparse
import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch as th

from train_act_gamestate import CHUNK_LEN, FLOAT_DIM, SEQ_LEN, ACTPolicy
from train_act_gamestate import SAVE_FILE as MODEL_PATH
from utils import N_BITS, _mirror_input, extract_action, frame_features

DATA_PATH   = "data/game_state_1780768383.json"
STREAM_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

checkpoint = th.load(MODEL_PATH, map_location=DEVICE)
model = ACTPolicy(n_classes=checkpoint["n_classes"], n_action_ids=checkpoint["n_action_ids"]).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.input_ints      = checkpoint["input_ints"]
model.action_id_vocab = checkpoint["action_id_vocab"]
model.eval()
print(f"Loaded model: {checkpoint['n_classes']} action classes, {checkpoint['n_action_ids']} action IDs")


_BIT_LABELS = ["↑", "↓", "←", "→", "LP", "MP", "HP", "LK", "MK", "HK"]

def _bits_to_int(bits) -> int:
    return sum(int(b) << i for i, b in enumerate(bits))

_DIAGONALS = {
    (0, 3): "↗", (0, 2): "↖",
    (1, 3): "↘", (1, 2): "↙",
}

def _int_to_label(val: int) -> str:
    if val == 0:
        return "neutral"
    bits    = [i for i in range(len(_BIT_LABELS)) if (val >> i) & 1]
    dirs    = [b for b in bits if b < 4]
    attacks = [_BIT_LABELS[b] for b in bits if b >= 4]
    if len(dirs) == 2 and tuple(dirs) in _DIAGONALS:
        parts = [_DIAGONALS[tuple(dirs)]] + attacks
    else:
        parts = [_BIT_LABELS[b] for b in dirs] + attacks
    return "+".join(parts)


# ── Stateful inference helper ──────────────────────────────────────────────────

REPLAN_INTERVAL = CHUNK_LEN    # replan every chunk — predictions are fully consumed before replanning

class ACTInference:
    """Manages the rolling observation window and chunk buffer for live inference.

    The model predicts CHUNK_LEN frames at once (long enough to cover a full combo),
    but replans every replan_interval frames so neutral play stays responsive.
    """

    def __init__(self, act_model: ACTPolicy, replan_interval: int = REPLAN_INTERVAL):
        self.model               = act_model
        self.replan_interval     = replan_interval
        self.obs_window          = deque(maxlen=SEQ_LEN)
        self.chunk_buffer        = []
        self.frames_since_replan = 0

    def step(self, frame: dict, mirror: bool = False) -> np.ndarray:
        """Push one frame, return the next predicted bit array (N_BITS,)."""
        floats, action_ids = frame_features(frame)
        self.obs_window.append((floats, action_ids))

        if not self.chunk_buffer or self.frames_since_replan >= self.replan_interval:
            self._refill()

        self.frames_since_replan += 1
        bits = self.chunk_buffer.pop(0)
        if mirror:
            val  = _bits_to_int(bits)
            val  = _mirror_input(val)
            bits = np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int)
        return bits

    def _refill(self):
        window = list(self.obs_window)
        while len(window) < SEQ_LEN:
            window = [window[0]] + window

        obs_floats  = np.array([f for f, _ in window], dtype=np.float32)
        obs_ids_raw = np.array([ids for _, ids in window], dtype=np.int64)
        self.chunk_buffer        = self.model.predict_chunk(obs_floats, obs_ids_raw)
        self.frames_since_replan = 0


# ── Offline evaluation ─────────────────────────────────────────────────────────

def run_offline():
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]

    agent   = ACTInference(model)
    correct = 0
    times   = []
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    for i, frame in enumerate(frames):
        t0   = time.perf_counter()
        bits = agent.step(frame, mirror=False)
        times.append((time.perf_counter() - t0) * 1000)

        predicted_int = _bits_to_int(bits)
        actual_bits   = extract_action(frame).astype(int)
        actual_int    = _bits_to_int(actual_bits)

        if predicted_int == actual_int:
            correct += 1
            class_correct[actual_int] += 1
        class_total[actual_int] += 1

    total = len(frames)
    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    neutral_c = class_correct[0]
    neutral_t = class_total[0]
    attack_c  = correct - neutral_c
    attack_t  = total - neutral_t
    print(f"  Neutral (input=0): {neutral_c}/{neutral_t} ({100*neutral_c/max(neutral_t,1):.1f}%)")
    print(f"  Attack  (input≠0): {attack_c}/{attack_t} ({100*attack_c/max(attack_t,1):.1f}%)")

    print("\nPer-class accuracy (top 20 by frequency):")
    print(f"  {'input':<20}  {'correct':>7}  {'total':>7}  {'acc':>6}")
    for val, t in sorted(class_total.items(), key=lambda x: -x[1])[:20]:
        c = class_correct[val]
        print(f"  {_int_to_label(val):<20}  {c:>7}  {t:>7}  {100*c/t:>5.1f}%")

    t = np.array(times)
    print(f"\nInference (ms) — mean: {t.mean():.3f}  min: {t.min():.3f}  p99: {np.percentile(t, 99):.3f}  max: {t.max():.3f}")


# ── Live stream ────────────────────────────────────────────────────────────────

def run_stream():
    from utils import _get_player_and_opponent

    print(f"Waiting for stream file: {STREAM_PATH}")
    while not os.path.exists(STREAM_PATH):
        time.sleep(0.5)
    print(f"Using device: {DEVICE}")
    print("Stream detected. Running ACT inference (Ctrl+C to stop)...\n")

    agent = ACTInference(model)

    with open(STREAM_PATH, "r", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.001)
                continue
            clean = line.strip()
            if not clean:
                continue
            try:
                frame = json.loads(clean)
            except json.JSONDecodeError:
                continue

            player, _ = _get_player_and_opponent(frame)
            mirror    = not bool(player["dir"])
            bits      = agent.step(frame, mirror=mirror)
            _ = _bits_to_int(bits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true",
                        help="Read live frames from sf6_stream.txt instead of the static JSON file")
    args = parser.parse_args()

    if args.stream:
        run_stream()
    else:
        run_offline()
