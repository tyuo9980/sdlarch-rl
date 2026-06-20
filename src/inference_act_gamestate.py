import argparse
import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch as th

from train_act_gamestate import CHUNK_LEN, SEQ_LEN, ACTPolicy
from train_act_gamestate import SAVE_FILE as MODEL_PATH
from utils import (
    _get_player_and_opponent,
    canonicalize_action_ids,
    extract_input,
    frame_features,
    split_episodes,
)

DATA_PATH   = "data/game_state_1781293343.json"
#DATA_PATH   = "data/dr_link.json"
STREAM_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

checkpoint = th.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = ACTPolicy(
    n_classes=checkpoint["n_classes"],
    n_action_ids=checkpoint["n_action_ids"],
).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.input_ints      = checkpoint["input_ints"]
model.action_id_vocab = checkpoint["action_id_vocab"]
model.eval()
#model = th.compile(model, mode="reduce-overhead")
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
        return "N"
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

    def __init__(self, act_model: ACTPolicy, replan_interval: int = REPLAN_INTERVAL,
                 temperature: float = 1.0):
        self.model               = act_model
        self.replan_interval     = replan_interval
        self.temperature         = temperature
        self.obs_window          = deque(maxlen=SEQ_LEN)
        self.chunk_buffer        = deque()
        self.frames_since_replan = 0
        self._prev_opp_action    = None
        self._opp_action         = None
        self._z                  = None

    def step(self, frame: dict) -> np.ndarray:
        """Push one frame, return the next predicted bit array (N_BITS,)."""
        canonicalize_action_ids([frame])
        floats, action_ids = frame_features(frame)
        self.obs_window.append((floats, action_ids))

        player, opponent = _get_player_and_opponent(frame)
        opp_action = opponent["action"][0]
        action_changed = (self._prev_opp_action is not None and abs(opp_action - self._prev_opp_action) > 1)
        #print(opp_action)
        self._prev_opp_action = opp_action
        self._opp_action      = opp_action

        if action_changed and ((200 <= opp_action < 300) or (600 <= opp_action < 700) or (30 <= opp_action < 40)):
            self.chunk_buffer.clear()
            self._z = None

        if not self.chunk_buffer or self.frames_since_replan >= self.replan_interval:
            self._refill()

        self.frames_since_replan += 1
        bits = self.chunk_buffer.popleft()
        mirror = not bool(player["dir"])
        if mirror:
            bits[2], bits[3] = bits[3], bits[2]
        return bits

    def _refill(self):
        while len(self.obs_window) < SEQ_LEN:
            self.obs_window.appendleft(self.obs_window[0])

        obs_floats  = np.array([f for f, _ in self.obs_window], dtype=np.float32)
        obs_ids_raw = np.array([ids for _, ids in self.obs_window], dtype=np.int64)
        carry_z     = self._z if (self._opp_action is not None and 200 <= self._opp_action <= 299) else None
        self.chunk_buffer        = deque(self.model.predict_chunk(obs_floats, obs_ids_raw, self.temperature, z=carry_z))
        self._z                  = self.model.last_z
        self.frames_since_replan = 0


# ── Offline evaluation ─────────────────────────────────────────────────────────

def run_offline():
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]
    canonicalize_action_ids(frames)

    agent   = ACTInference(model)
    correct = 0
    times   = []
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    for frame in frames:
        t0   = time.perf_counter()
        bits = agent.step(frame)
        times.append((time.perf_counter() - t0) * 1000)

        predicted_int = _bits_to_int(bits)
        actual_bits   = extract_input(frame).astype(int)
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


# ── Round-start probe ─────────────────────────────────────────────────────────

def run_roundstart(n_frames: int = 20, temperature: float = 0.0):
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]
    canonicalize_action_ids(frames)

    episodes = split_episodes(frames)
    if not episodes:
        print("No episodes found.")
        return

    ep    = episodes[0]
    agent = ACTInference(model, temperature=temperature)

    print(f"Episode 0: {len(ep)} frames total. Showing first {n_frames}:\n")
    print(f"  {'frame':>5}  {'predicted':<22}  {'actual':<22}  match")
    print(f"  {'-'*5}  {'-'*22}  {'-'*22}  -----")
    for i, frame in enumerate(ep[:n_frames]):
        print("p1", frame["p1"])
        print("p2", frame["p2"])
        bits          = agent.step(frame)
        predicted_int = _bits_to_int(bits)
        actual_int    = _bits_to_int(extract_input(frame).astype(int))
        match         = "✓" if predicted_int == actual_int else "✗"
        print(f"  {i:>5}  {_int_to_label(predicted_int):<22}  {_int_to_label(actual_int):<22}  {match}")


# ── Live stream ────────────────────────────────────────────────────────────────

def run_stream():
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

            bits = agent.step(frame)
            _ = _bits_to_int(bits)


def run_probe(episode_idx: int = 0, frame_idx: int = 0, n_samples: int = 5, temperature: float = 1.0):
    """Feed a single frame into the model and show N different sampled chunks.

    Because the CVAE samples z ~ N(0,1) at inference, each call can produce a
    different action sequence from the same game state. This lets you quickly
    check whether the model has learned something meaningful or is still stuck
    predicting neutral for every frame.
    """
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]
    canonicalize_action_ids(frames)

    episodes = split_episodes(frames)
    if not episodes:
        print("No episodes found.")
        return

    ep_idx = min(episode_idx, len(episodes) - 1)
    ep     = episodes[ep_idx]
    frame  = ep[min(frame_idx, len(ep) - 1)]
    print(f"Episode {ep_idx}/{len(episodes)-1}  ({len(ep)} frames)  |  Frame {min(frame_idx, len(ep)-1)}")

    floats, action_ids = frame_features(frame)
    obs_floats  = np.array([floats],      dtype=np.float32)   # (1, FLOAT_DIM)
    obs_ids_raw = np.array([action_ids],  dtype=np.int64)     # (1, 2)

    player, opponent = _get_player_and_opponent(frame)
    print(f"Frame {frame_idx}:")
    print(f"  P hp={player.get('hp','-')}  meter={player.get('drive','-')}  "
          f"action_id={player.get('action','-')}  pos={player.get('pos','-')}  "
          f"hitstun={player.get('hitstun', '-')}  blockstun={player.get('blockstun','-')}")
    print(f"  O hp={opponent.get('hp','-')}  meter={opponent.get('drive','-')}  "
          f"action_id={opponent.get('action','-')}  pos={opponent.get('pos','-')}  "
          f"hitstun={opponent.get('hitstun', '-')}  blockstun={opponent.get('blockstun','-')}")
    actual_int = _bits_to_int(extract_input(frame).astype(int))
    print(f"  Actual input this frame: {_int_to_label(actual_int)}\n")

    for s in range(n_samples):
        chunk = model.predict_chunk(obs_floats, obs_ids_raw, temperature=temperature)
        parts = [_int_to_label(_bits_to_int(b)) for b in chunk]
        print(f"  Sample {s+1}: {' | '.join(parts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true",
                        help="Read live frames from sf6_stream.txt instead of the static JSON file")
    parser.add_argument("--roundstart", action="store_true",
                        help="Show predicted vs actual inputs for the first 20 frames of episode 0")
    parser.add_argument("--probe", action="store_true",
                        help="Feed a single frame and show N sampled chunks from the CVAE prior")
    parser.add_argument("--episode", type=int, default=0,
                        help="Which episode to probe (default: 0)")
    parser.add_argument("--frame-idx", type=int, default=0,
                        help="Which frame in the episode to probe (default: 0)")
    parser.add_argument("--samples", type=int, default=10,
                        help="How many independent CVAE samples to show with --probe (default: 5)")
    parser.add_argument("--frames", type=int, default=20,
                        help="Number of frames to show with --roundstart (default: 20)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature — 0=greedy/deterministic, >0=sample from distribution")
    args = parser.parse_args()

    if args.stream:
        run_stream()
    elif args.roundstart:
        run_roundstart(n_frames=args.frames, temperature=args.temperature)
    elif args.probe:
        run_probe(episode_idx=args.episode, frame_idx=args.frame_idx, n_samples=args.samples, temperature=args.temperature)
    else:
        run_offline()
