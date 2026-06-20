import argparse
import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch as th
from lerobot.policies.act.modeling_act import ACTPolicy as LeRobotACTPolicy

from train_lerobot_gamestate import SAVE_FILE as MODEL_PATH
from train_lerobot_gamestate import _build_obs
from utils import N_BITS, _mirror_input, extract_input, frame_features, split_episodes

DATA_PATH   = "data/game_state_1780768383.json"
STREAM_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

checkpoint  = th.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
policy      = LeRobotACTPolicy(checkpoint["config"])
policy.load_state_dict(checkpoint["policy_state_dict"])
policy.to(DEVICE)
policy.eval()
N_OBS_STEPS = checkpoint["n_obs_steps"]
INPUT_INTS  = checkpoint["input_ints"]   # list[int]: class_idx → input integer
print(f"Loaded LeRobot ACT model (n_obs_steps={N_OBS_STEPS}, vocab_size={checkpoint['vocab_size']})")


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

class LeRobotACTInference:
    """Wraps lerobot's ACTPolicy for frame-by-frame inference.

    Maintains a rolling history window of N_OBS_STEPS frames, flattened into
    the FLAT_DIM observation vector the policy expects. Chunk buffering and
    replanning are handled internally by policy.select_action().
    """

    def __init__(self, act_policy: LeRobotACTPolicy, n_obs_steps: int = N_OBS_STEPS):
        self.policy      = act_policy
        self.n_obs_steps = n_obs_steps
        self.history     = deque(maxlen=n_obs_steps)
        act_policy.reset()

    def step(self, frame: dict, mirror: bool = False) -> np.ndarray:
        """Push one frame, return the next predicted bit array (N_BITS,)."""
        floats, action_ids = frame_features(frame)
        self.history.append((floats, action_ids))

        obs_flat = _build_obs(list(self.history))
        batch = {
            "observation.state":             th.tensor(obs_flat, dtype=th.float32).unsqueeze(0).to(DEVICE),
            "observation.environment_state": th.zeros(1, 1, dtype=th.float32).to(DEVICE),
        }
        with th.no_grad():
            action = self.policy.select_action(batch)   # (1, vocab_size)

        raw      = action[0].cpu().numpy()
        cls      = int(np.argmax(raw))
        val      = INPUT_INTS[cls]
        if getattr(self, "debug", False):
            print(f"  class={cls}  input={val}  top_score={raw[cls]:.3f}")
        if mirror:
            val = _mirror_input(val)
        bits = np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int)
        return bits


# ── Offline evaluation ─────────────────────────────────────────────────────────

def run_offline():
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]

    agent   = LeRobotACTInference(policy)
    correct = 0
    times   = []
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    for frame in frames:
        t0   = time.perf_counter()
        bits = agent.step(frame, mirror=False)
        times.append((time.perf_counter() - t0) * 1000)

        predicted_int = _bits_to_int(bits)
        actual_int    = _bits_to_int(extract_input(frame).astype(int))

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


# ── Round-start probe ──────────────────────────────────────────────────────────

def run_roundstart(n_frames: int = 20):
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]

    episodes = split_episodes(frames)
    if not episodes:
        print("No episodes found.")
        return

    ep    = episodes[0]
    agent = LeRobotACTInference(policy)
    agent.debug = True

    print(f"Episode 0: {len(ep)} frames total. Showing first {n_frames}:\n")
    print(f"  {'frame':>5}  {'predicted':<22}  {'actual':<22}  match")
    print(f"  {'-'*5}  {'-'*22}  {'-'*22}  -----")
    for i, frame in enumerate(ep[:n_frames]):
        print("p1", frame["p1"])
        print("p2", frame["p2"])
        bits          = agent.step(frame, mirror=False)
        predicted_int = _bits_to_int(bits)
        actual_int    = _bits_to_int(extract_input(frame).astype(int))
        match         = "✓" if predicted_int == actual_int else "✗"
        print(f"  {i:>5}  {_int_to_label(predicted_int):<22}  {_int_to_label(actual_int):<22}  {match}")


# ── Live stream ────────────────────────────────────────────────────────────────

def run_stream():
    from utils import _get_player_and_opponent

    print(f"Waiting for stream file: {STREAM_PATH}")
    while not os.path.exists(STREAM_PATH):
        time.sleep(0.5)
    print(f"Using device: {DEVICE}")
    print("Stream detected. Running LeRobot ACT inference (Ctrl+C to stop)...\n")

    agent = LeRobotACTInference(policy)

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
            agent.step(frame, mirror=mirror)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true",
                        help="Read live frames from sf6_stream.txt instead of the static JSON file")
    parser.add_argument("--roundstart", action="store_true",
                        help="Show predicted vs actual inputs for the first N frames of episode 0")
    parser.add_argument("--frames", type=int, default=20,
                        help="Number of frames to show with --roundstart (default: 20)")
    args = parser.parse_args()

    if args.stream:
        run_stream()
    elif args.roundstart:
        run_roundstart(n_frames=args.frames)
    else:
        run_offline()
