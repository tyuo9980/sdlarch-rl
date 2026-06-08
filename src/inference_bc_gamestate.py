import argparse
import json
import os
import time

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from train_bc_gamestate import EmbeddingExtractor  # noqa: F401 — needed for policy load
from utils import HISTORY, N_BITS, _get_player_and_opponent, _mirror_input, extract_action, extract_obs_embed

MODEL_PATH  = "models/bc_gamestate/bc_policy_gamestate.zip"
VOCAB_PATH  = "models/bc_gamestate/bc_action_vocab.json"
DATA_PATH   = "data/game_state_1780768383.json"
STREAM_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

policy = ActorCriticPolicy.load(MODEL_PATH)
policy.to(DEVICE)
policy.eval()

with open(VOCAB_PATH) as f:
    input_ints = json.load(f)


def predict(buffer: list[dict]) -> tuple[np.ndarray, int]:
    frame = buffer[-1]
    player, _ = _get_player_and_opponent(frame)
    mirror = not bool(player["dir"])

    obs = extract_obs_embed(buffer)
    obs_tensor = th.tensor(obs).unsqueeze(0).to(DEVICE)
    with th.no_grad():
        class_idx = policy._predict(obs_tensor, deterministic=True).item()

    val = input_ints[int(class_idx)]
    if mirror:
        val = _mirror_input(val)
    bits = np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int)
    return bits, val


def _print_frame(frame_num, actual_bits, actual_int, predicted_bits, predicted_int):
    match = predicted_int == actual_int
    print(f"frame {frame_num:5d} | actual: {actual_int:4d} {actual_bits} | predicted: {predicted_int:4d} {predicted_bits} | {'OK' if match else '--'}")
    return match

def run_offline():
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]

    buffer  = []
    correct = 0
    times   = []

    for i, frame in enumerate(frames):
        buffer.append(frame)
        if len(buffer) > HISTORY:
            buffer.pop(0)

        t0 = time.perf_counter()
        predicted_bits, predicted_int = predict(buffer)
        times.append((time.perf_counter() - t0) * 1000)

        actual_bits = extract_action(frame).astype(int)
        actual_int  = sum(int(b) << i for i, b in enumerate(actual_bits))

        if _print_frame(i + 1, actual_bits, actual_int, predicted_bits, predicted_int):
            correct += 1

    total = len(frames)
    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    t = np.array(times)
    print(f"Inference (ms) — mean: {t.mean():.3f}  min: {t.min():.3f}  p99: {np.percentile(t, 99):.3f}  max: {t.max():.3f}")

def run_stream():
    print(f"Waiting for stream file: {STREAM_PATH}")
    while not os.path.exists(STREAM_PATH):
        time.sleep(0.5)
    print(f"Using device: {DEVICE}")
    print("Stream detected. Running inference (Ctrl+C to stop)...\n")

    buffer    = []
    frame_num = 0

    with open(STREAM_PATH, "r", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.001)
                f.seek(0, os.SEEK_END)
                continue
            clean = line.strip()
            if not clean:
                continue
            try:
                frame = json.loads(clean)
            except json.JSONDecodeError:
                continue

            buffer.append(frame)
            if len(buffer) > HISTORY:
                buffer.pop(0)

            predicted_bits, predicted_int = predict(buffer)
            actual_bits = extract_action(frame).astype(int)
            actual_int  = sum(int(b) << i for i, b in enumerate(actual_bits))

            frame_num += 1
            _print_frame(frame_num, actual_bits, actual_int, predicted_bits, predicted_int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Read live frames from sf6_stream.txt instead of the static JSON file")
    args = parser.parse_args()

    if args.stream:
        run_stream()
    else:
        run_offline()
