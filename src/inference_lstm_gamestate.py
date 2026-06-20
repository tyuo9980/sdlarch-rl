import argparse
import json
import os
import time

import torch as th

from train_lstm_gamestate import SAVE_FILE as MODEL_PATH
from train_lstm_gamestate import LSTMPolicy
from utils import extract_input, frame_features

DATA_PATH   = "data/game_state_1780768383.json"
STREAM_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

checkpoint = th.load(MODEL_PATH, map_location=DEVICE)
model = LSTMPolicy(n_classes=checkpoint["n_classes"], n_action_ids=checkpoint["n_action_ids"]).to(DEVICE)
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
    bits = [i for i in range(len(_BIT_LABELS)) if (val >> i) & 1]
    dirs = [b for b in bits if b < 4]
    attacks = [_BIT_LABELS[b] for b in bits if b >= 4]
    if len(dirs) == 2 and tuple(dirs) in _DIAGONALS:
        parts = [_DIAGONALS[tuple(dirs)]] + attacks
    else:
        parts = [_BIT_LABELS[b] for b in dirs] + attacks
    return "+".join(parts)

def _print_frame(frame_num, actual_bits, actual_int, predicted_bits, predicted_int):
    match = predicted_int == actual_int
    #print(f"frame {frame_num:5d} | actual: {actual_int:4d} {actual_bits} | predicted: {predicted_int:4d} {predicted_bits} | {'OK' if match else '--'}")
    return match


def run_offline():
    from collections import defaultdict

    import numpy as np
    with open(DATA_PATH) as f:
        frames = json.load(f)["frame_data"]

    hidden  = None
    correct = 0
    times   = []
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    for i, frame in enumerate(frames):
        float_feats, action_ids = frame_features(frame)
        t0 = time.perf_counter()
        bits, hidden = model.predict(float_feats, action_ids, hidden)
        times.append((time.perf_counter() - t0) * 1000)
        predicted_int = _bits_to_int(bits)

        actual_bits = extract_input(frame).astype(int)
        actual_int  = _bits_to_int(actual_bits)

        match = _print_frame(i + 1, actual_bits, actual_int, bits, predicted_int)
        if match:
            correct += 1
            class_correct[actual_int] += 1
        class_total[actual_int] += 1

    total = len(frames)
    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    # neutral vs attack breakdown
    neutral_c = class_correct[0]
    neutral_t = class_total[0]
    attack_c  = correct - neutral_c
    attack_t  = total - neutral_t
    print(f"  Neutral (input=0): {neutral_c}/{neutral_t} ({100*neutral_c/max(neutral_t,1):.1f}%)")
    print(f"  Attack  (input≠0): {attack_c}/{attack_t} ({100*attack_c/max(attack_t,1):.1f}%)")

    # per-class breakdown sorted by frequency
    print("\nPer-class accuracy (top 20 by frequency):")
    print(f"  {'input':<20}  {'correct':>7}  {'total':>7}  {'acc':>6}")
    for val, t in sorted(class_total.items(), key=lambda x: -x[1])[:20]:
        c = class_correct[val]
        print(f"  {_int_to_label(val):<20}  {c:>7}  {t:>7}  {100*c/t:>5.1f}%")

    t = np.array(times)
    print(f"\nInference (ms) — mean: {t.mean():.3f}  min: {t.min():.3f}  p99: {np.percentile(t, 99):.3f}  max: {t.max():.3f}")


def run_stream():
    print(f"Waiting for stream file: {STREAM_PATH}")
    while not os.path.exists(STREAM_PATH):
        time.sleep(0.5)
    print(f"Using device: {DEVICE}")
    print("Stream detected. Running inference (Ctrl+C to stop)...\n")

    hidden    = None
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

            float_feats, action_ids = frame_features(frame)
            bits, hidden  = model.predict(float_feats, action_ids, hidden)
            predicted_int = _bits_to_int(bits)

            actual_bits = extract_input(frame).astype(int)
            actual_int  = _bits_to_int(actual_bits)

            frame_num += 1
            #_print_frame(frame_num, actual_bits, actual_int, bits, predicted_int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Read live frames from sf6_stream.txt instead of the static JSON file")
    args = parser.parse_args()

    if args.stream:
        run_stream()
    else:
        run_offline()
