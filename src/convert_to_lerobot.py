"""
One-time conversion: JSON game-state recordings -> LeRobotDataset on disk.

Run from the project root:
    python src/convert_to_lerobot.py

Output: data/lerobot_sf6/ (used by train_lerobot_gamestate.py)
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils import (
    DATA_DIR,
    FLOAT_DIM,
    MAX_ACTION_ID,
    N_BITS,
    extract_input,
    frame_features,
    split_episodes,
)

LEROBOT_DATA_DIR = "data/lerobot_sf6"
REPO_ID          = "local/sf6-cammy"
FPS              = 60
STATE_DIM        = FLOAT_DIM + 2   # 18 floats + 2 normalised action IDs


def frame_to_state(floats: list, action_ids: list) -> np.ndarray:
    norm_ids = [a / MAX_ACTION_ID for a in action_ids]
    return np.array(floats + norm_ids, dtype=np.float32)


def main():
    root = Path(LEROBOT_DATA_DIR)
    if root.exists():
        print(f"Removing existing dataset at {root}")
        shutil.rmtree(root)

    # ── Load and segment all JSON recordings ───────────────────────────────────
    all_frames = []
    episodes = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            data = json.load(f)
        file_frames = data["frame_data"]
        eps = split_episodes(file_frames)
        episodes.extend(eps)
        ep_frames = sum(len(e) for e in eps)
        print(f"{filename}: {len(file_frames)} raw -> {ep_frames} after cleaning -> {len(eps)} episodes")
        all_frames += file_frames

    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Total frames:   {sum(len(e) for e in episodes)}")

    # ── Create dataset ─────────────────────────────────────────────────────────
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action":            {"dtype": "float32", "shape": (N_BITS,),    "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=root,
        use_videos=False,
    )

    # ── Write episodes ─────────────────────────────────────────────────────────
    for ep_idx, ep in enumerate(episodes):
        for frame in ep:
            floats, action_ids = frame_features(frame)
            ds.add_frame({
                "observation.state": frame_to_state(floats, action_ids),
                "action":            extract_input(frame),
                "task":              "play sf6",
            })
        ds.save_episode()
        if (ep_idx + 1) % 10 == 0:
            print(f"  saved {ep_idx + 1}/{len(episodes)} episodes")

    ds.finalize()
    print(f"\nDone. {ds.num_episodes} episodes, {ds.num_frames} frames -> {root}")


if __name__ == "__main__":
    main()
