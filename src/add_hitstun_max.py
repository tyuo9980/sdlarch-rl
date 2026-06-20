"""
For each game_state JSON in src/data/, extend hitstun from [val] to [val, max_val]
where max_val is the maximum hitstun[0] seen while the player's action_id is unchanged.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def process_file(path: str) -> int:
    with open(path) as f:
        data = json.load(f)

    frames = data["frame_data"]
    max_hitstun = {"p1": 0, "p2": 0}
    prev_hitstun = {"p1": 0, "p2": 0}

    for frame in frames:
        for player in ("p1", "p2"):
            hitstun = frame[player]["hitstun"]
            cur_hitstun = hitstun[0]
            if cur_hitstun == 0:
                max_hitstun[player] = 1
            elif cur_hitstun > max_hitstun[player]:
                max_hitstun[player] = cur_hitstun
            elif prev_hitstun[player] - cur_hitstun > 1:
                max_hitstun[player] = cur_hitstun

            prev_hitstun[player] = cur_hitstun

            hitstun[1] = max_hitstun[player]


    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    return len(frames)


def main():
    files = [
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.endswith(".json")
    ]
    print(f"Found {len(files)} files.")
    for path in sorted(files):
        n = process_file(path)
        print(f"  {os.path.basename(path)}: {n} frames processed")
    print("Done.")


if __name__ == "__main__":
    main()
