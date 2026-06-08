import json
import os
import sys
import threading
import time

import gymnasium as gym
import numpy as np
import torch as th
import vgamepad as vg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import HISTORY, N_BITS, _get_player_and_opponent, extract_obs

# ── Config ─────────────────────────────────────────────────────────────────────
BC_MODEL_PATH  = "models/bc_gamestate/bc_policy_gamestate.zip"
SAVE_DIR       = "models/ppo_sf6/"
TENSORBOARD    = "tensorboard/ppo_sf6/"
STREAM_PATH    = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

OBS_DIM        = 32 * HISTORY
MAX_STEPS      = 3600          # ~60s at 60fps, one long round
TOTAL_STEPS    = 5_000_000
N_STEPS        = 4096
BATCH_SIZE     = 256
LR             = 1e-5          # low LR to preserve BC pretraining
ENT_COEF       = 0.0
CLIP_RANGE     = 0.1

BIT_UP, BIT_DOWN, BIT_LEFT, BIT_RIGHT = 0, 1, 2, 3
BIT_SQUARE, BIT_TRIANGLE, BIT_R1      = 4, 5, 6
BIT_CROSS, BIT_CIRCLE, BIT_R2         = 7, 8, 9

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD, exist_ok=True)


# ── Environment ────────────────────────────────────────────────────────────────

class SF6Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = gym.spaces.MultiBinary(N_BITS)

        self._gamepad   = vg.VX360Gamepad()
        self._prev_bits = set()

        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self._frame_event  = threading.Event()
        self._frame_buf    = []
        self._prev_frame   = None
        self._step_count   = 0

        threading.Thread(target=self._stream_reader, daemon=True).start()

    # ── Stream ─────────────────────────────────────────────────────────────────

    def _stream_reader(self):
        while not os.path.exists(STREAM_PATH):
            time.sleep(0.5)
        with open(STREAM_PATH, "r", errors="ignore") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.001)
                    continue
                try:
                    frame = json.loads(line.strip())
                    with self._frame_lock:
                        self._latest_frame = frame
                    self._frame_event.set()
                except json.JSONDecodeError:
                    pass

    def _next_frame(self, timeout=0.2):
        self._frame_event.wait(timeout)
        self._frame_event.clear()
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    # ── Gamepad ────────────────────────────────────────────────────────────────

    def _apply_bits(self, bits):
        current = set()
        mapping = [
            (BIT_UP,       vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP),
            (BIT_DOWN,     vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN),
            (BIT_LEFT,     vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT),
            (BIT_RIGHT,    vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT),
            (BIT_SQUARE,   vg.XUSB_BUTTON.XUSB_GAMEPAD_X),
            (BIT_TRIANGLE, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y),
            (BIT_R1,       vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER),
            (BIT_CROSS,    vg.XUSB_BUTTON.XUSB_GAMEPAD_A),
            (BIT_CIRCLE,   vg.XUSB_BUTTON.XUSB_GAMEPAD_B),
        ]
        for bit, btn in mapping:
            if bits[bit]:
                current.add(btn)
        for b in current - self._prev_bits:
            self._gamepad.press_button(b)
        for b in self._prev_bits - current:
            self._gamepad.release_button(b)
        self._gamepad.right_trigger_float(1.0 if bits[BIT_R2] else 0.0)
        self._gamepad.update()
        self._prev_bits = current

    def _release_all(self):
        self._apply_bits(np.zeros(N_BITS, dtype=int))

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(self, frame):
        if self._prev_frame is None:
            return 0.0
        player, opponent = _get_player_and_opponent(frame)
        prev_p, prev_opp = _get_player_and_opponent(self._prev_frame)
        dmg_dealt = (prev_opp["hp"] - opponent["hp"]) / 10000.0
        dmg_taken = (prev_p["hp"]   - player["hp"])   / 10000.0
        return dmg_dealt - dmg_taken

    def _is_round_start(self, frame):
        p1, p2 = frame["p1"], frame["p2"]
        return (p1["hp"] >= 10000 and p2["hp"] >= 10000
                and abs(p1["pos"]["x"]) == 150 and abs(p2["pos"]["x"]) == 150
                and p1["action"][0] == 1 and p2["action"][0] == 1)

    def _is_round_over(self, frame):
        return frame["p1"]["hp"] <= 0 or frame["p2"]["hp"] <= 0

    # ── Gym ────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_all()
        self._frame_buf  = []
        self._prev_frame = None
        self._step_count = 0

        while True:
            frame = self._next_frame(timeout=2.0)
            if frame and self._is_round_start(frame):
                break

        self._frame_buf  = [frame]
        self._prev_frame = frame
        return extract_obs(self._frame_buf), {}

    def step(self, action):
        bits = np.array(action, dtype=int)

        # un-mirror direction bits if Cammy is on the right side
        if self._prev_frame is not None:
            player, _ = _get_player_and_opponent(self._prev_frame)
            if not bool(player["dir"]):
                bits[BIT_LEFT], bits[BIT_RIGHT] = bits[BIT_RIGHT], bits[BIT_LEFT]

        self._apply_bits(bits)

        frame = self._next_frame()
        if frame is None:
            return extract_obs(self._frame_buf), 0.0, False, False, {}

        reward     = self._compute_reward(frame)
        terminated = self._is_round_over(frame)
        self._step_count += 1
        truncated  = self._step_count >= MAX_STEPS

        self._prev_frame = frame
        self._frame_buf.append(frame)
        if len(self._frame_buf) > HISTORY:
            self._frame_buf.pop(0)

        return extract_obs(self._frame_buf), reward, terminated, truncated, {}


# ── Checkpoint callback ────────────────────────────────────────────────────────

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_dir):
        super().__init__()
        self.save_freq = save_freq
        self.save_dir  = save_dir

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"ppo_sf6_{self.num_timesteps}")
            self.model.save(path)
            print(f"Saved {path}")
        return True


# ── Main ───────────────────────────────────────────────────────────────────────

def linear_schedule(initial_lr):
    return lambda progress: progress * initial_lr

def main():
    env = SF6Env()

    latest = max(
        (os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".zip")),
        key=os.path.getmtime,
        default=None,
    ) if os.path.isdir(SAVE_DIR) else None

    if latest:
        print(f"Resuming from {latest}")
        model = PPO.load(
            latest,
            env=env,
            verbose=1,
            tensorboard_log=TENSORBOARD,
            learning_rate=linear_schedule(LR),
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
        )
    else:
        print(f"Initializing from BC weights: {BC_MODEL_PATH}")
        bc = ActorCriticPolicy.load(BC_MODEL_PATH)

        model = PPO(
            policy=bc.__class__,
            env=env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=1,
            tensorboard_log=TENSORBOARD,
            learning_rate=linear_schedule(LR),
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
        )
        model.policy.load_state_dict(bc.state_dict(), strict=False)
        print("BC weights loaded into PPO policy.")

    callback = CheckpointCallback(save_freq=5000, save_dir=SAVE_DIR)
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False, callback=callback)
    model.save(os.path.join(SAVE_DIR, "ppo_sf6_final"))
    print("Done.")


if __name__ == "__main__":
    main()
