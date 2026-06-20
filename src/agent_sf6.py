import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch as th
import vgamepad as vg
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference_act_gamestate import ACTInference
from inference_lerobot_gamestate import LeRobotACTInference
from train_act_gamestate import ACTPolicy
from train_bc_gamestate import EmbeddingExtractor  # noqa: F401 — needed for policy load
from train_lstm_gamestate import LSTMPolicy
from utils import (
    HISTORY,
    N_BITS,
    _get_player_and_opponent,
    _mirror_input,
    extract_obs_embed,
    frame_features,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BC_MODEL_PATH   = "models/bc_gamestate/bc_policy_gamestate.zip"
BC_VOCAB_PATH   = "models/bc_gamestate/bc_action_vocab.json"
LSTM_MODEL_PATH = "models/lstm_gamestate/lstm_policy.pt"
STREAM_PATH     = r"C:\Program Files (x86)\Steam\steamapps\common\Street Fighter 6\reframework\data\sf6_stream.txt"

# ── Bit → gamepad mapping ──────────────────────────────────────────────────────
BIT_UP       = 0
BIT_DOWN     = 1
BIT_LEFT     = 2
BIT_RIGHT    = 3
BIT_SQUARE   = 4    # lp
BIT_TRIANGLE = 5    # mp
BIT_R1       = 6    # hp
BIT_CROSS    = 7    # lk
BIT_CIRCLE   = 8    # mk
BIT_R2       = 9    # hk

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ── Gamepad ────────────────────────────────────────────────────────────────────
gamepad   = vg.VX360Gamepad()
prev_bits = set()

def apply_gamepad(bits: np.ndarray):
    global prev_bits

    current = set()
    if bits[BIT_UP]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
    if bits[BIT_DOWN]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
    if bits[BIT_LEFT]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
    if bits[BIT_RIGHT]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
    if bits[BIT_SQUARE]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    if bits[BIT_TRIANGLE]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
    if bits[BIT_R1]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    if bits[BIT_CROSS]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    if bits[BIT_CIRCLE]:
        current.add(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    for b in current - prev_bits:
        gamepad.press_button(b)
    for b in prev_bits - current:
        gamepad.release_button(b)

    gamepad.right_trigger_float(1.0 if bits[BIT_R2] else 0.0)
    gamepad.update()
    prev_bits = current


# ── BC inference ───────────────────────────────────────────────────────────────

def load_bc():
    print(f"Loading BC model from {BC_MODEL_PATH}...")
    policy = ActorCriticPolicy.load(BC_MODEL_PATH)
    policy.to(DEVICE)
    policy.eval()
    with open(BC_VOCAB_PATH) as f:
        input_ints = json.load(f)
    print(f"BC model loaded. ({len(input_ints)} action classes)")
    return policy, input_ints

def predict_bc(policy, input_ints: list, buffer: list[dict], mirror: bool = False) -> np.ndarray:
    obs = extract_obs_embed(buffer)
    obs_tensor = th.tensor(obs).unsqueeze(0).to(DEVICE)
    with th.no_grad():
        class_idx = policy._predict(obs_tensor, deterministic=True).item()
    val = input_ints[int(class_idx)]
    if mirror:
        val = _mirror_input(val)
    return np.array([(val >> i) & 1 for i in range(N_BITS)], dtype=int)


# ── LSTM inference ─────────────────────────────────────────────────────────────

def load_lstm():
    print(f"Loading LSTM model from {LSTM_MODEL_PATH}...")
    checkpoint   = th.load(LSTM_MODEL_PATH, map_location=DEVICE)
    model        = LSTMPolicy(n_classes=checkpoint["n_classes"], n_action_ids=checkpoint["n_action_ids"]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.input_ints      = checkpoint["input_ints"]
    model.action_id_vocab = checkpoint["action_id_vocab"]
    model.eval()
    model = th.compile(model, mode="reduce-overhead")
    print(f"LSTM model loaded. ({checkpoint['n_classes']} action classes, {checkpoint['n_action_ids']} action IDs)")
    return model


# ── ACT inference ──────────────────────────────────────────────────────────────

ACT_MODEL_PATH         = "models/act_gamestate/act_policy.pt"
LEROBOT_ACT_MODEL_PATH = "models/lerobot_act/act_policy.pt"


def load_lerobot_act():
    from lerobot.policies.act.modeling_act import ACTPolicy as LeRobotACTPolicy
    print(f"Loading LeRobot ACT model from {LEROBOT_ACT_MODEL_PATH}...")
    checkpoint = th.load(LEROBOT_ACT_MODEL_PATH, map_location=DEVICE, weights_only=False)
    policy     = LeRobotACTPolicy(checkpoint["config"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(DEVICE)
    policy.eval()
    n_obs_steps = checkpoint["n_obs_steps"]
    print(f"LeRobot ACT model loaded. (n_obs_steps={n_obs_steps}, flat_dim={checkpoint['flat_dim']})")
    return LeRobotACTInference(policy, n_obs_steps)


def load_act(replan_interval: int = 16, temperature: float = 1):
    print(f"Loading ACT model from {ACT_MODEL_PATH}...")
    checkpoint = th.load(ACT_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model      = ACTPolicy(
        n_classes=checkpoint["n_classes"],
        n_action_ids=checkpoint["n_action_ids"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.input_ints      = checkpoint["input_ints"]
    model.action_id_vocab = checkpoint["action_id_vocab"]
    model.eval()
    print(f"ACT model loaded. ({checkpoint['n_classes']} action classes, {checkpoint['n_action_ids']} action IDs, replan_interval={replan_interval}, temperature={temperature})")
    return ACTInference(model, replan_interval=replan_interval, temperature=temperature)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bc", "lstm", "act", "lerobot"], default="bc",
                        help="Which model to use: 'bc' (MLP), 'lstm', 'act' (Transformer), or 'lerobot' (LeRobot ACT)")
    parser.add_argument("--replan-interval", type=int, default=40,
                        help="ACT only: replan every N frames (lower=more reactive, higher=better combo follow-through)")
    parser.add_argument("--temperature", type=float, default=1,
                        help="ACT only: sampling temperature (0=greedy/deterministic, 1=sample from distribution, >1=more random)")
    args = parser.parse_args()

    if args.model == "lstm":
        model  = load_lstm()
        hidden = None
    elif args.model == "act":
        agent = load_act(replan_interval=args.replan_interval, temperature=args.temperature)
    elif args.model == "lerobot":
        agent = load_lerobot_act()
    else:
        policy, input_ints = load_bc()
        buffer = deque()

    print(f"Waiting for stream file: {STREAM_PATH}")
    while not os.path.exists(STREAM_PATH):
        time.sleep(0.5)
    print(f"Stream detected. Running agent [{args.model.upper()}]...")

    with open(STREAM_PATH, "r", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        while True:
            t_infer = time.perf_counter()


            line = f.readline()
            if not line:
                time.sleep(0.001)
                continue

            #print(line)

            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue

            

            
            if args.model == "lstm":
                player, _ = _get_player_and_opponent(frame)
                mirror = not bool(player["dir"])
                float_feats, action_ids = frame_features(frame)
                bits, hidden = model.predict(float_feats, action_ids, hidden, mirror=mirror)
            elif args.model in ("act", "lerobot"):
                bits = agent.step(frame)
            else:
                player, _ = _get_player_and_opponent(frame)
                mirror = not bool(player["dir"])
                buffer.append(frame)
                if len(buffer) > HISTORY:
                    buffer.pop(0)
                bits = predict_bc(policy, input_ints, buffer, mirror=mirror)
            t_infer = (time.perf_counter() - t_infer) * 1000


            #print(line)
            #t_gamepad = time.perf_counter()
            apply_gamepad(bits)
            #t_gamepad = (time.perf_counter() - t_gamepad) * 1000

            #if t_infer > 16:
            #    print(f"infer={t_infer:.2f}ms")

            #print(time.time())

if __name__ == "__main__":
    main()
