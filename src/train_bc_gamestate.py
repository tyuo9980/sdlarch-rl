import json
import os

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Trajectory
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from utils import (DATA_DIR, FLOAT_DIM, HISTORY, MAX_ACTION_ID, build_input_vocab,
                   extract_action_idx, extract_obs_embed, split_episodes)

# ── Config ─────────────────────────────────────────────────────────────────────
SAVE_PATH  = "models/bc_gamestate/"
EMBED_DIM  = 8
N_IDS      = 2 * HISTORY   # player + opponent action ID per frame
N_FLOATS   = FLOAT_DIM * HISTORY
OBS_DIM    = N_IDS + N_FLOATS
BATCH_SIZE = 1024
N_EPOCHS   = 30
LR         = 2e-4
L2         = 1e-4
ENT_WEIGHT = 1e-2

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(SAVE_PATH, exist_ok=True)
rng = np.random.default_rng(0)


# ── Embedding extractor ────────────────────────────────────────────────────────

class EmbeddingExtractor(BaseFeaturesExtractor):
    """Embeds the 2*HISTORY action IDs and concatenates with the float features."""
    def __init__(self, observation_space, embed_dim: int = EMBED_DIM):
        features_dim = N_IDS * embed_dim + N_FLOATS
        super().__init__(observation_space, features_dim=features_dim)
        self.action_embed = nn.Embedding(MAX_ACTION_ID, embed_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        ids    = obs[:, :N_IDS].long()   # (B, 2*HISTORY)
        floats = obs[:, N_IDS:]          # (B, 9*HISTORY)
        embs   = self.action_embed(ids).flatten(1)  # (B, 2*HISTORY*embed_dim)
        return th.cat([embs, floats], dim=-1)


# ── Trajectories ───────────────────────────────────────────────────────────────

def build_trajectories(frames: list, vocab: dict) -> list[Trajectory]:
    def buildTrajectory(ep):
        obs_list = []
        for i in range(len(ep)):
            window = ep[max(0, i - HISTORY + 1) : i + 1]
            obs_list.append(extract_obs_embed(window))

        obs   = np.array(obs_list, dtype=np.float32)
        acts  = np.array([extract_action_idx(f, vocab) for f in ep[:-1]], dtype=np.int64)
        infos = np.array([{}] * (len(ep) - 1))
        return Trajectory(obs=obs, acts=acts, infos=infos, terminal=True)

    return [buildTrajectory(ep) for ep in split_episodes(frames)]


# ── Train ──────────────────────────────────────────────────────────────────────

def main():
    frames = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path) as f:
            data = json.load(f)
        frames += data["frame_data"]
        print(f"Loaded {filename}: {len(data['frame_data'])} frames")
    print(f"Total frames: {len(frames)}")

    vocab, input_ints = build_input_vocab(frames)
    N_CLASSES = len(input_ints)
    print(f"Action vocabulary: {N_CLASSES} distinct inputs")

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space      = gym.spaces.Discrete(N_CLASSES)

    policy = ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        features_extractor_class=EmbeddingExtractor,
        features_extractor_kwargs=dict(embed_dim=EMBED_DIM),
    ).to(DEVICE)

    trajectories = build_trajectories(frames, vocab)
    print(f"Built {len(trajectories)} trajectories")

    transitions = rollout.flatten_trajectories(trajectories)
    print(f"Total transitions: {len(transitions)}")

    trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        policy=policy,
        rng=rng,
        batch_size=BATCH_SIZE,
        ent_weight=ENT_WEIGHT,
        l2_weight=L2,
        optimizer_kwargs=dict(lr=LR),
    )

    # imitation moves obs to device but not acts — patch the loss calculator to fix this
    _original_loss = trainer.loss_calculator
    def _loss_on_device(policy, obs, acts):
        return _original_loss(policy, obs, acts.to(DEVICE))
    trainer.loss_calculator = _loss_on_device

    trainer.train(n_epochs=N_EPOCHS)

    save_file  = os.path.join(SAVE_PATH, "bc_policy_gamestate.zip")
    vocab_file = os.path.join(SAVE_PATH, "bc_action_vocab.json")
    trainer.policy.save(save_file)
    with open(vocab_file, "w") as f:
        json.dump(input_ints, f)
    print(f"Saved to {save_file}")
    print(f"Vocab saved to {vocab_file}")

if __name__ == "__main__":
    main()
