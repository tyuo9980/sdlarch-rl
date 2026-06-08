import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import re
from pathlib import Path

import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_steps=10_000):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunk, info = self.env.step(action)
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
        return obs, reward, done, trunk, info


class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class AugmentObservation(gym.ObservationWrapper):
    def __init__(self, env, noise=False, debug=False):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.debug = debug
        self.noise = noise

    def observation(self, obs):
        augmented = obs
        if np.random.rand() < 0.5:
            frame = obs.copy()
            frame = self._random_brightness(frame)
            frame = self._safe_blur(frame)
            if self.noise:
                frame = self._add_noise(frame)
            frame = self._random_shift(frame)
            augmented = frame
            if self.debug:
                obs_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                cv2.imshow("Augmentation", obs_bgr)
                cv2.waitKey(1)
        return augmented

    def _random_brightness(self, frame):
        factor = np.random.uniform(0.85, 1.15)
        return np.clip(frame * factor, 0, 255).astype(np.uint8)

    def _add_noise(self, frame):
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        return np.clip(frame + noise, 0, 255)

    def _safe_blur(self, frame):
        if np.random.rand() < 0.4:
            important_mask = self._get_important_objects_mask(frame)
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            result = np.where(important_mask[..., np.newaxis], frame, blurred)
            return result.astype(np.uint8)
        return frame

    def _get_important_objects_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        combined = np.logical_or(red_mask > 0, blue_mask > 0)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(combined.astype(np.uint8), kernel, iterations=1)

    def _random_shift(self, frame):
        max_shift = 1
        tx = np.random.randint(-max_shift, max_shift + 1)
        ty = np.random.randint(-max_shift, max_shift + 1)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)


class HPInfoWrapper(gym.Wrapper):
    def __init__(self, env, start_x, end_x, start_y, end_y, max_hp_pixels, start_color, end_color, debug=False):
        super().__init__(env)
        self.max_hp_pixels = max_hp_pixels
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.start_color = start_color
        self.end_color = end_color
        self.debug = debug

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hp_roi = obs[self.start_x:self.end_x, self.start_y:self.end_y].copy()
        hsv = cv2.cvtColor(hp_roi, cv2.COLOR_RGB2HSV)
        lower = np.array([self.start_color, 100, 100])
        upper = np.array([self.end_color, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        if self.debug:
            hp_roi_db = cv2.resize(hp_roi, ((self.end_y - self.start_y) * 3, (self.end_x - self.start_x) * 3), interpolation=cv2.INTER_AREA)
            cv2.imshow("HP Bar", hp_roi_db)
            cv2.waitKey(1)
        hp_pixels = cv2.countNonZero(mask)
        info["hp_percent"] = hp_pixels / self.max_hp_pixels
        info["hp_pixels"] = hp_pixels
        return obs, reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip: int, stochastic: bool = False):
        super().__init__(env)
        self._skip = skip
        self.skip = skip
        self._stochastic = stochastic

    def reset(self, **kwargs):
        if self._stochastic and self._skip >= 2:
            self.skip = np.random.randint(2, self._skip)
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            observation, reward, terminated, trunk, info = self.env.step(action)
            total_reward += reward
            if terminated or trunk:
                break
        return observation, total_reward, terminated, trunk, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.shape[1], self.shape[0], 3), dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class GrayResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, keep_dim=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.keep_dim = keep_dim
        shape = (self.height, self.width, 1) if keep_dim else (self.height, self.width)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.keep_dim:
            resized = np.expand_dims(resized, axis=-1)
        return resized.astype(np.uint8)


def get_latest_model(path):
    models = list(path.glob("best_model_*"))
    if not models:
        return None
    model_numbers = [int(re.search(r"best_model_(\d+)", str(m)).group(1)) for m in models]
    return path / f"best_model_{max(model_numbers)}"


def get_last_index(path: str, file_name: str, extension: str) -> int:
    last_index = -1
    extension = extension.lstrip(".")
    for p in Path(path).glob(f"{file_name}*.{extension}"):
        suffix = p.stem[len(file_name):]
        if suffix.isdigit():
            last_index = max(last_index, int(suffix))
    return last_index


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, save_freq, model, use_curriculum=False,
                 verbose=1, reward_net=None, logger=None, use_call=False):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.use_curriculum = use_curriculum
        self.model = model
        self.reward_net = reward_net
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.counter = 0
        self.my_logger = None
        self.use_call = use_call
        if logger is not None:
            self.my_logger = logger

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def __call__(self, _locals=None, _globals=None):
        if self.use_call:
            self.counter += 1
            latest_model = get_latest_model(self.save_path)
            next_save_step = (int(re.search(r"best_model_(\d+)", str(latest_model)).group(1)) + 1) if latest_model else self.counter
            model_path = self.save_path / f"best_model_{next_save_step}"
            reward_path = self.save_path / f"reward_net_{next_save_step}.pt"
            self.model.save(model_path)
            th.save(self.reward_net.state_dict(), reward_path)
            for key, value in self.logger.name_to_value.items():
                self.my_logger.record(key, value)
            self.my_logger.dump(int(next_save_step))
            print(f"Model saved in: {model_path}")

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self.current_episode_reward = reward
        done = self.locals["dones"][0]
        self.episode_rewards.append(self.current_episode_reward)
        if done:
            if self.use_curriculum:
                self.logger.record("current_phase", self.training_env.get_attr("current_phase")[0])
            print(f"Done Rewards Step Cnt: {len(self.episode_rewards)}")
            self.episode_rewards = []
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            latest_model = get_latest_model(self.save_path)
            next_save_step = (int(re.search(r"best_model_(\d+)", str(latest_model)).group(1)) + self.check_freq) if latest_model else self.n_calls
            model_path = self.save_path / f"best_model_{next_save_step}"
            self.model.save(model_path)
            print(f"Model saved in: {model_path}")
        return True


class GenericCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros(1, n_input_channels, 96, 96)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.float()
        if observations.max() > 1.0:
            observations = observations / 255.0
        if observations.dim() == 5:
            observations = observations.squeeze(-1)
        if observations.dim() == 4 and observations.shape[3] == 1:
            observations = observations.squeeze(-1)
        return self.linear(self.cnn(observations))
