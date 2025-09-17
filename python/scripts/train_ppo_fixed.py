#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO训练脚本 - 基于用户提供的完整代码
"""

import os
import csv
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym.vector
from collections import deque
import cv2
import gym
from gym import spaces
from gym.wrappers import Monitor
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 设置环境变量
os.environ['PYGLET_HEADLESS'] = '1'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['DISPLAY'] = ''

cv2.ocl.setUseOpenCL(False)

# Super Mario Bros. environment action space
COMPLEX_MOVEMENT = COMPLEX_MOVEMENT

# --- 环境包装器 ---

class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking a random number of no-ops on reset."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, info = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeMario(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = info.get("life", 0)
        
        # This check is crucial to handle the first step after a real done reset
        # When lives are lost, the environment should be treated as done for RL purposes.
        if lives < self.lives and lives > 0:
            done = True
        
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            # 如果上一个回合是真正的游戏结束，则完全重置环境。
            # 这会返回新的初始观察和信息。
            obs = self.env.reset(**kwargs)
            # 在某些情况下，原始环境的reset可能不会返回info，所以我们需要一个step来获取它。
            # 这是因为 gym-super-mario-bros 在 reset 时不会提供 life 信息。
            obs, _, _, info = self.env.step(0)
            self.lives = info.get("life", 0)

        else:
            # 如果不是真正的游戏结束（只是失去一条命），则继续当前关卡。
            # 执行一个空操作 (step(0)) 来获取新回合的初始状态和生命值。
            obs, _, _, info = self.env.step(0)
            self.lives = info.get("life", 0)
            
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame"""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    """Bin reward to {+1, 0, -1} by its sign."""
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    """Grayscale and resize the frames."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    """Stack k frames together to provide temporal context."""
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize frame data to float32 in [0, 1]."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    """A lazy way to stack frames to save memory."""
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def get_reward(r):
    """A custom reward shaping function to improve training stability."""
    r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
    return r

def make_env_mario(environment, monitor=False, monitor_path=None):
    """Creates and wraps a single Super Mario Bros environment with all wrappers."""
    env = gym_super_mario_bros.make(environment)
    if monitor:
        env = Monitor(env, monitor_path, force=True, video_callable=lambda episode_id: True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeMario(env)
    env = WarpFrame(env, width=84, height=84)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)
    return env

PPO_HYPERPARAMS = {
    "environment": "SuperMarioBros-1-1-v0",
    "num_env": 8,
    "rollout_steps": 128,
    "epochs": 4,
    "minibatch_size": 64,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
}

# --- PPO Agent ---

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

training_metrics = []

class ActorCritic(nn.Module):
    """The Actor-Critic neural network architecture."""
    def __init__(self, n_frame, act_dim):
        super().__init__()
        # Convolutional layers to process the stacked frames
        self.net = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), # Modified
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), # Modified
            nn.ReLU(),
        )
        # Fully connected layers for policy and value heads
        self.linear = nn.Linear(3136, 512) # Modified
        self.policy_head = nn.Linear(512, act_dim)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # Permute dimensions for PyTorch's Conv2d layer (batch, channels, height, width)
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.net(x)
        x = x.reshape(x.size(0), -1) # Modified
        x = torch.relu(self.linear(x))

        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, obs):
        """Select an action based on the current observation."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

def compute_gae_batch(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute General Advantage Estimation (GAE) for a batch of rollouts."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns

def rollout_with_bootstrap(envs, model, rollout_steps, init_obs):
    """
    Performs a policy rollout for a specified number of steps,
    collecting experiences and computing advantages using GAE.
    """
    obs = init_obs
    obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
    flag_get_in_rollout = False
    episode_infos = []
    episode_rewards = [] # 新增：用于存储每个回合的总奖励
    current_rewards = np.zeros(envs.num_envs) # 新增：用于跟踪每个环境的当前奖励

    for _ in range(rollout_steps):
        obs_buf.append(obs)

        with torch.no_grad():
            action, logp, value = model.act(torch.tensor(obs, dtype=torch.float32).to(device))

        val_buf.append(value)
        logp_buf.append(logp)
        act_buf.append(action)

        actions = action.cpu().numpy()
        next_obs, reward, done, infos = envs.step(actions)

        # 累加每个环境的奖励
        current_rewards += reward
        
        # 将原始奖励传入，用于计算优势
        reward = [get_reward(r) for r in reward]

        rew_buf.append(torch.tensor(reward, dtype=torch.float32).to(device))
        done_buf.append(torch.tensor(done, dtype=torch.float32).to(device))
        
        for i, d in enumerate(done):
            if d:
                info = infos[i]
                if info.get('flag_get', False):
                    flag_get_in_rollout = True
                
                # Check for `episode` key which is added by gym.vector.SyncVectorEnv on done
                if 'episode' in info:
                    episode_infos.append(info['episode'])
                
                # 新增：如果回合结束，记录并重置总奖励
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0

        obs = next_obs
    
    with torch.no_grad():
        _, last_value = model.forward(torch.tensor(obs, dtype=torch.float32).to(device))

    obs_buf = np.array(obs_buf)
    act_buf = torch.stack(act_buf)
    rew_buf = torch.stack(rew_buf)
    done_buf = torch.stack(done_buf)
    val_buf = torch.stack(val_buf)
    val_buf = torch.cat([val_buf, last_value.unsqueeze(0)], dim=0)
    logp_buf = torch.stack(logp_buf)
    
    # Reshape obs_buf for proper tensor conversion later
    obs_buf = obs_buf.reshape(rollout_steps, envs.num_envs, *envs.single_observation_space.shape)
    obs_buf = torch.tensor(obs_buf, dtype=torch.float32).to(device)
    
    adv_buf, ret_buf = compute_gae_batch(rew_buf, val_buf, done_buf)
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    # Extract max stage from episode infos
    max_stage = max([i.get('stage', 1) for i in episode_infos]) if episode_infos else 1
    
    # 新增：计算并返回平均原始总奖励
    avg_total_reward = np.mean(episode_rewards) if episode_rewards else 0

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logprobs": logp_buf,
        "advantages": adv_buf,
        "returns": ret_buf,
        "max_stage": max_stage,
        "last_obs": obs,
        "flag_get": flag_get_in_rollout,
        "avg_total_reward": avg_total_reward # 新增
    }

def evaluate_policy(env, model, episodes=5, render=False):
    """Function to evaluate the learned policy."""
    model.eval()
    total_returns = []
    actions = []
    stages = []
    flag_get_eval = False
    
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = (
                torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()
                actions.append(action)

            obs, reward, done, info = env.step(action)
            stages.append(info["stage"])
            total_reward += reward
            if info.get('flag_get', False):
                flag_get_eval = True

        total_returns.append(total_reward)
    
    model.train()
    info = {"action_count": Counter(actions)}
    return np.mean(total_returns), info, max(stages), flag_get_eval

def save_metrics_to_csv(file_name):
    """Save collected episode metrics to a CSV file."""
    filename = f'{file_name}_training_metrics.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        # 新增 'avg_total_reward' 列
        writer = csv.DictWriter(f, fieldnames=['episode', 'avg_return', 'max_stage', 'flag_get', 'avg_total_reward'])
        if not file_exists:
            writer.writeheader()
        
        for metric in training_metrics:
            writer.writerow(metric)
    
    training_metrics.clear()

def train_ppo():
    """Main function to train the PPO agent."""
    # Load hyperparameters from the config file
    num_env = PPO_HYPERPARAMS["num_env"]
    rollout_steps = PPO_HYPERPARAMS["rollout_steps"]
    epochs = PPO_HYPERPARAMS["epochs"]
    minibatch_size = PPO_HYPERPARAMS["minibatch_size"]
    clip_eps = PPO_HYPERPARAMS["clip_eps"]
    vf_coef = PPO_HYPERPARAMS["vf_coef"]
    ent_coef = PPO_HYPERPARAMS["ent_coef"]
    
    # Create the vectorized environment
    envs = gym.vector.SyncVectorEnv([lambda: make_env_mario(PPO_HYPERPARAMS['environment']) for _ in range(num_env)])
    obs_dim = envs.single_observation_space.shape[-1]
    act_dim = envs.single_action_space.n
    print(f"Observation space dim: {obs_dim}, Action space dim: {act_dim}")
    model = ActorCritic(obs_dim, act_dim).to(device)
    # try:
    #     model.load_state_dict(torch.load("mario_1_1_ppo.pt"))
    # except FileNotFoundError:
    #     print("No pre-trained model found, starting from scratch.")
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

    init_obs = envs.reset()
    update = 0
    
    highest_training_return = -np.inf
    
    while True:
        update += 1
        batch = rollout_with_bootstrap(envs, model, rollout_steps, init_obs)
        init_obs = batch["last_obs"]

        T, N = rollout_steps, envs.num_envs
        total_size = T * N

        obs = batch["obs"].reshape(total_size, *envs.single_observation_space.shape)
        act = batch["actions"].reshape(total_size)
        logp_old = batch["logprobs"].reshape(total_size)
        adv = batch["advantages"].reshape(total_size)
        ret = batch["returns"].reshape(total_size)

        for _ in range(epochs):
            idx = torch.randperm(total_size)
            for start in range(0, total_size, minibatch_size):
                i = idx[start : start + minibatch_size]
                logits, value = model(obs[i])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[i])
                ratio = torch.exp(logp - logp_old[i])
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
                policy_loss = -torch.min(ratio * adv[i], clipped).mean()
                value_loss = (ret[i] - value).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # logging
        avg_return = batch["returns"].mean().item()
        max_stage = batch["max_stage"]
        flag_get = batch["flag_get"]
        avg_total_reward = batch["avg_total_reward"] # 新增
        print(f"Update {update}: avg return = {avg_return:.2f} max_stage={max_stage} flag_get={flag_get} avg_total_reward={avg_total_reward:.2f}")

        metrics_dict = {
            'episode': update,
            'avg_return': round(avg_return, 3),
            'max_stage': max_stage,
            'flag_get': flag_get,
            'avg_total_reward': round(avg_total_reward, 3) # 新增
        }
        training_metrics.append(metrics_dict)
        save_metrics_to_csv(f"{PPO_HYPERPARAMS['environment']}_ppo")

        # eval and save
        if update % 10 == 0:
            should_save_video = False
            if avg_return > highest_training_return:
                should_save_video = True
                print(f"New training high score detected! Saving video...")
            if flag_get:
                should_save_video = True
                print(f"Stage cleared! Saving video...")
            
            highest_training_return = max(highest_training_return, avg_return)
            
            if should_save_video:
                monitor_path = f"experiments/{PPO_HYPERPARAMS['environment']}_ppo/run_{update}"
                os.makedirs(monitor_path, exist_ok=True)
                eval_env_with_monitor = make_env_mario(PPO_HYPERPARAMS['environment'], monitor=True, monitor_path=monitor_path)

                avg_score, info, eval_max_stage, flag_get_eval = evaluate_policy(
                    eval_env_with_monitor, model, episodes=1, render=False
                )
                eval_env_with_monitor.close()
            else:
                eval_env_no_monitor = make_env_mario(PPO_HYPERPARAMS['environment'])
                avg_score, info, eval_max_stage, flag_get_eval = evaluate_policy(
                    eval_env_no_monitor, model, episodes=1, render=False
                )
                eval_env_no_monitor.close()
                print(f"No new training high score or stage clear. Skipping video save.")

            print(f"[Eval] Update {update}: avg return = {avg_score:.2f} info: {info}")
            if flag_get_eval:
                torch.save(model.state_dict(), f"{PPO_HYPERPARAMS['environment']}_clear.pt")
                print(f"Model saved to {PPO_HYPERPARAMS['environment']}_clear.pt after completing the stage!")
                break
        
        if update > 0 and update % 50 == 0:
            torch.save(model.state_dict(), f"{PPO_HYPERPARAMS['environment']}_ppo.pt")

if __name__ == "__main__":
    train_ppo()





