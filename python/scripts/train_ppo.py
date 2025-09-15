#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO训练脚本 - 使用新的模块化结构
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

# 修复pyglet兼容性问题并设置无头模式
import os
os.environ['PYGLET_HEADLESS'] = '1'  # 设置无头模式
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 禁用SDL视频驱动
os.environ['DISPLAY'] = ''  # 清空显示环境变量

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 使用修复版本的PPO训练逻辑
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
from datetime import datetime

# 设置环境变量
os.environ['PYGLET_HEADLESS'] = '1'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['DISPLAY'] = ''

cv2.ocl.setUseOpenCL(False)

# Super Mario Bros. environment action space
COMPLEX_MOVEMENT = COMPLEX_MOVEMENT

# 导入实验管理器
from python.utils.experiment_manager import experiment_manager
from python.games.mario.core.constants import DEFAULT_ENVIRONMENT, DEFAULT_ACTION_SPACE

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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO训练脚本')
    
    # 环境参数
    parser.add_argument('--environment', type=str, default=DEFAULT_ENVIRONMENT,
                       help='环境名称')
    parser.add_argument('--action-space', type=str, default=DEFAULT_ACTION_SPACE,
                       choices=['simple', 'complex', 'right_only'],
                       help='动作空间')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=50000,
                       help='训练episode数')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--lambda-gae', type=float, default=0.95,
                       help='GAE lambda参数')
    parser.add_argument('--clip-eps', type=float, default=0.2,
                       help='PPO裁剪参数')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                       help='PPO裁剪参数（兼容参数名）')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='价值函数损失系数')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                       help='价值函数损失系数（兼容参数名）')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='熵系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='熵系数（兼容参数名）')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='最大梯度范数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='最大梯度范数（兼容参数名）')
    parser.add_argument('--epochs', type=int, default=4,
                       help='PPO更新轮数')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                       help='PPO更新轮数（兼容参数名）')
    parser.add_argument('--minibatch-size', type=int, default=64,
                       help='小批次大小')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='小批次大小（兼容参数名）')
    parser.add_argument('--num-env', type=int, default=8,
                       help='并行环境数量')
    parser.add_argument('--rollout-steps', type=int, default=128,
                       help='rollout步数')
    
    # 其他参数
    parser.add_argument('--render', action='store_true',
                       help='启用渲染')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='保存模型')
    parser.add_argument('--force-cpu', action='store_true',
                       help='强制使用CPU')
    parser.add_argument('--save-path', type=str, default='models',
                       help='模型保存路径')
    parser.add_argument('--log-frequency', type=int, default=10,
                       help='日志输出频率')
    parser.add_argument('--save-frequency', type=int, default=100,
                       help='模型保存频率')
    parser.add_argument('--eval-frequency', type=int, default=50,
                       help='评估频率')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='评估episode数')
    
    return parser.parse_args()


def train_ppo():
    """Main function to train the PPO agent."""
    args = parse_args()
    
    # 设置实验目录
    config = {
        'environment': args.environment,
        'action_space': args.action_space,
        'algorithm': 'PPO',
        'learning_rate': getattr(args, 'learning_rate', 2.5e-4),
        'gamma': args.gamma,
        'lambda_gae': args.lambda_gae,
        'clip_eps': getattr(args, 'clip_eps', getattr(args, 'clip_ratio', 0.2)),
        'vf_coef': getattr(args, 'vf_coef', getattr(args, 'value_loss_coef', 0.5)),
        'ent_coef': getattr(args, 'ent_coef', getattr(args, 'entropy_coef', 0.01)),
        'max_grad_norm': getattr(args, 'max_grad_norm', 0.5),
        'epochs': getattr(args, 'epochs', getattr(args, 'ppo_epochs', 4)),
        'minibatch_size': getattr(args, 'minibatch_size', getattr(args, 'batch_size', 64)),
        'num_env': getattr(args, 'num_env', 8),
        'rollout_steps': getattr(args, 'rollout_steps', 128),
        'force_cpu': args.force_cpu,
        'log_frequency': args.log_frequency,
        'save_frequency': args.save_frequency,
        'eval_frequency': args.eval_frequency,
        'eval_episodes': args.eval_episodes
    }
    
    # 创建实验目录
    experiment_dir = experiment_manager.create_experiment_dir(
        args.environment, 'PPO', config
    )
    metrics_path = experiment_manager.get_metrics_path(experiment_dir)
    weights_dir = experiment_manager.get_weights_dir(experiment_dir)
    
    print(f"[PPO Training] 实验目录: {experiment_dir}", flush=True)
    print(f"[PPO Training] CSV log file created: {metrics_path}", flush=True)
    
    # 创建CSV文件并写入表头（PPO专用格式，包含更多有用信息）
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'best_reward', 'average_reward', 'episode_length', 'total_steps', 'epsilon', 'average_loss', 'learning_rate', 'gamma', 'max_stage', 'flag_get', 'avg_total_reward', 'policy_loss', 'value_loss', 'entropy', 'timestamp'])
    
    # Load hyperparameters from the config file
    num_env = getattr(args, 'num_env', 8)
    rollout_steps = getattr(args, 'rollout_steps', 128)
    epochs = getattr(args, 'epochs', getattr(args, 'ppo_epochs', 4))
    minibatch_size = getattr(args, 'minibatch_size', getattr(args, 'batch_size', 64))
    clip_eps = getattr(args, 'clip_eps', getattr(args, 'clip_ratio', 0.2))
    vf_coef = getattr(args, 'vf_coef', getattr(args, 'value_loss_coef', 0.5))
    ent_coef = getattr(args, 'ent_coef', getattr(args, 'entropy_coef', 0.01))
    
    # Create the vectorized environment
    envs = gym.vector.SyncVectorEnv([lambda: make_env_mario(args.environment) for _ in range(num_env)])
    obs_dim = envs.single_observation_space.shape[-1]
    act_dim = envs.single_action_space.n
    print(f"Observation space dim: {obs_dim}, Action space dim: {act_dim}", flush=True)
    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(args, 'learning_rate', 2.5e-4))

    init_obs = envs.reset()
    update = 0
    
    highest_training_return = -np.inf
    best_reward = -np.inf
    total_rewards_history = []
    
    while update < args.episodes:
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

        # 记录损失值
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
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
                
                # 记录损失值
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())

        # logging
        avg_return = batch["returns"].mean().item()
        max_stage = batch["max_stage"]
        flag_get = batch["flag_get"]
        avg_total_reward = batch["avg_total_reward"]
        
        # 更新最佳奖励
        if avg_total_reward > best_reward:
            best_reward = avg_total_reward
        
        # 计算平均损失
        avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0
        avg_value_loss = np.mean(epoch_value_losses) if epoch_value_losses else 0.0
        avg_entropy = np.mean(epoch_entropies) if epoch_entropies else 0.0
        avg_loss = avg_policy_loss + avg_value_loss  # 总损失
        
        print(f"Update {update}: avg return = {avg_return:.2f} max_stage={max_stage} flag_get={flag_get} avg_total_reward={avg_total_reward:.2f} policy_loss={avg_policy_loss:.4f} value_loss={avg_value_loss:.4f}", flush=True)

        # 保存指标到CSV文件（PPO专用格式）
        metrics_data = [
            update,                                    # episode
            round(avg_total_reward, 3),              # total_reward
            round(best_reward, 3),                   # best_reward (历史最佳)
            round(avg_return, 3),                     # average_reward
            rollout_steps * num_env,                  # episode_length
            update * rollout_steps * num_env,         # total_steps
            0.0,                                      # epsilon (PPO不使用epsilon)
            round(avg_loss, 6),                      # average_loss
            getattr(args, 'learning_rate', 2.5e-4),   # learning_rate
            0.99,                                     # gamma
            max_stage,                                # max_stage
            flag_get,                                 # flag_get
            round(avg_total_reward, 3),              # avg_total_reward
            round(avg_policy_loss, 6),               # policy_loss
            round(avg_value_loss, 6),                # value_loss
            round(avg_entropy, 6),                   # entropy
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # timestamp
        ]
        
        with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_data)

        # eval and save
        if update % 10 == 0:
            if avg_return > highest_training_return:
                print(f"New training high score detected!", flush=True)
            if flag_get:
                print(f"Stage cleared!", flush=True)
            
            highest_training_return = max(highest_training_return, avg_return)
            
            # 进行评估（不录制视频，和DQN保持一致）
            eval_env = make_env_mario(args.environment, monitor=False)
            avg_score, info, eval_max_stage, flag_get_eval = evaluate_policy(
                eval_env, model, episodes=1, render=False
            )
            eval_env.close()

            print(f"[Eval] Update {update}: avg return = {avg_score:.2f} info: {info}", flush=True)
            if flag_get_eval:
                # 使用实验管理器保存模型
                model_path = experiment_manager.save_model(
                    experiment_dir, f"clear_stage_update_{update}", 
                    model.state_dict()
                )
                print(f"Model saved to {model_path} after completing the stage!", flush=True)
                break
        
        # 定期保存模型
        if update > 0 and update % args.save_frequency == 0:
            model_path = experiment_manager.save_model(
                experiment_dir, f"checkpoint_update_{update}", 
                model.state_dict()
            )
            print(f"Model saved to {model_path}", flush=True)

def main():
    """主函数"""
    train_ppo()


if __name__ == '__main__':
    main()