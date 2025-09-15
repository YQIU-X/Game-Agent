"""
PPO专用环境包装器
为PPO算法优化的Mario环境包装器
"""

import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces
from gym.wrappers import Monitor
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 设置OpenCV不使用OpenCL
cv2.ocl.setUseOpenCL(False)


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
            obs = self.env.reset(**kwargs)
            # 在某些情况下，原始环境的reset可能不会返回info，所以我们需要一个step来获取它。
            obs, _, _, info = self.env.step(0)
            self.lives = info.get("life", 0)
        else:
            # 如果不是真正的游戏结束（只是失去一条命），则继续当前关卡。
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


def wrap_ppo_environment(environment, action_space=None, monitor=False, monitor_path=None):
    """
    创建PPO专用的Mario环境包装器
    
    Args:
        environment: 环境名称 (e.g., 'SuperMarioBros-1-1-v0')
        action_space: 动作空间，默认为COMPLEX_MOVEMENT
        monitor: 是否启用监控录制
        monitor_path: 监控录制路径
    
    Returns:
        env: 包装后的PPO环境
    """
    # 设置环境变量
    import os
    os.environ['PYGLET_HEADLESS'] = '1'
    
    # 创建原始环境
    env = gym_super_mario_bros.make(environment)
    
    # 启用监控录制
    if monitor:
        env = Monitor(env, monitor_path, force=True, video_callable=lambda episode_id: True)
    
    # 设置动作空间
    if action_space is None:
        action_space = COMPLEX_MOVEMENT
    env = JoypadSpace(env, action_space)
    
    # 应用PPO专用包装器
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeMario(env)
    env = WarpFrame(env, width=84, height=84)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)
    
    return env


def get_ppo_actions():
    """获取PPO默认动作空间"""
    return COMPLEX_MOVEMENT


# PPO超参数配置
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
