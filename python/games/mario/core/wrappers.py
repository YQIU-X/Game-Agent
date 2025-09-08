#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mario游戏环境包装器
提供图像预处理、奖励塑形等功能
"""

import cv2
import numpy as np
from collections import deque
from gym import make, ObservationWrapper, wrappers, Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace


class FrameDownsample(ObservationWrapper):
    """帧下采样包装器"""
    
    def __init__(self, env, width=84, height=84):
        super(FrameDownsample, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        self._width = width
        self._height = height

    def observation(self, observation):
        """将RGB图像转换为灰度并下采样"""
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None].astype(np.uint8)


class MaxAndSkipEnv(Wrapper):
    """最大帧跳过包装器"""
    
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """执行动作并跳过帧"""
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

    def reset(self):
        """重置环境"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FireResetEnv(Wrapper):
    """火焰重置包装器"""
    
    def __init__(self, env):
        Wrapper.__init__(self, env)
        if len(env.unwrapped.get_action_meanings()) < 3:
            raise ValueError('Expected an action space of at least 3!')

    def reset(self, **kwargs):
        """重置环境并执行火焰动作"""
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FrameBuffer(ObservationWrapper):
    """帧缓冲包装器"""
    
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBuffer, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(
            obs_space.low.repeat(num_steps, axis=0),
            obs_space.high.repeat(num_steps, axis=0),
            dtype=self._dtype
        )

    def reset(self):
        """重置帧缓冲区"""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """更新帧缓冲区"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(ObservationWrapper):
    """图像转PyTorch格式包装器"""
    
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(obs_shape[::-1]),
            dtype=np.float32
        )

    def observation(self, observation):
        """转换图像维度顺序"""
        return np.moveaxis(observation, 2, 0)


class NormalizeFloats(ObservationWrapper):
    """浮点数归一化包装器"""
    
    def observation(self, obs):
        """归一化像素值到[0,1]范围"""
        return np.array(obs).astype(np.float32) / 255.0


class CustomReward(Wrapper):
    """自定义奖励包装器"""
    
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        """计算自定义奖励"""
        state, reward, done, info = self.env.step(action)
        
        # 分数奖励
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        
        # 结束奖励
        if done:
            if info['flag_get']:
                reward += 350.0  # 通关奖励
            else:
                reward -= 50.0   # 失败惩罚
        
        return state, reward / 10.0, done, info
    
    def reset(self):
        """重置环境"""
        self._current_score = 0
        return self.env.reset()


def wrap_mario_environment(environment, action_space, monitor=False, iteration=0):
    """
    包装Mario环境
    Args:
        environment: 环境名称
        action_space: 动作空间
        monitor: 是否监控
        iteration: 迭代次数
    Returns:
        env: 包装后的环境
    """
    env = make(environment)
    
    if monitor:
        env = wrappers.Monitor(env, f'recording/run{iteration}', force=True)
    
    # 应用包装器
    env = JoypadSpace(env, action_space)
    env = MaxAndSkipEnv(env)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = FrameDownsample(env)
    env = ImageToPyTorch(env)
    env = FrameBuffer(env, 4)
    env = NormalizeFloats(env)
    env = CustomReward(env)
    
    return env
