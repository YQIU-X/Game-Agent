"""
Atari游戏环境包装器
为Atari游戏提供标准化的环境包装
"""

import gym
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation


def wrap_atari_environment(env_name: str, **kwargs):
    """
    包装Atari环境
    
    Args:
        env_name: 环境名称
        **kwargs: 其他参数
    
    Returns:
        包装后的环境
    """
    # 创建原始环境
    env = gym.make(env_name)
    
    # 应用标准包装器
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)
    
    return env


def get_atari_actions(env_name: str):
    """获取Atari环境的动作空间"""
    env = gym.make(env_name)
    return env.action_space.n

