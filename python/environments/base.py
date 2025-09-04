"""
基础环境类
定义所有游戏环境的通用接口
"""

import gym
from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class BaseGameEnv(ABC):
    """基础游戏环境类"""
    
    def __init__(self, env_name: str, **kwargs):
        self.env_name = env_name
        self.env = None
        self._setup_environment()
    
    @abstractmethod
    def _setup_environment(self):
        """设置游戏环境"""
        pass
    
    @abstractmethod
    def get_actions(self) -> List:
        """获取动作列表"""
        pass
    
    @abstractmethod
    def preprocess_frame(self, frame) -> Any:
        """预处理帧"""
        pass
    
    def reset(self):
        """重置环境"""
        reset_result = self.env.reset()
        
        # 兼容旧版Gym：确保返回值是numpy数组
        if isinstance(reset_result, tuple):
            return reset_result[0]  # 如果是tuple，取第一个元素
        else:
            return reset_result
    
    def step(self, action):
        """执行动作"""
        return self.env.step(action)
    
    def render(self, mode='rgb_array'):
        """渲染环境"""
        return self.env.render(mode=mode)
    
    def close(self):
        """关闭环境"""
        if self.env:
            self.env.close()
    
    @property
    def action_space(self):
        """动作空间"""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """观察空间"""
        return self.env.observation_space
