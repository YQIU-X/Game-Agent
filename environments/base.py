"""
基础游戏环境类：所有游戏环境的基类
"""

import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional


class BaseGameEnv(ABC):
    """所有游戏环境的基类"""
    
    def __init__(self, env_name: str, action_space_type: str = 'SIMPLE', **kwargs):
        self.env_name = env_name
        self.action_space_type = action_space_type
        self.env = None
        self.actions = None
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
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        pass
    
    def reset(self):
        """重置环境"""
        return self.env.reset()
    
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
    
    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'name': self.env_name,
            'action_space_type': self.action_space_type,
            'num_actions': len(self.actions) if self.actions else 0,
            'actions': self.actions
        }
