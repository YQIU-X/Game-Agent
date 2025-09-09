"""
基础包装器类
定义所有环境包装器的通用接口
"""

from abc import ABC, abstractmethod
import gym


class BaseWrapper(ABC):
    """环境包装器基类"""
    
    @abstractmethod
    def wrap(self, env):
        """包装环境"""
        pass
    
    @abstractmethod
    def get_observation_space(self):
        """获取观察空间"""
        pass
    
    @abstractmethod
    def get_action_space(self):
        """获取动作空间"""
        pass


def create_wrapper(wrapper_type: str, **kwargs):
    """创建包装器实例"""
    if wrapper_type == 'mario':
        from .mario_wrappers import wrap_mario_environment
        return wrap_mario_environment
    elif wrapper_type == 'atari':
        from .atari_wrappers import wrap_atari_environment
        return wrap_atari_environment
    else:
        raise ValueError(f"不支持的包装器类型: {wrapper_type}")

