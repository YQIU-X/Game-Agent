"""
Atari游戏环境
雅达利游戏环境实现
"""

import gym
from .base import BaseGameEnv


class AtariEnv(BaseGameEnv):
    """Atari游戏环境"""
    
    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)
    
    def _setup_environment(self):
        """设置Atari环境"""
        self.env = gym.make(self.env_name)
    
    def get_actions(self) -> list:
        """获取动作列表"""
        # Atari环境的动作是离散的，返回动作数量
        return list(range(self.env.action_space.n))
    
    def preprocess_frame(self, frame):
        """预处理帧 - 这里返回原始帧，具体预处理在包装器中完成"""
        return frame
