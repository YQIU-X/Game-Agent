"""
Mario游戏环境
超级马里奥兄弟游戏环境实现
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from .base import BaseGameEnv


class MarioEnv(BaseGameEnv):
    """Mario游戏环境"""
    
    def __init__(self, env_name: str, action_space_type: str = 'SIMPLE', **kwargs):
        self.action_space_type = action_space_type
        super().__init__(env_name, **kwargs)
    
    def _setup_environment(self):
        """设置Mario环境"""
        # 创建原始环境
        self.env = gym_super_mario_bros.make(self.env_name)
        
        # 根据动作空间类型选择动作
        if self.action_space_type == 'SIMPLE':
            self.actions = SIMPLE_MOVEMENT
        else:
            self.actions = COMPLEX_MOVEMENT
        
        # 包装环境
        self.env = JoypadSpace(self.env, self.actions)
    
    def get_actions(self) -> list:
        """获取动作列表"""
        return self.actions
    
    def preprocess_frame(self, frame):
        """预处理帧 - 这里返回原始帧，具体预处理在包装器中完成"""
        return frame
