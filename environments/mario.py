"""
马里奥游戏环境
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np
import cv2
from .base import BaseGameEnv


class MarioEnv(BaseGameEnv):
    """马里奥游戏环境"""
    
    def _setup_environment(self):
        """设置马里奥环境"""
        # 创建基础环境
        self.env = gym_super_mario_bros.make(self.env_name)
        
        # 根据动作空间类型选择动作集
        if self.action_space_type == 'SIMPLE':
            self.actions = SIMPLE_MOVEMENT
        elif self.action_space_type == 'COMPLEX':
            self.actions = COMPLEX_MOVEMENT
        else:
            raise ValueError(f"不支持的动作空间类型: {self.action_space_type}")
        
        # 包装环境
        self.env = JoypadSpace(self.env, self.actions)
    
    def get_actions(self):
        """获取动作列表"""
        return self.actions
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 调整大小到84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 归一化到[0, 1]
        normalized = resized / 255.0
        
        return normalized
