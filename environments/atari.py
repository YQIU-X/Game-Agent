"""
Atari游戏环境
"""

import gym
import numpy as np
import cv2
from .base import BaseGameEnv


class AtariEnv(BaseGameEnv):
    """Atari游戏环境"""
    
    def _setup_environment(self):
        """设置Atari环境"""
        # 创建基础环境
        self.env = gym.make(self.env_name)
        
        # Atari游戏通常使用离散动作空间
        self.actions = list(range(self.env.action_space.n))
    
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
