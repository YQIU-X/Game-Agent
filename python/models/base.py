"""
基础模型类
定义所有强化学习模型的通用接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseModel(ABC, nn.Module):
    """基础模型类"""
    
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    @abstractmethod
    def act(self, state: torch.Tensor) -> int:
        """选择动作"""
        pass
    
    def load_weights(self, weights_path: str):
        """加载权重"""
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(weights)
        self.eval()
    
    def save_weights(self, weights_path: str):
        """保存权重"""
        torch.save(self.state_dict(), weights_path)
