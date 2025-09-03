"""
基础模型类：所有模型的基类
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseModel(nn.Module, ABC):
    """所有强化学习模型的基类"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int, **kwargs):
        super().__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    @abstractmethod
    def act(self, state_tensor: torch.Tensor) -> int:
        """选择动作"""
        pass
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape
    
    @property
    def num_actions(self) -> int:
        return self._num_actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'type': self.__class__.__name__,
            'input_shape': self._input_shape,
            'num_actions': self._num_actions
        }
