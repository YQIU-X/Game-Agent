"""
CNN DQN模型
深度Q网络的卷积神经网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class CNNDQN(BaseModel):
    """CNN DQN模型"""
    
    def __init__(self, input_shape: tuple, num_actions: int):
        super().__init__(input_shape, num_actions)
        
        # 使用与权重文件兼容的层名称
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # 计算特征大小
        self._feature_size = self._get_feature_size()
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _get_feature_size(self) -> int:
        """计算特征大小"""
        x = torch.zeros(1, *self.input_shape)
        x = self.features(x)
        return x.view(1, -1).size(1)
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def act(self, state: torch.Tensor) -> int:
        """选择动作"""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
