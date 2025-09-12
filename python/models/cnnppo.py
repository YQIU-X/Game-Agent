#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN PPO模型
PPO算法的Actor-Critic卷积神经网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class CNNPPO(BaseModel):
    """CNN PPO Actor-Critic模型"""
    
    def __init__(self, input_shape: tuple, num_actions: int, hidden_size: int = 512):
        super().__init__(input_shape, num_actions)
        
        self.hidden_size = hidden_size
        
        # 卷积特征提取层 - 匹配权重文件结构
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # 计算特征大小
        self._feature_size = self._get_feature_size()
        
        # 共享全连接层 - 匹配权重文件结构
        self.linear = nn.Linear(self._feature_size, hidden_size)
        
        # Actor头（策略网络） - 匹配权重文件结构
        self.policy_head = nn.Linear(hidden_size, num_actions)
        
        # Critic头（价值网络） - 匹配权重文件结构
        self.value_head = nn.Linear(hidden_size, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _get_feature_size(self) -> int:
        """计算特征大小"""
        x = torch.zeros(1, *self.input_shape)
        x = self.net(x)
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
    
    def forward(self, x: torch.Tensor) -> tuple:
        """前向传播，返回logits和value"""
        # 完全按照原始PPO代码的维度处理
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)  # (height, width, channels) -> (1, channels, height, width)
        
        x = self.net(x)
        x = x.reshape(x.size(0), -1)  # 展平
        x = torch.relu(self.linear(x))
        
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        return logits, value.squeeze(-1)
    
    def act(self, state: torch.Tensor) -> tuple:
        """选择动作，返回action, logprob, value"""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        _, value = self.forward(state)
        return value
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None) -> tuple:
        """获取动作和价值"""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, logprob, entropy, value
    
    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> tuple:
        """评估动作"""
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return logprob, entropy, value