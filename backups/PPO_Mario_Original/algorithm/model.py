#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO Actor-Critic模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    """Actor-Critic神经网络架构"""
    
    def __init__(self, n_frame, act_dim, hidden_size=512):
        super().__init__()
        
        # 卷积层处理堆叠的帧 - 适配84x84输入
        self.net = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),      # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),       # 9x9 -> 7x7
            nn.ReLU(),
        )
        
        # 计算卷积后的特征图大小: 7x7x64 = 3136
        self.linear = nn.Linear(3136, hidden_size)
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # 初始化权重
        self._init_weights()
    
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
    
    def forward(self, x):
        """前向传播"""
        # 输入应该是 (batch, channels, height, width) 或 (channels, height, width)
        if x.dim() == 3:
            # 单个样本: (channels, height, width) -> (1, channels, height, width)
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            # 批次样本: (batch, channels, height, width) - 已经是正确格式
            pass
        else:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")
        
        x = self.net(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear(x))
        
        return self.policy_head(x), self.value_head(x).squeeze(-1)
    
    def act(self, obs):
        """基于当前观察选择动作"""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value
    
    def get_value(self, obs):
        """获取状态价值"""
        _, value = self.forward(obs)
        return value
    
    def get_action_and_value(self, obs, action=None):
        """获取动作和价值"""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, logprob, entropy, value
    
    def evaluate_actions(self, obs, actions):
        """评估动作"""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return logprob, entropy, value


