"""
CNN PPO模型：用于近端策略优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from .base import BaseModel


class CNNPPO(BaseModel):
    """CNN PPO模型 - 使用ActorCritic架构"""
    
    def __init__(self, input_shape, num_actions):
        super().__init__(input_shape, num_actions)
        
        # 卷积层处理堆叠帧
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        
        # 计算特征大小
        self._feature_size = self._get_feature_size()
        
        # 全连接层用于策略和价值头
        self.linear = nn.Linear(self._feature_size, 512)
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回策略头输出"""
        # 确保输入维度正确 (batch, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加batch维度
        
        x = self.net(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear(x))
        
        return self.policy_head(x)

    def get_action_probs(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布"""
        logits = self.forward(state_tensor)
        return F.softmax(logits, dim=-1)
    
    def get_value(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        # 确保输入维度正确
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)  # 添加batch维度
        
        x = self.net(state_tensor)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear(x))
        
        return self.value_head(x).squeeze(-1)

    def act(self, state_tensor: torch.Tensor) -> int:
        """选择动作"""
        # 确保输入维度正确
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)  # 添加batch维度
        
        logits, value = self._forward_with_value(state_tensor)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item())

    def _forward_with_value(self, x: torch.Tensor) -> tuple:
        """前向传播并返回策略和价值"""
        # 确保输入维度正确 (batch, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加batch维度
        
        x = self.net(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear(x))
        
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def get_action_logprob_value(self, state_tensor: torch.Tensor) -> tuple:
        """获取动作、对数概率和价值（用于PPO训练）"""
        logits, value = self._forward_with_value(state_tensor)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    @property
    def feature_size(self) -> int:
        return self._feature_size

    def _get_feature_size(self) -> int:
        """计算特征大小"""
        x = torch.zeros(1, *self._input_shape)
        x = self.net(x)
        return x.view(1, -1).size(1)
