"""
CNN A2C模型
优势演员评论家的卷积神经网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class CNNA2C(BaseModel):
    """CNN A2C模型 - Actor-Critic架构"""
    
    def __init__(self, input_shape: tuple, num_actions: int):
        super().__init__(input_shape, num_actions)
        
        # 共享的卷积特征提取器
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 共享的全连接层
        self.shared_fc = nn.Linear(64 * 7 * 7, 512)
        
        # Actor网络（策略网络）
        self.actor_fc = nn.Linear(512, 256)
        self.actor_head = nn.Linear(256, num_actions)
        
        # Critic网络（价值网络）
        self.critic_fc = nn.Linear(512, 256)
        self.critic_head = nn.Linear(256, 1)
        
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
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取共享特征"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.shared_fc(x))
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回策略输出"""
        features = self._extract_features(x)
        actor_out = F.relu(self.actor_fc(features))
        return self.actor_head(actor_out)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        features = self._extract_features(x)
        critic_out = F.relu(self.critic_fc(features))
        return self.critic_head(critic_out)
    
    def act(self, state: torch.Tensor) -> int:
        """选择动作"""
        with torch.no_grad():
            policy_output = self.forward(state)
            # 使用softmax获取动作概率
            action_probs = F.softmax(policy_output, dim=-1)
            # 采样动作
            action = torch.multinomial(action_probs, 1).item()
            return action
