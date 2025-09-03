"""
CNN DQN模型：用于深度Q网络
"""

import torch
import torch.nn as nn
from .base import BaseModel


class CNNDQN(BaseModel):
    """CNN DQN模型"""
    
    def __init__(self, input_shape, num_actions):
        super().__init__(input_shape, num_actions)
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

    @property
    def feature_size(self) -> int:
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

    @torch.no_grad()
    def act(self, state_tensor: torch.Tensor) -> int:
        q = self.forward(state_tensor)
        return int(q.argmax(dim=1).item())
