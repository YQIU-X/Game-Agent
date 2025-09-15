#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mario游戏配置常量
"""

# 游戏环境列表
MARIO_ENVIRONMENTS = [
    'SuperMarioBros-1-1-v0',
    'SuperMarioBros-1-2-v0',
    'SuperMarioBros-1-3-v0',
    'SuperMarioBros-1-4-v0',
    'SuperMarioBros-2-1-v0',
    'SuperMarioBros-2-2-v0',
    'SuperMarioBros-2-3-v0',
    'SuperMarioBros-2-4-v0',
    'SuperMarioBros-3-1-v0',
    'SuperMarioBros-3-2-v0',
    'SuperMarioBros-3-3-v0'
]

# 默认环境
DEFAULT_ENVIRONMENT = "SuperMarioBros-1-1-v0"

# 动作空间定义
ACTION_SPACES = {
    'simple': 'SIMPLE_MOVEMENT',
    'complex': 'COMPLEX_MOVEMENT',
    'right_only': 'RIGHT_ONLY'
}

# 默认动作空间
DEFAULT_ACTION_SPACE = "complex"

# 图像处理参数
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_SKIP = 4
FRAME_BUFFER_SIZE = 4

# 奖励参数
SCORE_REWARD_SCALE = 40.0
FLAG_REWARD = 350.0
DEATH_PENALTY = -50.0
REWARD_SCALE = 10.0

# 游戏控制参数
MAX_STEPS_PER_EPISODE = 10000
RENDER_FPS = 30