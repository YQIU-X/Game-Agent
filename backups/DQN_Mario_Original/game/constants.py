#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mario游戏配置常量
"""

# 游戏环境
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
DEFAULT_ENVIRONMENT = 'SuperMarioBros-1-1-v0'

# 动作空间
ACTION_SPACES = {
    'simple': 'SIMPLE_MOVEMENT',
    'complex': 'COMPLEX_MOVEMENT',
    'right_only': 'RIGHT_ONLY'
}

# 默认动作空间
DEFAULT_ACTION_SPACE = 'complex'

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

# 训练参数
MAX_STEPS_PER_EPISODE = 10000
RENDER_FPS = 30

# DQN算法参数
ACTION_SPACE = 'complex'
BATCH_SIZE = 32
BETA_FRAMES = 10000
BETA_START = 0.4
ENVIRONMENT = 'SuperMarioBros-3-3-v0'
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 100000
GAMMA = 0.99
INITIAL_LEARNING = 10000
LEARNING_RATE = 1e-4
MEMORY_CAPACITY = 20000
NUM_EPISODES = 50000
PRETRAINED_MODELS = 'pretrained_models'
TEMP_MODELS = 'temp_models'
TARGET_UPDATE_FREQUENCY = 1000
TRANSFER = 'SuperMarioBros-1-1-v0'