#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN算法配置常量
"""

# DQN算法核心参数
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON_START = 1
EPSILON_FINAL = 0.01
EPSILON_DECAY = 100000

# 经验回放参数
BATCH_SIZE = 32
MEMORY_CAPACITY = 20000
TARGET_UPDATE_FREQUENCY = 1000
INITIAL_LEARNING = 10000

# 优先经验回放参数
BETA_START = 0.4
BETA_FRAMES = 10000

# 网络结构参数
INPUT_CHANNELS = 4
HIDDEN_SIZE = 512
OUTPUT_SIZE = 12  # complex movement action space

# 训练控制参数
NUM_EPISODES = 50000
SAVE_FREQUENCY = 100
LOG_FREQUENCY = 10
RENDER = True
SAVE_MODEL = True
USE_GPU = True
VERBOSE = True

# 文件路径配置
CONFIG_DIR = 'configs'
LOGS_DIR = 'logs'
RESULTS_DIR = 'results'
PRETRAINED_MODELS = 'pretrained_models'
TEMP_MODELS = 'temp_models'

# 迁移学习参数
TRANSFER = 'SuperMarioBros-1-1-v0'