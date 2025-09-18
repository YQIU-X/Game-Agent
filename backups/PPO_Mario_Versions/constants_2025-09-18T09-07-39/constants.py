#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO算法常量配置
"""

# 训练参数
NUM_EPISODES = 50000
LEARNING_RATE = 0.0001
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_RATIO = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 64
NUM_ENV = 8
ROLLOUT_STEPS = 128
TARGET_KL = 0.01
SAVE_FREQUENCY = 100
LOG_FREQUENCY = 10
EVAL_FREQUENCY = 50
EVAL_EPISODES = 5
RENDER = True
SAVE_MODEL = True
USE_GPU = True
VERBOSE = True

# PPO超参数字典（用于前端兼容性）
PPO_HYPERPARAMS = {
    "environment": "SuperMarioBros-1-1-v0",
    "num_env": NUM_ENV,
    "rollout_steps": ROLLOUT_STEPS,
    "epochs": PPO_EPOCHS,
    "minibatch_size": BATCH_SIZE,
    "clip_eps": CLIP_RATIO,
    "vf_coef": VALUE_LOSS_COEF,
    "ent_coef": ENTROPY_COEF,
    "learning_rate": LEARNING_RATE,
    "gamma": GAMMA,
    "lambda_gae": LAMBDA_GAE,
    "max_grad_norm": MAX_GRAD_NORM,
    "save_frequency": SAVE_FREQUENCY,
    "log_frequency": LOG_FREQUENCY,
    "eval_frequency": EVAL_FREQUENCY,
    "eval_episodes": EVAL_EPISODES,
    "render": RENDER,
    "save_model": SAVE_MODEL,
    "use_gpu": USE_GPU,
    "verbose": VERBOSE
}

# 设备配置
DEVICE_CONFIG = {
    'auto': True,
    'force_cpu': False,
    'force_cuda': False,
    'force_mps': False
}

# 环境配置
ENVIRONMENT_CONFIG = {
    'mario': {
        'name': 'Super Mario Bros',
        'environments': [
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
        ],
        'action_spaces': {
            'simple': 'SIMPLE_MOVEMENT',
            'complex': 'COMPLEX_MOVEMENT',
            'right_only': 'RIGHT_ONLY'
        }
    }
}

# 日志配置
LOG_CONFIG = {
    'save_frequency': SAVE_FREQUENCY,
    'log_frequency': LOG_FREQUENCY,
    'eval_frequency': EVAL_FREQUENCY,
    'eval_episodes': EVAL_EPISODES,
    'verbose': VERBOSE
}

# 模型配置 - 基于原始代码的ActorCritic模型
MODEL_CONFIG = {
    'cnn_channels': [32, 64, 64],  # 原始代码: 32, 64, 64
    'cnn_kernels': [8, 4, 3],      # 原始代码: 8, 4, 3
    'cnn_strides': [4, 2, 1],      # 原始代码: 4, 2, 1
    'hidden_size': 512,            # 原始代码: 512
    'activation': 'relu',
    'dropout': 0.0,
    # 原始代码中的线性层输出大小
    'linear_output_size': 3136,    # 原始代码中计算出的特征图大小
    'policy_head_size': 512,       # 原始代码: 512
    'value_head_size': 1           # 原始代码: 1
}

def get_device():
    """获取计算设备"""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_ppo_config():
    """获取PPO配置"""
    return PPO_HYPERPARAMS.copy()

def get_model_config():
    """获取模型配置"""
    return MODEL_CONFIG.copy()

def get_log_config():
    """获取日志配置"""
    return LOG_CONFIG.copy()


