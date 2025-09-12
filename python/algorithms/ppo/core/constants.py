#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO算法常量配置
"""

# PPO超参数
PPO_HYPERPARAMS = {
    "environment": "SuperMarioBros-1-1-v0",
    "num_env": 8,
    "rollout_steps": 128,
    "epochs": 4,
    "minibatch_size": 64,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "lambda_gae": 0.95,
    "max_grad_norm": 0.5,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "clip_ratio": 0.2,
    "ppo_epochs": 4,
    "batch_size": 64,
    "num_env": 8,
    "rollout_steps": 128,
    "target_kl": 0.01,
    "anneal_lr": True,
    "gae": True,
    "norm_adv": True,
    "clip_vloss": True,
    "update_epochs": 4,
    "mini_batch_size": 64,
    "save_frequency": 100,
    "log_frequency": 10,
    "eval_frequency": 50,
    "eval_episodes": 5
}

# 训练参数
NUM_EPISODES = 1000
LEARNING_RATE = 3e-4
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
    'save_frequency': 100,
    'log_frequency': 10,
    'eval_frequency': 50,
    'eval_episodes': 5,
    'verbose': True
}

# 模型配置
MODEL_CONFIG = {
    'cnn_channels': [32, 64, 64],
    'cnn_kernels': [8, 4, 3],
    'cnn_strides': [4, 2, 1],
    'hidden_size': 512,
    'activation': 'relu',
    'dropout': 0.0
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

