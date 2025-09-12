#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
算法配置模板 - 支持多种强化学习算法
"""

# 游戏环境配置
GAME_ENVIRONMENTS = {
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
    },
    'atari': {
        'name': 'Atari Games',
        'environments': [
            'Breakout-v4',
            'Pong-v4',
            'SpaceInvaders-v4',
            'MsPacman-v4',
            'Qbert-v4',
            'Seaquest-v4'
        ],
        'action_spaces': {
            'discrete': 'DISCRETE',
            'minimal': 'MINIMAL'
        }
    }
}

# 算法配置模板
ALGORITHM_CONFIGS = {
    'DQN': {
        'name': 'Deep Q-Network',
        'script': 'python/scripts/train_dqn.py',
        'description': '基于深度Q网络的强化学习算法',
        'parameters': {
            'learning_rate': {
                'type': 'float',
                'default': 1e-4,
                'min': 1e-6,
                'max': 1e-2,
                'step': 1e-5,
                'precision': 6,
                'description': '学习率'
            },
            'gamma': {
                'type': 'float',
                'default': 0.99,
                'min': 0.1,
                'max': 1.0,
                'step': 0.01,
                'precision': 2,
                'description': '折扣因子'
            },
            'epsilon_start': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 1.0,
                'step': 0.1,
                'precision': 1,
                'description': '初始探索率'
            },
            'epsilon_end': {
                'type': 'float',
                'default': 0.01,
                'min': 0.001,
                'max': 0.1,
                'step': 0.001,
                'precision': 3,
                'description': '最终探索率'
            },
            'epsilon_decay': {
                'type': 'int',
                'default': 100000,
                'min': 10000,
                'max': 1000000,
                'step': 10000,
                'description': '探索率衰减步数'
            },
            'batch_size': {
                'type': 'int',
                'default': 32,
                'min': 16,
                'max': 128,
                'step': 16,
                'description': '批次大小'
            },
            'memory_size': {
                'type': 'int',
                'default': 10000,
                'min': 1000,
                'max': 100000,
                'step': 1000,
                'description': '经验回放缓冲区大小'
            },
            'target_update': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'step': 100,
                'description': '目标网络更新频率'
            }
        },
        'flags': {
            'render': {
                'type': 'boolean',
                'default': False,
                'description': '启用渲染'
            },
            'save_model': {
                'type': 'boolean',
                'default': True,
                'description': '保存模型'
            }
        }
    },
    
    'PPO': {
        'name': 'Proximal Policy Optimization',
        'script': 'python/scripts/train_ppo.py',
        'description': '近端策略优化算法',
        'parameters': {
            'learning_rate': {
                'type': 'float',
                'default': 3e-4,
                'min': 1e-6,
                'max': 1e-2,
                'step': 1e-5,
                'precision': 6,
                'description': '学习率'
            },
            'gamma': {
                'type': 'float',
                'default': 0.99,
                'min': 0.1,
                'max': 1.0,
                'step': 0.01,
                'precision': 2,
                'description': '折扣因子'
            },
            'lambda_gae': {
                'type': 'float',
                'default': 0.95,
                'min': 0.1,
                'max': 1.0,
                'step': 0.01,
                'precision': 2,
                'description': 'GAE lambda参数'
            },
            'clip_eps': {
                'type': 'float',
                'default': 0.2,
                'min': 0.1,
                'max': 0.5,
                'step': 0.01,
                'precision': 2,
                'description': 'PPO裁剪参数'
            },
            'vf_coef': {
                'type': 'float',
                'default': 0.5,
                'min': 0.1,
                'max': 1.0,
                'step': 0.1,
                'precision': 1,
                'description': '价值函数损失系数'
            },
            'ent_coef': {
                'type': 'float',
                'default': 0.01,
                'min': 0.001,
                'max': 0.1,
                'step': 0.001,
                'precision': 3,
                'description': '熵系数'
            },
            'max_grad_norm': {
                'type': 'float',
                'default': 0.5,
                'min': 0.1,
                'max': 2.0,
                'step': 0.1,
                'precision': 1,
                'description': '最大梯度范数'
            },
            'epochs': {
                'type': 'int',
                'default': 4,
                'min': 1,
                'max': 20,
                'step': 1,
                'description': 'PPO更新轮数'
            },
            'minibatch_size': {
                'type': 'int',
                'default': 64,
                'min': 16,
                'max': 256,
                'step': 16,
                'description': '小批次大小'
            },
            'num_env': {
                'type': 'int',
                'default': 8,
                'min': 1,
                'max': 32,
                'step': 1,
                'description': '并行环境数量'
            },
            'rollout_steps': {
                'type': 'int',
                'default': 128,
                'min': 32,
                'max': 512,
                'step': 32,
                'description': 'rollout步数'
            }
        },
        'flags': {
            'render': {
                'type': 'boolean',
                'default': False,
                'description': '启用渲染'
            },
            'save_model': {
                'type': 'boolean',
                'default': True,
                'description': '保存模型'
            }
        }
    },
    
    'A2C': {
        'name': 'Advantage Actor-Critic',
        'script': 'python/scripts/train_a2c.py',
        'description': '优势演员-评论家算法',
        'parameters': {
            'learning_rate': {
                'type': 'float',
                'default': 7e-4,
                'min': 1e-6,
                'max': 1e-2,
                'step': 1e-5,
                'precision': 6,
                'description': '学习率'
            },
            'gamma': {
                'type': 'float',
                'default': 0.99,
                'min': 0.1,
                'max': 1.0,
                'step': 0.01,
                'precision': 2,
                'description': '折扣因子'
            },
            'value_loss_coef': {
                'type': 'float',
                'default': 0.5,
                'min': 0.1,
                'max': 1.0,
                'step': 0.1,
                'precision': 1,
                'description': '价值损失系数'
            },
            'entropy_coef': {
                'type': 'float',
                'default': 0.01,
                'min': 0.001,
                'max': 0.1,
                'step': 0.001,
                'precision': 3,
                'description': '熵系数'
            },
            'max_grad_norm': {
                'type': 'float',
                'default': 0.5,
                'min': 0.1,
                'max': 2.0,
                'step': 0.1,
                'precision': 1,
                'description': '最大梯度范数'
            },
            'batch_size': {
                'type': 'int',
                'default': 32,
                'min': 16,
                'max': 128,
                'step': 16,
                'description': '批次大小'
            }
        },
        'flags': {
            'render': {
                'type': 'boolean',
                'default': False,
                'description': '启用渲染'
            },
            'save_model': {
                'type': 'boolean',
                'default': True,
                'description': '保存模型'
            }
        }
    }
}

# 通用训练参数
COMMON_PARAMETERS = {
    'episodes': {
        'type': 'int',
        'default': 1000,
        'min': 100,
        'max': 100000,
        'step': 100,
        'description': '训练轮数'
    },
    'max_steps_per_episode': {
        'type': 'int',
        'default': 10000,
        'min': 1000,
        'max': 50000,
        'step': 1000,
        'description': '每轮最大步数'
    },
    'save_frequency': {
        'type': 'int',
        'default': 100,
        'min': 10,
        'max': 1000,
        'step': 10,
        'description': '模型保存频率'
    },
    'log_frequency': {
        'type': 'int',
        'default': 10,
        'min': 1,
        'max': 100,
        'step': 1,
        'description': '日志输出频率'
    }
}

# 通用标志
COMMON_FLAGS = {
    'render': {
        'type': 'boolean',
        'default': False,
        'description': '启用渲染'
    },
    'save_model': {
        'type': 'boolean',
        'default': True,
        'description': '保存模型'
    },
    'use_gpu': {
        'type': 'boolean',
        'default': True,
        'description': '使用GPU'
    },
    'verbose': {
        'type': 'boolean',
        'default': True,
        'description': '详细输出'
    }
}

def get_algorithm_config(algorithm_name):
    """获取算法配置"""
    return ALGORITHM_CONFIGS.get(algorithm_name, {})

def get_game_config(game_name):
    """获取游戏配置"""
    return GAME_ENVIRONMENTS.get(game_name, {})

def get_all_algorithms():
    """获取所有可用算法"""
    return list(ALGORITHM_CONFIGS.keys())

def get_all_games():
    """获取所有可用游戏"""
    return list(GAME_ENVIRONMENTS.keys())

def get_algorithm_parameters(algorithm_name):
    """获取算法参数配置"""
    config = get_algorithm_config(algorithm_name)
    if not config:
        return {}
    
    # 合并通用参数和算法特定参数
    params = COMMON_PARAMETERS.copy()
    params.update(config.get('parameters', {}))
    return params

def get_algorithm_flags(algorithm_name):
    """获取算法标志配置"""
    config = get_algorithm_config(algorithm_name)
    if not config:
        return {}
    
    # 合并通用标志和算法特定标志
    flags = COMMON_FLAGS.copy()
    flags.update(config.get('flags', {}))
    return flags
