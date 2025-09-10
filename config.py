"""
配置文件：定义支持的模型、环境和游戏
"""

# 支持的模型类型
SUPPORTED_MODELS = {
    'CNNDQN': {
        'name': 'CNN Deep Q-Network',
        'description': '用于深度Q学习的卷积神经网络',
        'algorithms': ['DQN', 'DDQN', 'DuelingDQN']
    },
    'CNNPPO': {
        'name': 'CNN Proximal Policy Optimization',
        'description': '用于近端策略优化的卷积神经网络',
        'algorithms': ['PPO']
    },
    'CNNA2C': {
        'name': 'CNN Advantage Actor-Critic',
        'description': '用于优势演员评论家的卷积神经网络',
        'algorithms': ['A2C', 'A3C']
    }
}

# 支持的游戏环境
SUPPORTED_ENVIRONMENTS = {
    'mario': {
        'name': 'Super Mario Bros',
        'description': '超级马里奥兄弟游戏',
        'env_prefix': 'SuperMarioBros',
        'action_spaces': ['SIMPLE', 'COMPLEX'],
        'models': ['CNNDQN', 'CNNPPO', 'CNNA2C']
    },
    'atari': {
        'name': 'Atari Games',
        'description': 'Atari经典游戏',
        'env_prefix': 'ALE',
        'action_spaces': ['DISCRETE'],
        'models': ['CNNDQN', 'CNNPPO', 'CNNA2C']
    }
}

# 默认配置
DEFAULT_CONFIG = {
    'input_shape': (4, 84, 84),
    'frame_stack': 4,
    'frame_skip': 4,
    'max_episode_steps': 10000,
    'render_fps': 10
}

# 模型检测规则
MODEL_DETECTION_RULES = {
    'CNNDQN': {
        'indicators': ['fc', 'q_values', 'q_network'],
        'output_shape': 'num_actions'
    },
    'CNNPPO': {
        'indicators': ['actor', 'critic', 'policy'],
        'output_shape': 'num_actions'
    },
    'CNNA2C': {
        'indicators': ['actor', 'critic', 'advantage'],
        'output_shape': 'num_actions'
    }
}

# 动作空间检测规则
ACTION_SPACE_DETECTION_RULES = {
    'SIMPLE': {
        'num_actions': 7,
        'indicators': ['simple', 'basic']
    },
    'COMPLEX': {
        'num_actions': 12,
        'indicators': ['complex', 'full']
    },
    'DISCRETE': {
        'num_actions': 'auto',
        'indicators': ['discrete', 'atari']
    }
}
