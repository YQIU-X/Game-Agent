"""
游戏代理平台配置文件
定义支持的模型、环境和默认设置
"""

# 支持的模型类型
SUPPORTED_MODELS = {
    'CNNDQN': {
        'name': 'CNN DQN',
        'description': '卷积神经网络深度Q网络',
        'supported_games': ['mario', 'atari'],
        'default_fps': 30
    },
    'CNNPPO': {
        'name': 'CNN PPO', 
        'description': '卷积神经网络近端策略优化',
        'supported_games': ['mario', 'atari'],
        'default_fps': 30
    },
    'CNNA2C': {
        'name': 'CNN A2C',
        'description': '卷积神经网络优势演员评论家',
        'supported_games': ['mario', 'atari'],
        'default_fps': 30
    }
}

# 支持的游戏环境
SUPPORTED_ENVIRONMENTS = {
    'mario': {
        'name': 'Super Mario Bros',
        'description': '超级马里奥兄弟',
        'supported_models': ['CNNDQN', 'CNNPPO', 'CNNA2C'],
        'action_spaces': ['SIMPLE', 'COMPLEX'],
        'levels': [
            {'id': 'SuperMarioBros-v0', 'name': '1-1', 'description': '第一关第一小关'},
            {'id': 'SuperMarioBros-v1', 'name': '1-2', 'description': '第一关第二小关'},
            {'id': 'SuperMarioBros-v2', 'name': '1-3', 'description': '第一关第三小关'},
            {'id': 'SuperMarioBros-v3', 'name': '1-4', 'description': '第一关第四小关'},
            {'id': 'SuperMarioBros2-v0', 'name': '2-1', 'description': '第二关第一小关'},
            {'id': 'SuperMarioBros2-v1', 'name': '2-2', 'description': '第二关第二小关'},
            {'id': 'SuperMarioBros2-v2', 'name': '2-3', 'description': '第二关第三小关'},
            {'id': 'SuperMarioBros2-v3', 'name': '2-4', 'description': '第二关第四小关'},
            {'id': 'SuperMarioBros3-v0', 'name': '3-1', 'description': '第三关第一小关'},
            {'id': 'SuperMarioBros3-v1', 'name': '3-2', 'description': '第三关第二小关'},
            {'id': 'SuperMarioBros3-v2', 'name': '3-3', 'description': '第三关第三小关'},
            {'id': 'SuperMarioBros3-v3', 'name': '3-4', 'description': '第三关第四小关'},
            {'id': 'SuperMarioBros4-v0', 'name': '4-1', 'description': '第四关第一小关'},
            {'id': 'SuperMarioBros4-v1', 'name': '4-2', 'description': '第四关第二小关'},
            {'id': 'SuperMarioBros4-v2', 'name': '4-3', 'description': '第四关第三小关'},
            {'id': 'SuperMarioBros4-v3', 'name': '4-4', 'description': '第四关第四小关'},
            {'id': 'SuperMarioBros5-v0', 'name': '5-1', 'description': '第五关第一小关'},
            {'id': 'SuperMarioBros5-v1', 'name': '5-2', 'description': '第五关第二小关'},
            {'id': 'SuperMarioBros5-v2', 'name': '5-3', 'description': '第五关第三小关'},
            {'id': 'SuperMarioBros5-v3', 'name': '5-4', 'description': '第五关第四小关'},
            {'id': 'SuperMarioBros6-v0', 'name': '6-1', 'description': '第六关第一小关'},
            {'id': 'SuperMarioBros6-v1', 'name': '6-2', 'description': '第六关第二小关'},
            {'id': 'SuperMarioBros6-v2', 'name': '6-3', 'description': '第六关第三小关'},
            {'id': 'SuperMarioBros6-v3', 'name': '6-4', 'description': '第六关第四小关'},
            {'id': 'SuperMarioBros7-v0', 'name': '7-1', 'description': '第七关第一小关'},
            {'id': 'SuperMarioBros7-v1', 'name': '7-2', 'description': '第七关第二小关'},
            {'id': 'SuperMarioBros7-v2', 'name': '7-3', 'description': '第七关第三小关'},
            {'id': 'SuperMarioBros7-v3', 'name': '7-4', 'description': '第七关第四小关'},
        ]
    },
    'atari': {
        'name': 'Atari Games',
        'description': '雅达利游戏',
        'supported_models': ['CNNDQN', 'CNNPPO', 'CNNA2C'],
        'action_spaces': ['DISCRETE'],
        'levels': [
            {'id': 'Breakout-v0', 'name': 'Breakout', 'description': '打砖块'},
            {'id': 'Pong-v0', 'name': 'Pong', 'description': '乒乓球'},
            {'id': 'SpaceInvaders-v0', 'name': 'Space Invaders', 'description': '太空侵略者'},
            {'id': 'MsPacman-v0', 'name': 'Ms. Pacman', 'description': '吃豆人小姐'},
        ]
    }
}

# 默认设置
DEFAULT_SETTINGS = {
    'fps': 30,
    'action_space': 'SIMPLE',
    'model_type': 'CNNDQN',
    'game_type': 'mario',
    'level': 'SuperMarioBros-v0'
}

# 文件路径配置
PATHS = {
    'pretrained_models': 'pretrained_models',
    'python_scripts': 'python/scripts',
    'python_utils': 'python/utils',
    'python_wrappers': 'python/wrappers',
    'python_configs': 'python/configs'
}
