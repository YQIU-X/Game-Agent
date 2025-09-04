"""
环境工厂
用于动态创建游戏环境
"""

from typing import Dict, Any
from ..environments import MarioEnv, AtariEnv


class EnvironmentFactory:
    """环境工厂类"""
    
    @staticmethod
    def create_environment(env_name: str, action_space_type: str = 'SIMPLE', **kwargs):
        """
        创建游戏环境
        
        Args:
            env_name: 环境名称
            action_space_type: 动作空间类型
            **kwargs: 其他参数
            
        Returns:
            环境实例
        """
        # 根据环境名称判断游戏类型
        if 'SuperMarioBros' in env_name:
            return MarioEnv(env_name, action_space_type, **kwargs)
        elif any(game in env_name for game in ['Breakout', 'Pong', 'SpaceInvaders', 'MsPacman']):
            return AtariEnv(env_name, **kwargs)
        else:
            raise ValueError(f"不支持的环境: {env_name}")
    
    @staticmethod
    def get_supported_environments():
        """获取支持的环境列表"""
        return {
            'mario': [
                'SuperMarioBros-v0', 'SuperMarioBros-v1', 'SuperMarioBros-v2', 'SuperMarioBros-v3',
                'SuperMarioBros2-v0', 'SuperMarioBros2-v1', 'SuperMarioBros2-v2', 'SuperMarioBros2-v3',
                'SuperMarioBros3-v0', 'SuperMarioBros3-v1', 'SuperMarioBros3-v2', 'SuperMarioBros3-v3',
                'SuperMarioBros4-v0', 'SuperMarioBros4-v1', 'SuperMarioBros4-v2', 'SuperMarioBros4-v3',
                'SuperMarioBros5-v0', 'SuperMarioBros5-v1', 'SuperMarioBros5-v2', 'SuperMarioBros5-v3',
                'SuperMarioBros6-v0', 'SuperMarioBros6-v1', 'SuperMarioBros6-v2', 'SuperMarioBros6-v3',
                'SuperMarioBros7-v0', 'SuperMarioBros7-v1', 'SuperMarioBros7-v2', 'SuperMarioBros7-v3',
            ],
            'atari': [
                'Breakout-v0', 'Pong-v0', 'SpaceInvaders-v0', 'MsPacman-v0'
            ]
        }
