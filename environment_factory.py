"""
环境工厂：根据游戏类型自动创建正确的环境
"""

from typing import Dict, Any, Optional
from environments import MarioEnv, AtariEnv
from environments.base import BaseGameEnv


class EnvironmentFactory:
    """环境工厂类"""
    
    # 环境类型映射
    ENV_CLASSES = {
        'mario': MarioEnv,
        'atari': AtariEnv,
    }
    
    @staticmethod
    def create_environment(env_name: str, action_space_type: str = 'SIMPLE', **kwargs) -> BaseGameEnv:
        """
        创建游戏环境
        
        Args:
            env_name: 环境名称
            action_space_type: 动作空间类型
            
        Returns:
            创建的环境实例
        """
        # 检测游戏类型
        game_type = EnvironmentFactory._detect_game_type(env_name)
        
        if game_type not in EnvironmentFactory.ENV_CLASSES:
            raise ValueError(f"不支持的游戏类型: {game_type}")
        
        env_class = EnvironmentFactory.ENV_CLASSES[game_type]
        return env_class(env_name, action_space_type, **kwargs)
    
    @staticmethod
    def _detect_game_type(env_name: str) -> str:
        """检测游戏类型"""
        env_name_lower = env_name.lower()
        
        if 'mario' in env_name_lower or 'super' in env_name_lower:
            return 'mario'
        elif any(game in env_name_lower for game in ['pong', 'breakout', 'space', 'asteroids']):
            return 'atari'
        else:
            # 默认返回mario
            return 'mario'
