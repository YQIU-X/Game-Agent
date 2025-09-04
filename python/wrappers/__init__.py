"""
环境包装器模块
提供各种游戏环境的包装器，用于统一环境接口
"""

from .mario_wrappers import wrap_environment
from .base_wrapper import BaseWrapper

__all__ = [
    'wrap_environment',
    'BaseWrapper'
]
