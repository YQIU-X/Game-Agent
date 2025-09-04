"""
脚本模块
包含各种Python脚本
"""

from .agent_inference import main as agent_inference_main
from .player_control import main as player_control_main

__all__ = [
    'agent_inference_main',
    'player_control_main'
]

