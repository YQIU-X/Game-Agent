#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mario游戏包
"""

from .wrappers import (
    FrameDownsample,
    MaxAndSkipEnv,
    FireResetEnv,
    FrameBuffer,
    ImageToPyTorch,
    NormalizeFloats,
    CustomReward,
    wrap_mario_environment
)
from .constants import *

__all__ = [
    'FrameDownsample',
    'MaxAndSkipEnv',
    'FireResetEnv',
    'FrameBuffer',
    'ImageToPyTorch',
    'NormalizeFloats',
    'CustomReward',
    'wrap_mario_environment',
    'MARIO_ENVIRONMENTS',
    'DEFAULT_ENVIRONMENT',
    'ACTION_SPACES',
    'DEFAULT_ACTION_SPACE'
]
