#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：验证环境兼容性修复
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

print("[INFO] 开始测试环境兼容性修复...")

try:
    # 创建环境
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print("[INFO] 环境创建成功")
    
    # 测试reset
    reset_result = env.reset()
    print(f"[INFO] reset() 返回类型: {type(reset_result)}")
    
    # 兼容旧版Gym：确保obs是numpy数组
    if isinstance(reset_result, tuple):
        obs = reset_result[0]  # 如果是tuple，取第一个元素
        print(f"[INFO] reset() 返回tuple，提取观察: {obs.shape}")
    else:
        obs = reset_result
        print(f"[INFO] reset() 返回数组: {obs.shape}")
    
    # 测试step
    step_result = env.step(0)  # NOOP action
    print(f"[INFO] step() 返回值数量: {len(step_result)}")
    
    # 处理不同版本的返回值
    if len(step_result) == 4:
        # 旧版Gym: (obs, reward, done, info)
        obs, reward, done, info = step_result
        print(f"[INFO] 旧版Gym格式: obs.shape={obs.shape}, reward={reward}, done={done}")
    elif len(step_result) == 5:
        # 新版Gym: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        print(f"[INFO] 新版Gym格式: obs.shape={obs.shape}, reward={reward}, done={done}")
    else:
        raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
    
    print("[INFO] 环境兼容性测试通过！")
    
except Exception as e:
    print(f"[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'env' in locals():
        env.close()

print("[INFO] 测试完成")


