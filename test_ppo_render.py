#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO渲染测试脚本
测试PPO环境包装器的渲染功能
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io
import base64

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入PPO专用环境包装器
from python.wrappers.ppo_wrappers import wrap_ppo_environment, get_ppo_actions

def frame_to_base64(frame: np.ndarray) -> str:
    """将帧转换为base64编码"""
    # 确保帧是uint8格式
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    
    # 转换为PIL图像
    img = Image.fromarray(frame)
    
    # 转换为JPEG格式的base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=70)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_ppo_render():
    """测试PPO渲染功能"""
    print("[测试] 开始测试PPO渲染功能", file=sys.stderr)
    
    try:
        # 创建PPO环境
        env = wrap_ppo_environment('SuperMarioBros-1-1-v0', monitor=False)
        print("[测试] PPO环境创建成功", file=sys.stderr)
        
        # 初始化环境
        state = env.reset()
        print(f"[测试] 环境重置成功，观察空间: {env.observation_space.shape}", file=sys.stderr)
        
        # 执行几步动作
        for step in range(5):
            # 随机动作
            action = env.action_space.sample()
            
            # 执行动作
            step_result = env.step(action)
            
            # 处理不同版本的返回值
            if len(step_result) == 4:
                state, reward, done, info = step_result
            elif len(step_result) == 5:
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
            
            print(f"[测试] 步骤 {step+1}: 动作={action}, 奖励={reward:.2f}", file=sys.stderr)
            
            # 测试渲染
            try:
                frame = env.unwrapped.screen
                print(f"[测试] 成功获取帧数据: shape={frame.shape}, dtype={frame.dtype}", file=sys.stderr)
                
                # 测试base64编码
                frame_base64 = frame_to_base64(frame)
                print(f"[测试] 成功编码帧数据: base64长度={len(frame_base64)}", file=sys.stderr)
                
                # 输出JSON格式（模拟推理脚本的输出）
                output = {
                    "type": "frame",
                    "t": step,
                    "agent_action": action,
                    "reward": float(reward),
                    "frame": frame_base64
                }
                print(f"[测试] 帧数据输出成功", file=sys.stderr)
                
            except Exception as e:
                print(f"[测试] 渲染失败: {e}", file=sys.stderr)
                return False
            
            if done:
                print("[测试] 游戏结束", file=sys.stderr)
                break
        
        env.close()
        print("[测试] PPO渲染功能测试成功！", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"[测试] PPO渲染功能测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_ppo_render()
    sys.exit(0 if success else 1)
