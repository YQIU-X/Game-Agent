#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用推理脚本：支持多种游戏和算法
"""

import argparse
import sys
import time
import json
import base64
import numpy as np
import torch
from PIL import Image
import io
import os

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from python.utils.model_factory import ModelFactory
from python.utils.environment_factory import EnvironmentFactory


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


def main():
    parser = argparse.ArgumentParser(description='通用游戏推理脚本')
    parser.add_argument('--env', type=str, required=True, help='游戏环境名称')
    parser.add_argument('--weights', type=str, required=True, help='权重文件路径')
    parser.add_argument('--fps', type=int, default=10, help='帧率')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 获取权重文件路径
    if os.path.isabs(args.weights):
        wpath = args.weights
    else:
        # 相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        wpath = os.path.join(project_root, args.weights)
    print(f"[INFO] 检测权重文件: {wpath}", file=sys.stderr)
    
    try:
        # 检测模型信息
        model_info = ModelFactory.detect_model_info(wpath)
        print(f"[INFO] 检测到模型类型: {model_info['model_type']}", file=sys.stderr)
        print(f"[INFO] 检测到动作空间: {model_info['action_space']}", file=sys.stderr)
        
        # 创建模型
        model = ModelFactory.create_model(model_info)
        model = model.to(device)
        model.eval()
        print(f"[INFO] 模型加载成功", file=sys.stderr)
        
        # 创建环境
        if model_info['model_type'] == 'CNNDQN':
            # CNNDQN模型使用专用环境包装器
            from python.wrappers.mario_wrappers import wrap_environment
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
            
            if model_info['action_space'] == 'SIMPLE':
                actions = SIMPLE_MOVEMENT
            else:
                actions = COMPLEX_MOVEMENT
            
            env = wrap_environment(args.env, actions, monitor=False)
            print(f"[INFO] 使用DQN专用环境包装器", file=sys.stderr)
        else:
            # 其他模型使用通用环境工厂
            env = EnvironmentFactory.create_environment(
                args.env, 
                action_space_type=model_info['action_space']
            )
            print(f"[INFO] 使用通用环境工厂", file=sys.stderr)
        
        # 获取动作列表
        if model_info['model_type'] == 'CNNDQN':
            # CNNDQN模型使用动作空间
            actions = actions  # 已经在上面定义了
        else:
            # 其他模型使用环境工厂
            actions = env.get_actions()
        
        # 初始化环境 - 完全按照你的代码逻辑
        state = env.reset()
        
        # 兼容旧版Gym：确保state是numpy数组
        if isinstance(state, tuple):
            state = state[0]  # 如果是tuple，取第一个元素
        
        total_reward = 0.0
        frame_count = 0
        
        print(f"[INFO] 开始推理，FPS: {args.fps}", file=sys.stderr)
        
        while True:
            start_time = time.time()
            
            # 完全按照你的代码逻辑：直接使用环境包装器处理后的观察
            state_v = torch.tensor(np.array([state], copy=False))
            state_v = state_v.to(device)
            
            # 模型推理
            with torch.no_grad():
                q_vals = model(state_v).data.cpu().numpy()[0]
                action = np.argmax(q_vals)
            
            # 执行动作
            action_name = actions[action]
            
            # 环境步进 - 兼容旧版Gym
            step_result = env.step(action)
            
            # 处理不同版本的返回值
            if len(step_result) == 4:
                # 旧版Gym: (obs, reward, done, info)
                state, reward, done, info = step_result
            elif len(step_result) == 5:
                # 新版Gym: (obs, reward, terminated, truncated, info)
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
            total_reward += reward
            
            # 渲染帧用于显示
            frame = env.unwrapped.screen
            frame_base64 = frame_to_base64(frame)
            
            # 输出JSON
            output = {
                "type": "frame",
                "t": frame_count,
                "agent_action": action_name,
                "reward": float(reward),
                "frame": frame_base64
            }
            print(json.dumps(output, ensure_ascii=False))
            
            # 检查是否获得旗帜
            if info.get('flag_get', False):
                print(f"[INFO] 获得旗帜！", file=sys.stderr)
            
            # 游戏结束后停止循环
            if done:
                print(f"[INFO] 游戏结束，总奖励: {total_reward}", file=sys.stderr)
                break
            
            frame_count += 1
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"[ERROR] 推理失败: {e}", file=sys.stderr)
        return 1
    
    finally:
        if 'env' in locals():
            env.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
