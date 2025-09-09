#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
玩家控制脚本：完全独立的玩家游戏控制
与智能体控制彻底分开，使用原始环境进行玩家操作
"""

import argparse
import sys
import time
import json
import base64
import numpy as np
from PIL import Image
import io
import os
import threading
from queue import Queue

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


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
    parser = argparse.ArgumentParser(description='玩家控制脚本')
    parser.add_argument('--env', type=str, required=True, help='游戏环境名称')
    parser.add_argument('--action-space', type=str, default='SIMPLE', help='动作空间类型 (SIMPLE/COMPLEX)')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    
    args = parser.parse_args()
    
    try:
        # 选择动作空间
        if args.action_space == 'SIMPLE':
            actions = SIMPLE_MOVEMENT
        else:
            actions = COMPLEX_MOVEMENT
        
        # 创建原始环境（不使用任何包装器，保持原始环境）
        env = gym_super_mario_bros.make(args.env)
        env = JoypadSpace(env, actions)
        print(f"[INFO] 玩家环境创建成功", file=sys.stderr)
        
        # 获取动作列表
        action_list = actions
        
        # 初始化环境
        obs = env.reset()
        
        total_reward = 0.0
        frame_count = 0
        current_action = 0  # 默认无动作（NOOP）
        
        # 创建输入队列
        input_queue = Queue()
        
        # 输入处理线程
        def input_handler():
            while True:
                try:
                    input_line = input()
                    if not input_line.strip():
                        continue
                    
                    data = json.loads(input_line)
                    action_index = data.get('action', 0)
                    input_queue.put(action_index)
                    
                except (EOFError, KeyboardInterrupt):
                    input_queue.put(None)  # 结束信号
                    break
                except json.JSONDecodeError:
                    print(f"[ERROR] 无效的JSON输入", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"[ERROR] 输入处理错误: {e}", file=sys.stderr)
                    break
        
        # 启动输入处理线程（非守护线程）
        input_thread = threading.Thread(target=input_handler, daemon=False)
        input_thread.start()
        
        print(f"[INFO] 开始玩家控制，FPS: {args.fps}", file=sys.stderr)
        print(f"[INFO] 等待前端输入...", file=sys.stderr)
        
        # 游戏主循环
        while True:
            start_time = time.time()
            
            # 检查是否有新的输入（非阻塞）
            try:
                while not input_queue.empty():
                    new_action = input_queue.get_nowait()
                    if new_action is None:  # 结束信号
                        print(f"[INFO] 玩家控制结束", file=sys.stderr)
                        return 0
                    current_action = new_action
            except:
                pass
            
            # 执行当前动作
            action_name = action_list[current_action]
            
            # 环境步进
            obs, reward, done, info = env.step(current_action)
            
            total_reward += float(reward)
            
            # 渲染原始帧用于显示
            frame = env.render(mode='rgb_array')
            frame_base64 = frame_to_base64(frame)
            
            # 输出JSON（减少延迟）
            output = {
                "type": "player_frame",
                "t": frame_count,
                "agent_action": action_name,
                "reward": float(reward),
                "frame": frame_base64
            }
            print(json.dumps(output, ensure_ascii=False), flush=True)
            
            # 检查是否获得旗帜
            if info.get('flag_get', False):
                print(f"[INFO] 玩家获得旗帜！", file=sys.stderr)
            
            # 游戏结束后停止循环
            if done:
                print(f"[INFO] 游戏结束，总奖励: {total_reward}", file=sys.stderr)
                break
            
            frame_count += 1
            
            # 控制帧率 - 高刷新率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"[ERROR] 玩家控制失败: {e}", file=sys.stderr)
        return 1
    
    finally:
        if 'env' in locals():
            env.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
