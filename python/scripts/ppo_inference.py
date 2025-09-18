#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO专用推理脚本
使用PPO专用环境包装器
"""

import argparse
import sys
import time
import json
import base64
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入PPO专用环境包装器
from python.wrappers.ppo_wrappers import wrap_ppo_environment, get_ppo_actions

# PPO Actor-Critic模型
class ActorCritic(nn.Module):
    """The Actor-Critic neural network architecture."""
    def __init__(self, n_frame, act_dim):
        super().__init__()
        # Convolutional layers to process the stacked frames
        self.net = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        # Fully connected layers for policy and value heads
        self.linear = nn.Linear(3136, 512)
        self.policy_head = nn.Linear(512, act_dim)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # Permute dimensions for PyTorch's Conv2d layer (batch, channels, height, width)
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.net(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear(x))

        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, obs):
        """Select an action based on the current observation."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

# PPO环境包装器已移至 python/wrappers/ppo_wrappers.py

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
    parser = argparse.ArgumentParser(description='PPO专用推理脚本')
    parser.add_argument('--env', type=str, required=True, help='游戏环境名称')
    parser.add_argument('--weights', type=str, required=True, help='权重文件路径')
    parser.add_argument('--fps', type=int, default=5, help='帧率')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--algorithm', type=str, help='算法类型 (兼容参数，PPO专用脚本)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 获取权重文件路径
    if os.path.isabs(args.weights):
        wpath = args.weights
    else:
        # 相对于项目根目录
        wpath = os.path.join(project_root, args.weights)
    
    print(f"[INFO] PPO推理开始", file=sys.stderr)
    print(f"[INFO] 环境: {args.env}", file=sys.stderr)
    print(f"[INFO] 权重文件: {wpath}", file=sys.stderr)
    print(f"[INFO] 设备: {device}", file=sys.stderr)
    
    try:
        # 创建PPO专用环境
        env = wrap_ppo_environment(args.env, monitor=False)
        print(f"[INFO] PPO专用环境创建成功", file=sys.stderr)
        
        # 获取PPO动作列表
        actions = get_ppo_actions()
        print(f"[INFO] 动作空间: {len(actions)} 个动作", file=sys.stderr)
        
        # 创建PPO模型
        obs_dim = env.observation_space.shape[-1]  # 4 (帧数)
        act_dim = len(actions)  # 12
        model = ActorCritic(obs_dim, act_dim).to(device)
        
        # 加载权重
        weights = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        print(f"[INFO] PPO模型加载成功", file=sys.stderr)
        
        # 初始化环境
        state = env.reset()
        
        # 兼容旧版Gym：确保state是numpy数组
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0.0
        frame_count = 0
        
        print(f"[INFO] 开始PPO推理，FPS: {args.fps}", file=sys.stderr)
        
        while True:
            start_time = time.time()
            
            # PPO推理逻辑
            state_v = torch.tensor(np.array([state], copy=True), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits, value = model(state_v)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()
            
            # 执行动作
            action_name = actions[action]
            
            # 环境步进
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
            frame = env.render(mode='rgb_array')
            frame_base64 = frame_to_base64(frame)
            
            # 输出JSON
            current_time = time.time()
            output = {
                "type": "frame",
                "t": frame_count,
                "timestamp": current_time,
                "agent_action": action_name,
                "reward": float(reward),
                "frame": frame_base64
            }
            print(json.dumps(output, ensure_ascii=False), flush=True)
            
            # 检查是否获得旗帜
            if info.get('flag_get', False):
                print(f"[INFO] PPO智能体获得旗帜！", file=sys.stderr)
            
            # 游戏结束后停止循环
            if done:
                print(f"[INFO] PPO游戏结束，总奖励: {total_reward}", file=sys.stderr)
                break
            
            frame_count += 1
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"[ERROR] PPO推理失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if 'env' in locals():
            env.close()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
