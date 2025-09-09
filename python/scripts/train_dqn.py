#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN训练脚本 - 使用新的模块化结构
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from python.algorithms.dqn import DQNTrainer
from python.algorithms.dqn.core.constants import *


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DQN训练脚本')
    
    # 环境参数
    parser.add_argument('--environment', type=str, default=ENVIRONMENT,
                       help='环境名称')
    parser.add_argument('--action-space', type=str, default=ACTION_SPACE,
                       choices=['simple', 'complex', 'right_only'],
                       help='动作空间')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                       help='训练episode数')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                       help='折扣因子')
    parser.add_argument('--epsilon-start', type=float, default=EPSILON_START,
                       help='初始探索率')
    parser.add_argument('--epsilon-final', type=float, default=EPSILON_FINAL,
                       help='最终探索率')
    parser.add_argument('--epsilon-decay', type=int, default=EPSILON_DECAY,
                       help='探索率衰减步数')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--memory-capacity', type=int, default=MEMORY_CAPACITY,
                       help='经验回放缓冲区容量')
    parser.add_argument('--target-update-frequency', type=int, default=TARGET_UPDATE_FREQUENCY,
                       help='目标网络更新频率')
    parser.add_argument('--initial-learning', type=int, default=INITIAL_LEARNING,
                       help='初始学习步数')
    parser.add_argument('--beta-start', type=float, default=BETA_START,
                       help='初始beta值')
    parser.add_argument('--beta-frames', type=int, default=BETA_FRAMES,
                       help='beta更新帧数')
    
    # 其他参数
    parser.add_argument('--render', action='store_true',
                       help='启用渲染')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='保存模型')
    parser.add_argument('--force-cpu', action='store_true',
                       help='强制使用CPU')
    parser.add_argument('--save-path', type=str, default='models',
                       help='模型保存路径')
    parser.add_argument('--log-frequency', type=int, default=10,
                       help='日志输出频率')
    parser.add_argument('--save-frequency', type=int, default=100,
                       help='模型保存频率')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建保存目录
    if args.save_model:
        os.makedirs(args.save_path, exist_ok=True)
    
    # 创建训练配置
    config = {
        'environment': args.environment,
        'action_space': args.action_space,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_final': args.epsilon_final,
        'epsilon_decay': args.epsilon_decay,
        'batch_size': args.batch_size,
        'memory_capacity': args.memory_capacity,
        'target_update_frequency': args.target_update_frequency,
        'initial_learning': args.initial_learning,
        'beta_start': args.beta_start,
        'beta_frames': args.beta_frames,
        'force_cpu': args.force_cpu,
        'log_frequency': args.log_frequency,
        'save_frequency': args.save_frequency
    }
    
    # 创建训练器
    trainer = DQNTrainer(config)
    
    print("开始训练")
    # 设置环境
    trainer.setup_environment(args.environment, args.action_space)
    
    # 开始训练
    save_path = args.save_path if args.save_model else None
    trainer.train(args.episodes, args.render, save_path)


if __name__ == '__main__':
    main()