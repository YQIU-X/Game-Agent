#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO训练脚本 - 使用新的模块化结构
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

# 修复pyglet兼容性问题并设置无头模式
import os
os.environ['PYGLET_HEADLESS'] = '1'  # 设置无头模式
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 禁用SDL视频驱动
os.environ['DISPLAY'] = ''  # 清空显示环境变量

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from python.algorithms.ppo.trainer import PPOTrainer
from python.algorithms.ppo.core.constants import *
from python.games.mario.core.constants import DEFAULT_ENVIRONMENT, DEFAULT_ACTION_SPACE


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO训练脚本')
    
    # 环境参数
    parser.add_argument('--environment', type=str, default=DEFAULT_ENVIRONMENT,
                       help='环境名称')
    parser.add_argument('--action-space', type=str, default=DEFAULT_ACTION_SPACE,
                       choices=['simple', 'complex', 'right_only'],
                       help='动作空间')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                       help='训练episode数')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                       help='折扣因子')
    parser.add_argument('--lambda-gae', type=float, default=LAMBDA_GAE,
                       help='GAE lambda参数')
    parser.add_argument('--clip-eps', type=float, default=CLIP_RATIO,
                       help='PPO裁剪参数')
    parser.add_argument('--vf-coef', type=float, default=VALUE_LOSS_COEF,
                       help='价值函数损失系数')
    parser.add_argument('--ent-coef', type=float, default=ENTROPY_COEF,
                       help='熵系数')
    parser.add_argument('--max-grad-norm', type=float, default=MAX_GRAD_NORM,
                       help='最大梯度范数')
    parser.add_argument('--epochs', type=int, default=PPO_EPOCHS,
                       help='PPO更新轮数')
    parser.add_argument('--minibatch-size', type=int, default=BATCH_SIZE,
                       help='小批次大小')
    parser.add_argument('--num-env', type=int, default=NUM_ENV,
                       help='并行环境数量')
    parser.add_argument('--rollout-steps', type=int, default=ROLLOUT_STEPS,
                       help='rollout步数')
    
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
    parser.add_argument('--eval-frequency', type=int, default=50,
                       help='评估频率')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='评估episode数')
    
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
        'algorithm': 'PPO',
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'lambda_gae': args.lambda_gae,
        'clip_eps': args.clip_eps,
        'vf_coef': args.vf_coef,
        'ent_coef': args.ent_coef,
        'max_grad_norm': args.max_grad_norm,
        'epochs': args.epochs,
        'minibatch_size': args.minibatch_size,
        'num_env': args.num_env,
        'rollout_steps': args.rollout_steps,
        'force_cpu': args.force_cpu,
        'log_frequency': args.log_frequency,
        'save_frequency': args.save_frequency,
        'eval_frequency': args.eval_frequency,
        'eval_episodes': args.eval_episodes
    }
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 设置环境
    trainer.setup_environment(args.environment, args.action_space)
    
    # 开始训练
    save_path = args.save_path if args.save_model else None
    trainer.train(args.episodes, args.render, save_path)


if __name__ == '__main__':
    main()