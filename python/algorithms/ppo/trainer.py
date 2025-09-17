#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO训练器
"""

import os
import sys
import json
import time
import argparse
import csv
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 应用pyglet兼容性补丁
try:
    from python.utils.pyglet_patch import patch_pyglet
    patch_pyglet()
except ImportError:
    # 如果补丁文件不存在，设置环境变量
    os.environ['PYGLET_HEADLESS'] = '1'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 禁用SDL视频驱动
    os.environ['DISPLAY'] = ''  # 清空显示环境变量

import numpy as np
import torch
import torch.optim as optim
import gym.vector

# 导入PPO相关模块
from python.algorithms.ppo.core import (
    ActorCritic, compute_gae_batch, rollout_with_bootstrap, 
    evaluate_policy, get_reward, get_device, get_ppo_config
)

# 导入Mario相关模块
from python.games.mario.core import wrap_mario_environment, ACTION_SPACES
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# 导入实验管理器
from python.utils.experiment_manager import experiment_manager


class PPOTrainer:
    """PPO训练器类"""
    
    def __init__(self, config):
        """
        初始化PPO训练器
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = get_device() if not config.get('force_cpu', False) else 'cpu'
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda_gae', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.epochs = config.get('epochs', 4)
        self.minibatch_size = config.get('minibatch_size', 64)
        self.num_env = config.get('num_env', 8)
        self.rollout_steps = config.get('rollout_steps', 128)
        
        # 训练状态
        self.update = 0
        self.best_reward = -float('inf')
        self.training_metrics = []
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        
        # 环境
        self.envs = None
        
        # 日志
        self.log_frequency = config.get('log_frequency', 10)
        self.save_frequency = config.get('save_frequency', 100)
        self.eval_frequency = config.get('eval_frequency', 50)
        self.eval_episodes = config.get('eval_episodes', 5)
        
        # 实验管理
        self.experiment_dir = None
        self.metrics_path = None
        self.logs_path = None
        self.weights_dir = None
        
    def setup_experiment(self, environment, algorithm='PPO'):
        """设置实验目录和文件路径"""
        # 创建实验目录
        self.experiment_dir = experiment_manager.create_experiment_dir(
            environment, algorithm, self.config
        )
        
        # 设置文件路径
        self.metrics_path = experiment_manager.get_metrics_path(self.experiment_dir)
        self.logs_path = experiment_manager.get_logs_path(self.experiment_dir)
        self.weights_dir = experiment_manager.get_weights_dir(self.experiment_dir)
        
        print(f"[PPO Training] 实验目录: {self.experiment_dir}", flush=True)
        
    def setup_csv_logging(self, environment):
        """设置CSV日志记录"""
        if not self.metrics_path:
            # 如果没有设置实验目录，使用旧的方式
            os.makedirs('training_metrics', exist_ok=True)
            self.csv_filename = f'training_metrics/{environment}_ppo.csv'
        else:
            # 使用新的实验目录
            self.csv_filename = self.metrics_path
        
        # 写入CSV头部
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_return', 'max_stage', 'flag_get', 'avg_total_reward', 'timestamp'])
        
        print(f"[PPO Training] CSV log file created: {self.csv_filename}", flush=True)
    
    def save_metrics_to_csv(self, episode, avg_return, max_stage, flag_get, avg_total_reward):
        """保存训练指标到CSV文件"""
        if self.csv_filename:
            # 准备数据
            metrics_data = [
                episode,
                round(avg_return, 3),
                max_stage,
                flag_get,
                round(avg_total_reward, 3),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            
            # 写入CSV
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(metrics_data)
        
    def setup_environment(self, environment, action_space):
        """
        设置环境
        Args:
            environment: 环境名称
            action_space: 动作空间
        """
        # 获取动作空间
        if action_space in ACTION_SPACES:
            action_space_obj = globals()[ACTION_SPACES[action_space]]
        else:
            action_space_obj = COMPLEX_MOVEMENT
            
        # 创建向量化环境
        def make_env():
            return wrap_mario_environment(environment, action_space_obj)
        
        self.envs = gym.vector.SyncVectorEnv([make_env for _ in range(self.num_env)])
        
        # 初始化模型
        obs_dim = self.envs.single_observation_space.shape[-1]
        act_dim = self.envs.single_action_space.n
        
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.model.device = self.device
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"[PPO Training] Observation space dim: {obs_dim}, Action space dim: {act_dim}", flush=True)
        
    def train(self, num_episodes, render=False, save_path=None):
        """
        开始训练
        Args:
            num_episodes: 训练episode数
            render: 是否渲染
            save_path: 模型保存路径（兼容旧接口）
        """
        print(f"[PPO Training] Starting training with {num_episodes} episodes", flush=True)
        print(f"[PPO Training] Device: {self.device}", flush=True)
        print(f"[PPO Training] Environment: {self.config.get('environment', 'Unknown')}", flush=True)
        print(f"[PPO Training] Action Space: {self.config.get('action_space', 'Unknown')}", flush=True)
        
        # 设置实验目录
        environment = self.config.get('environment', 'Unknown')
        algorithm = self.config.get('algorithm', 'PPO')
        self.setup_experiment(environment, algorithm)
        
        # 设置CSV日志记录
        self.setup_csv_logging(environment)
        
        start_time = time.time()
        init_obs = self.envs.reset()
        
        while self.update < num_episodes:
            self.update += 1
            
            # 执行rollout
            batch = rollout_with_bootstrap(self.envs, self.model, self.rollout_steps, init_obs, get_reward)
            init_obs = batch["last_obs"]
            
            T, N = self.rollout_steps, self.num_env
            total_size = T * N
            
            obs = batch["obs"].reshape(total_size, *self.envs.single_observation_space.shape)
            act = batch["actions"].reshape(total_size)
            logp_old = batch["logprobs"].reshape(total_size)
            adv = batch["advantages"].reshape(total_size)
            ret = batch["returns"].reshape(total_size)
            
            # PPO更新
            for _ in range(self.epochs):
                idx = torch.randperm(total_size)
                for start in range(0, total_size, self.minibatch_size):
                    i = idx[start : start + self.minibatch_size]
                    
                    logits, value = self.model(obs[i])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(act[i])
                    ratio = torch.exp(logp - logp_old[i])
                    
                    clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[i]
                    policy_loss = -torch.min(ratio * adv[i], clipped).mean()
                    
                    value_loss = (ret[i] - value).pow(2).mean()
                    entropy = dist.entropy().mean()
                    
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            
            # 记录指标
            avg_return = batch["returns"].mean().item()
            max_stage = batch["max_stage"]
            flag_get = batch["flag_get"]
            avg_total_reward = batch["avg_total_reward"]
            
            # 保存指标到CSV
            self.save_metrics_to_csv(self.update, avg_return, max_stage, flag_get, avg_total_reward)
            
            # 日志输出
            if self.update % self.log_frequency == 0:
                elapsed_time = time.time() - start_time
                print(f"[PPO Training] Update {self.update}: avg return = {avg_return:.2f} "
                      f"max_stage={max_stage} flag_get={flag_get} "
                      f"avg_total_reward={avg_total_reward:.2f} "
                      f"Time: {elapsed_time:.1f}s", flush=True)
            
            # 评估和保存模型
            if self.update % self.eval_frequency == 0:
                should_save_video = False
                if avg_return > self.best_reward:
                    should_save_video = True
                    print(f"[PPO Training] New training high score detected! Saving video...", flush=True)
                if flag_get:
                    should_save_video = True
                    print(f"[PPO Training] Stage cleared! Saving video...", flush=True)
                
                self.best_reward = max(self.best_reward, avg_return)
                
                if should_save_video:
                    monitor_path = f"{self.experiment_dir}/run_{self.update}"
                    os.makedirs(monitor_path, exist_ok=True)
                    eval_env_with_monitor = wrap_mario_environment(
                        environment, COMPLEX_MOVEMENT, monitor=True, monitor_path=monitor_path
                    )
                    
                    avg_score, info, eval_max_stage, flag_get_eval = evaluate_policy(
                        eval_env_with_monitor, self.model, episodes=1, render=False
                    )
                    eval_env_with_monitor.close()
                else:
                    eval_env_no_monitor = wrap_mario_environment(environment, COMPLEX_MOVEMENT)
                    avg_score, info, eval_max_stage, flag_get_eval = evaluate_policy(
                        eval_env_no_monitor, self.model, episodes=1, render=False
                    )
                    eval_env_no_monitor.close()
                    print(f"[PPO Training] No new training high score or stage clear. Skipping video save.", flush=True)
                
                print(f"[PPO Training] [Eval] Update {self.update}: avg return = {avg_score:.2f} info: {info}", flush=True)
                
                if flag_get_eval:
                    if self.weights_dir:
                        model_path = experiment_manager.save_model(
                            self.experiment_dir, f"{environment}_clear", 
                            self.model.state_dict()
                        )
                        print(f"[PPO Training] Model saved to {model_path} after completing the stage!", flush=True)
                        break
            
            # 定期保存模型
            if self.update % self.save_frequency == 0:
                if self.weights_dir:
                    model_path = experiment_manager.save_model(
                        self.experiment_dir, f"{environment}_ppo_checkpoint_{self.update}", 
                        self.model.state_dict()
                    )
                    print(f"[PPO Training] Model saved to: {model_path}", flush=True)
                elif save_path:
                    # 兼容旧的方式
                    model_path = os.path.join(save_path, f"ppo_model_update_{self.update}.pth")
                    torch.save(self.model.state_dict(), model_path)
                    print(f"[PPO Training] Model saved to: {model_path}", flush=True)
        
        print("[PPO Training] Training completed!", flush=True)
        print(f"[PPO Training] Training metrics saved to: {self.csv_filename}", flush=True)
        
        # 保存最终模型
        if self.weights_dir:
            final_model_path = experiment_manager.save_model(
                self.experiment_dir, f"{environment}_ppo_final", 
                self.model.state_dict()
            )
            print(f"[PPO Training] Final model saved to: {final_model_path}", flush=True)
            
            # 更新实验状态
            experiment_manager.update_metadata(self.experiment_dir, {
                "status": "completed",
                "final_update": self.update,
                "best_reward": self.best_reward
            })
        elif save_path:
            # 兼容旧的方式
            final_model_path = os.path.join(save_path, f"{environment}_ppo_final.pth")
            torch.save(self.model.state_dict(), final_model_path)
            print(f"[PPO Training] Final model saved to: {final_model_path}", flush=True)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO训练脚本')
    
    # 环境参数
    parser.add_argument('--environment', type=str, default='SuperMarioBros-1-1-v0',
                       help='环境名称')
    parser.add_argument('--action-space', type=str, default='complex',
                       choices=['simple', 'complex', 'right_only'],
                       help='动作空间')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练episode数')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--lambda-gae', type=float, default=0.95,
                       help='GAE lambda参数')
    parser.add_argument('--clip-eps', type=float, default=0.2,
                       help='PPO裁剪参数')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='价值函数损失系数')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='熵系数')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='最大梯度范数')
    parser.add_argument('--epochs', type=int, default=4,
                       help='PPO更新轮数')
    parser.add_argument('--minibatch-size', type=int, default=64,
                       help='小批次大小')
    parser.add_argument('--num-env', type=int, default=8,
                       help='并行环境数量')
    parser.add_argument('--rollout-steps', type=int, default=128,
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







