#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN训练器
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
from torch.optim import Adam

# 导入DQN相关模块
from python.algorithms.dqn.core import (
    CNNDQN, DQN, PrioritizedBuffer, ReplayBuffer,
    compute_td_loss, update_epsilon, update_beta,
    set_device, initialize_models, save_model
)

# 导入Mario相关模块
from python.games.mario.core import wrap_mario_environment, ACTION_SPACES
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# 导入实验管理器
from python.utils.experiment_manager import experiment_manager


class DQNTrainer:
    """DQN训练器类"""
    
    def __init__(self, config):
        """
        初始化DQN训练器
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = set_device(config.get('force_cpu', False))
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_final = config.get('epsilon_final', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 100000)
        self.target_update_frequency = config.get('target_update_frequency', 1000)
        self.batch_size = config.get('batch_size', 32)
        self.memory_capacity = config.get('memory_capacity', 20000)
        self.initial_learning = config.get('initial_learning', 10000)
        self.beta_start = config.get('beta_start', 0.4)
        self.beta_frames = config.get('beta_frames', 10000)
        
        # 训练状态
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.episode_rewards = []
        
        # CSV记录相关
        self.training_metrics = []
        self.episode_losses = []
        self.csv_filename = None
        
        # 模型和优化器
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.replay_buffer = None
        
        # 环境
        self.env = None
        
        # 日志
        self.log_frequency = config.get('log_frequency', 10)
        self.save_frequency = config.get('save_frequency', 100)
        
        # 实验管理
        self.experiment_dir = None
        self.metrics_path = None
        self.logs_path = None
        self.weights_dir = None
        
    def setup_experiment(self, environment, algorithm='DQN'):
        """设置实验目录和文件路径"""
        # 创建实验目录
        self.experiment_dir = experiment_manager.create_experiment_dir(
            environment, algorithm, self.config
        )
        
        # 设置文件路径
        self.metrics_path = experiment_manager.get_metrics_path(self.experiment_dir)
        self.logs_path = experiment_manager.get_logs_path(self.experiment_dir)
        self.weights_dir = experiment_manager.get_weights_dir(self.experiment_dir)
        
        print(f"[DQN Training] 实验目录: {self.experiment_dir}", flush=True)
        
    def setup_csv_logging(self, environment):
        """设置CSV日志记录"""
        if not self.metrics_path:
            # 如果没有设置实验目录，使用旧的方式
            os.makedirs('training_metrics', exist_ok=True)
            self.csv_filename = f'training_metrics/{environment}.csv'
        else:
            # 使用新的实验目录
            self.csv_filename = self.metrics_path
        
        # 写入CSV头部
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'total_reward', 'best_reward', 'average_reward', 
                           'episode_length', 'total_steps', 'epsilon', 'average_loss', 
                           'learning_rate', 'gamma', 'timestamp'])
        
        print(f"[DQN Training] CSV log file created: {self.csv_filename}", flush=True)
    
    def save_metrics_to_csv(self, episode, episode_reward, episode_length, epsilon, avg_loss):
        """保存训练指标到CSV文件"""
        if self.csv_filename:
            # 计算平均奖励
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            
            # 准备数据
            metrics_data = [
                episode,
                round(episode_reward, 3),
                round(self.best_reward, 3),
                round(avg_reward, 3),
                episode_length,
                self.total_steps,
                round(epsilon, 4),
                round(avg_loss, 4) if not np.isnan(avg_loss) else 0.0,
                self.learning_rate,
                self.gamma,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            
            # 写入CSV
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(metrics_data)
    
    def shape_reward(self, reward):
        """
        奖励塑形函数，用于提升训练稳定性
        """
        return np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
        
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
            
        # 创建环境
        self.env = wrap_mario_environment(environment, action_space_obj)
        
        # 初始化模型
        input_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n
        
        self.model, self.target_model = initialize_models(
            input_shape, num_actions, self.device, 'cnn'
        )
        
        # 初始化优化器
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 初始化经验回放缓冲区
        self.replay_buffer = PrioritizedBuffer(self.memory_capacity)
        
    def update_models(self, beta):
        """更新模型"""
        if len(self.replay_buffer) > self.initial_learning:
            # 更新目标网络
            if self.total_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # 计算损失并更新
            self.optimizer.zero_grad()
            loss = compute_td_loss(
                self.model, self.target_model, self.replay_buffer,
                self.gamma, self.device, self.batch_size, beta
            )
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            self.episode_losses.append(loss.item())
    
    def run_episode(self, render=False):
        """
        运行一个episode
        Args:
            render: 是否渲染
        Returns:
            episode_reward: episode总奖励
            episode_length: episode长度
            flag_get: 是否通关
        """
        episode_reward = 0.0
        episode_length = 0
        flag_get = False
        state = self.env.reset()
        
        while True:
            # 更新探索率
            epsilon = update_epsilon(
                self.total_steps, self.epsilon_start, 
                self.epsilon_final, self.epsilon_decay
            )
            
            # 更新beta
            if len(self.replay_buffer) > self.batch_size:
                beta = update_beta(self.total_steps, self.beta_start, self.beta_frames)
            else:
                beta = self.beta_start
            
            # 选择动作
            action = self.model.act(state, epsilon, self.device)
            
            # 渲染 - 恢复原来的渲染方式
            if render:
                try:
                    self.env.render()
                except Exception as e:
                    # 如果渲染失败，记录错误但不中断训练
                    if not hasattr(self, '_render_error_logged'):
                        print(f"[DQN训练] 渲染失败，继续无头模式训练: {e}", flush=True)
                        self._render_error_logged = True
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 检查是否通关
            if info.get('flag_get', False):
                flag_get = True
            
            # 使用原始奖励记录episode总分
            episode_reward += reward
            
            # 使用塑形后的奖励进行训练
            shaped_reward = self.shape_reward(reward)
            self.replay_buffer.push(state, action, shaped_reward, next_state, done)
            
            # 更新状态
            state = next_state
            episode_length += 1
            self.total_steps += 1
            
            # 更新模型
            self.update_models(beta)
            
            if done:
                break
        
        return episode_reward, episode_length, flag_get
    
    def train(self, num_episodes, render=False, save_path=None):
        """
        开始训练
        Args:
            num_episodes: 训练episode数
            render: 是否渲染
            save_path: 模型保存路径（兼容旧接口）
        """
        print(f"[DQN Training] Starting training with {num_episodes} episodes", flush=True)
        print(f"[DQN Training] Device: {self.device}", flush=True)
        print(f"[DQN Training] Environment: {self.config.get('environment', 'Unknown')}", flush=True)
        print(f"[DQN Training] Action Space: {self.config.get('action_space', 'Unknown')}", flush=True)
        
        # 设置实验目录
        environment = self.config.get('environment', 'Unknown')
        algorithm = self.config.get('algorithm', 'DQN')
        self.setup_experiment(environment, algorithm)
        
        # 设置CSV日志记录
        self.setup_csv_logging(environment)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # 清空episode损失记录
            self.episode_losses.clear()
            
            # 运行episode
            episode_reward, episode_length, flag_get = self.run_episode(render)
            
            # 记录奖励
            self.episode_rewards.append(episode_reward)
            
            # 更新最佳奖励
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
            
            # 计算平均奖励
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            
            # 计算平均损失
            avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0.0
            
            # 计算当前epsilon
            current_epsilon = update_epsilon(
                self.total_steps, self.epsilon_start, 
                self.epsilon_final, self.epsilon_decay
            )
            
            # 保存指标到CSV
            self.save_metrics_to_csv(episode, episode_reward, episode_length, current_epsilon, avg_loss)
            
            # 日志输出 - 确保每次都输出到终端
            if episode % self.log_frequency == 0:
                elapsed_time = time.time() - start_time
                print(f"[DQN Training] Episode {episode} - Reward: {episode_reward:.2f}, "
                      f"Best: {self.best_reward:.2f}, Average: {avg_reward:.2f}, "
                      f"Length: {episode_length}, Steps: {self.total_steps}, "
                      f"Loss: {avg_loss:.4f}, Epsilon: {current_epsilon:.4f}, "
                      f"Time: {elapsed_time:.1f}s", flush=True)
            
            # 保存模型
            if episode % self.save_frequency == 0:
                if self.weights_dir:
                    # 使用新的实验目录
                    model_path = experiment_manager.save_model(
                        self.experiment_dir, f"checkpoint_episode_{episode}", 
                        self.model.state_dict()
                    )
                    print(f"[DQN Training] Model saved to: {model_path}", flush=True)
                elif save_path:
                    # 兼容旧的方式
                    model_path = os.path.join(save_path, f"dqn_model_episode_{episode}.pth")
                    save_model(self.model, model_path)
                    print(f"[DQN Training] Model saved to: {model_path}", flush=True)
            
            # 保存最佳模型
            if episode_reward > self.best_reward and self.weights_dir:
                best_model_path = experiment_manager.save_model(
                    self.experiment_dir, "best_model", 
                    self.model.state_dict()
                )
                print(f"[DQN Training] Best model saved to: {best_model_path}", flush=True)
            
            # 保存通关模型
            if flag_get and self.weights_dir:
                clear_model_path = experiment_manager.save_model(
                    self.experiment_dir, f"clear_stage_episode_{episode}", 
                    self.model.state_dict()
                )
                print(f"[DQN Training] Stage cleared! Model saved to: {clear_model_path}", flush=True)
                
                # 更新实验元数据，标记为通关
                experiment_manager.update_metadata(self.experiment_dir, {
                    "training_info": {
                        "final_update": "completed",
                        "best_reward": "stage_cleared",
                        "stage_cleared": True,
                        "clear_episode": episode
                    }
                })
                
                # 通关后可以选择继续训练或停止
                print(f"[DQN Training] Stage cleared at episode {episode}! Training completed.", flush=True)
                break
        
        print("[DQN Training] Training completed!", flush=True)
        print(f"[DQN Training] Training metrics saved to: {self.csv_filename}", flush=True)
        
        # 保存最终模型
        if self.weights_dir:
            final_model_path = experiment_manager.save_model(
                self.experiment_dir, "final_model", 
                self.model.state_dict()
            )
            print(f"[DQN Training] Final model saved to: {final_model_path}", flush=True)
            
            # 更新实验状态
            experiment_manager.update_metadata(self.experiment_dir, {
                "status": "completed",
                "final_episode": episode,
                "best_reward": self.best_reward,
                "total_steps": self.total_steps
            })
        elif save_path:
            # 兼容旧的方式
            final_model_path = os.path.join(save_path, "dqn_model_final.pth")
            save_model(self.model, final_model_path)
            print(f"[DQN Training] Final model saved to: {final_model_path}", flush=True)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DQN训练脚本')
    
    # 环境参数
    parser.add_argument('--environment', type=str, default='SuperMarioBros-1-1-v0',
                       help='环境名称')
    parser.add_argument('--action-space', type=str, default='complex',
                       choices=['simple', 'complex', 'right_only'],
                       help='动作空间')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练episode数')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='初始探索率')
    parser.add_argument('--epsilon-final', type=float, default=0.01,
                       help='最终探索率')
    parser.add_argument('--epsilon-decay', type=int, default=100000,
                       help='探索率衰减步数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--memory-capacity', type=int, default=20000,
                       help='经验回放缓冲区容量')
    parser.add_argument('--target-update-frequency', type=int, default=1000,
                       help='目标网络更新频率')
    parser.add_argument('--initial-learning', type=int, default=10000,
                       help='初始学习步数')
    parser.add_argument('--beta-start', type=float, default=0.4,
                       help='初始beta值')
    parser.add_argument('--beta-frames', type=int, default=10000,
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
        'algorithm': 'DQN',
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
    
    # 设置环境
    trainer.setup_environment(args.environment, args.action_space)
    
    # 开始训练
    save_path = args.save_path if args.save_model else None
    trainer.train(args.episodes, args.render, save_path)


if __name__ == '__main__':
    main()