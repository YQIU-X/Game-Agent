#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO训练脚本 - 支持实时日志输出和指标记录
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# 导入项目模块
from python.models.cnnppo import CNNPPO
from python.wrappers.mario_wrappers import wrap_environment

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        
        # 训练状态
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # 指标记录
        self.episode_rewards = []
        self.value_losses = []
        self.policy_losses = []
        self.total_losses = []
        
        # 初始化环境和模型
        self._setup_environment()
        self._setup_model()
        
    def _setup_environment(self):
        """设置训练环境"""
        env_name = self.config.get('env_name', 'SuperMarioBros-1-1-v0')
        action_space = self.config.get('action_space', 'SIMPLE')
        
        # 创建环境
        self.env = gym_super_mario_bros.make(env_name)
        
        # 设置动作空间
        if action_space == 'SIMPLE':
            self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
            self.action_list = SIMPLE_MOVEMENT
        else:
            self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
            self.action_list = COMPLEX_MOVEMENT
            
        # 应用包装器
        self.env = wrap_environment(self.env, model_type='CNNPPO')
        
        # 获取环境信息
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        self.obs_shape = obs.shape
        self.num_actions = len(self.action_list)
        
        self.log(f"环境设置完成: {env_name}, 动作空间: {action_space}, 观察形状: {self.obs_shape}, 动作数量: {self.num_actions}")
        
    def _setup_model(self):
        """设置PPO模型"""
        # 创建模型
        self.model = CNNPPO(self.obs_shape, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.log(f"模型设置完成: CNNPPO, 设备: {self.device}")
        
    def log(self, message):
        """输出日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message, file=sys.stderr)
        
        # 发送到stdout供前端获取
        try:
            print(json.dumps({
                'timestamp': timestamp,
                'message': message,
                'type': 'log'
            }), flush=True)
        except:
            pass
            
    def log_metrics(self, metrics):
        """输出训练指标"""
        try:
            print(json.dumps({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'metrics': metrics,
                'type': 'metrics'
            }), flush=True)
        except:
            pass
            
    def collect_episode(self):
        """收集一个episode的数据"""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 获取动作
            with torch.no_grad():
                action_probs = self.model(obs)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.model.get_value(obs)
            
            # 执行动作
            step_result = self.env.step(action.item())
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
            elif len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
                
            # 更新状态
            episode_reward += reward
            episode_length += 1
            obs = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            
            # 检查是否获得旗帜
            if info.get('flag_get', False):
                self.log(f"🎉 Episode {self.episode} 获得旗帜！")
                
        return episode_reward, episode_length
        
    def train_step(self):
        """执行一步训练（简化版本）"""
        # 这里简化训练逻辑，实际应该收集更多数据
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # 前向传播
        action_probs = self.model(obs)
        dist = Categorical(action_probs)
        values = self.model.get_value(obs)
        
        # 计算损失（简化）
        policy_loss = -dist.entropy().mean()
        value_loss = nn.MSELoss()(values, torch.zeros_like(values))
        total_loss = policy_loss + self.value_loss_coef * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 记录损失
        self.value_losses.append(value_loss.item())
        self.policy_losses.append(policy_loss.item())
        self.total_losses.append(total_loss.item())
        
    def save_model(self, filename):
        """保存模型"""
        torch.save(self.model.state_dict(), filename)
        self.log(f"模型已保存: {filename}")
        
    def train(self, max_episodes=1000, save_interval=100):
        """主训练循环"""
        self.log(f"开始训练，最大episodes: {max_episodes}")
        self.log(f"学习率: {self.learning_rate}, Gamma: {self.gamma}")
        
        start_time = time.time()
        
        for episode in range(max_episodes):
            self.episode = episode
            
            # 收集episode数据
            episode_reward, episode_length = self.collect_episode()
            
            # 记录指标
            self.episode_rewards.append(episode_reward)
            self.total_steps += episode_length
            
            # 输出episode信息
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            self.log(f"Episode {episode}: 奖励={episode_reward:.2f}, 长度={episode_length}, 平均奖励={avg_reward:.2f}")
            
            # 发送指标到前端
            metrics = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'avg_reward': avg_reward,
                'total_steps': self.total_steps,
                'value_loss': np.mean(self.value_losses[-10:]) if self.value_losses else 0,
                'policy_loss': np.mean(self.policy_losses[-10:]) if self.policy_losses else 0,
                'total_loss': np.mean(self.total_losses[-10:]) if self.total_losses else 0
            }
            self.log_metrics(metrics)
            
            # 保存最佳模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model(f"best_model_episode_{episode}.pth")
                
            # 定期保存
            if (episode + 1) % save_interval == 0:
                self.save_model(f"checkpoint_episode_{episode}.pth")
                
            # 检查是否应该停止训练
            if episode >= 100 and avg_reward < -100:
                self.log("训练效果不佳，提前停止")
                break
                
        # 训练完成
        training_time = time.time() - start_time
        self.log(f"训练完成！总时间: {training_time/3600:.2f}小时")
        self.log(f"最佳奖励: {self.best_reward:.2f}")
        self.log(f"最终平均奖励: {np.mean(self.episode_rewards[-100:]):.2f}")
        
        # 保存最终模型
        self.save_model("final_model.pth")
        
        return {
            'episode_rewards': self.episode_rewards,
            'value_losses': self.value_losses,
            'policy_losses': self.policy_losses,
            'total_losses': self.total_losses,
            'best_reward': self.best_reward,
            'total_steps': self.total_steps
        }

def main():
    parser = argparse.ArgumentParser(description='PPO训练脚本')
    parser.add_argument('--env', default='SuperMarioBros-1-1-v0', help='环境名称')
    parser.add_argument('--action-space', default='SIMPLE', choices=['SIMPLE', 'COMPLEX'], help='动作空间')
    parser.add_argument('--episodes', type=int, default=1000, help='最大episodes')
    parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--config', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'env_name': args.env,
            'action_space': args.action_space,
            'learning_rate': args.lr,
            'gamma': args.gamma,
            'clip_ratio': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4
        }
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 开始训练
    try:
        results = trainer.train(max_episodes=args.episodes)
        
        # 输出最终结果
        print(json.dumps({
            'type': 'training_complete',
            'results': results
        }), flush=True)
        
    except KeyboardInterrupt:
        trainer.log("训练被用户中断")
    except Exception as e:
        trainer.log(f"训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
