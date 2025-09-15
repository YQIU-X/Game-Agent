#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO辅助函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter


def compute_gae_batch(rewards, values, dones, gamma=0.99, lam=0.95):
    """计算批量GAE（广义优势估计）"""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    
    returns = advantages + values[:-1]
    return advantages, returns


def rollout_with_bootstrap(envs, model, rollout_steps, init_obs, get_reward_func):
    """
    执行策略rollout指定步数，收集经验并使用GAE计算优势
    """
    obs = init_obs
    obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
    flag_get_in_rollout = False
    episode_infos = []
    episode_rewards = []  # 用于存储每个回合的总奖励
    current_rewards = np.zeros(envs.num_envs)  # 用于跟踪每个环境的当前奖励
    
    for _ in range(rollout_steps):
        obs_buf.append(obs)
        
        with torch.no_grad():
            action, logp, value = model.act(torch.tensor(obs, dtype=torch.float32).to(model.device))
        
        val_buf.append(value)
        logp_buf.append(logp)
        act_buf.append(action)
        
        actions = action.cpu().numpy()
        next_obs, reward, done, infos = envs.step(actions)
        
        # 累加每个环境的奖励
        current_rewards += reward
        
        # 将原始奖励传入，用于计算优势
        reward = [get_reward_func(r) for r in reward]
        
        rew_buf.append(torch.tensor(reward, dtype=torch.float32).to(model.device))
        done_buf.append(torch.tensor(done, dtype=torch.float32).to(model.device))
        
        for i, d in enumerate(done):
            if d:
                info = infos[i]
                if info.get('flag_get', False):
                    flag_get_in_rollout = True
                
                # 检查gym.vector.SyncVectorEnv在done时添加的episode键
                if 'episode' in info:
                    episode_infos.append(info['episode'])
                
                # 如果回合结束，记录并重置总奖励
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0
        
        obs = next_obs
    
    with torch.no_grad():
        _, last_value = model.forward(torch.tensor(obs, dtype=torch.float32).to(model.device))
    
    obs_buf = np.array(obs_buf)
    act_buf = torch.stack(act_buf)
    rew_buf = torch.stack(rew_buf)
    done_buf = torch.stack(done_buf)
    val_buf = torch.stack(val_buf)
    val_buf = torch.cat([val_buf, last_value.unsqueeze(0)], dim=0)
    logp_buf = torch.stack(logp_buf)
    
    # 重塑obs_buf以便后续张量转换
    obs_buf = obs_buf.reshape(rollout_steps, envs.num_envs, *envs.single_observation_space.shape)
    obs_buf = torch.tensor(obs_buf, dtype=torch.float32).to(model.device)
    
    adv_buf, ret_buf = compute_gae_batch(rew_buf, val_buf, done_buf)
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
    
    # 从episode信息中提取最大关卡
    max_stage = max([i.get('stage', 1) for i in episode_infos]) if episode_infos else 1
    
    # 计算并返回平均原始总奖励
    avg_total_reward = np.mean(episode_rewards) if episode_rewards else 0
    
    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logprobs": logp_buf,
        "advantages": adv_buf,
        "returns": ret_buf,
        "max_stage": max_stage,
        "last_obs": obs,
        "flag_get": flag_get_in_rollout,
        "avg_total_reward": avg_total_reward
    }


def evaluate_policy(env, model, episodes=5, render=False):
    """评估学习到的策略"""
    model.eval()
    total_returns = []
    actions = []
    stages = []
    flag_get_eval = False
    
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = (
                torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(model.device)
            )
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()
                actions.append(action)
            
            obs, reward, done, info = env.step(action)
            stages.append(info["stage"])
            total_reward += reward
            if info.get('flag_get', False):
                flag_get_eval = True
        
        total_returns.append(total_reward)
    
    model.train()
    info = {"action_count": Counter(actions)}
    return np.mean(total_returns), info, max(stages), flag_get_eval


def get_reward(r):
    """自定义奖励塑形函数以提高训练稳定性"""
    r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
    return r


