#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN工具函数
包含损失计算、参数更新等辅助函数
"""

import math
import numpy as np
import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable


class Range:
    """参数范围验证类"""
    
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end


def compute_td_loss(model, target_net, replay_buffer, gamma, device, batch_size, beta=0.4):
    """
    计算TD损失
    Args:
        model: 主网络
        target_net: 目标网络
        replay_buffer: 经验回放缓冲区
        gamma: 折扣因子
        device: 计算设备
        batch_size: 批次大小
        beta: 重要性采样指数
    Returns:
        loss: TD损失
    """
    if isinstance(replay_buffer, PrioritizedBuffer):
        # 优先经验回放
        batch = replay_buffer.sample(batch_size, beta)
        state, action, reward, next_state, done, indices, weights = batch
        
        weights = Variable(FloatTensor(weights)).to(device)
    else:
        # 标准经验回放
        batch = replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = batch
        weights = None
        indices = None

    state = Variable(FloatTensor(np.float32(state))).to(device)
    next_state = Variable(FloatTensor(np.float32(next_state))).to(device)
    action = Variable(LongTensor(action)).to(device)
    reward = Variable(FloatTensor(reward)).to(device)
    done = Variable(FloatTensor(done)).to(device)

    # 计算Q值
    q_values = model(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # 计算损失
    loss = (q_value - expected_q_value.detach()).pow(2)
    
    if weights is not None:
        loss = loss * weights
        prios = loss + 1e-5
        loss = loss.mean()
        # 更新优先级
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    else:
        loss = loss.mean()
    
    return loss


def update_epsilon(episode, epsilon_start, epsilon_final, epsilon_decay):
    """
    更新探索率
    Args:
        episode: 当前episode
        epsilon_start: 初始探索率
        epsilon_final: 最终探索率
        epsilon_decay: 衰减参数
    Returns:
        epsilon: 新的探索率
    """
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
        math.exp(-1 * ((episode + 1) / epsilon_decay))
    return epsilon


def update_beta(episode, beta_start, beta_frames):
    """
    更新重要性采样指数
    Args:
        episode: 当前episode
        beta_start: 初始beta值
        beta_frames: beta更新帧数
    Returns:
        beta: 新的beta值
    """
    beta = beta_start + episode * (1.0 - beta_start) / beta_frames
    return min(1.0, beta)


def set_device(force_cpu=False):
    """
    设置计算设备
    Args:
        force_cpu: 是否强制使用CPU
    Returns:
        device: 计算设备
    """
    device = torch.device('cpu')
    if not force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    return device


def load_model(model_path, model, target_model=None):
    """
    加载预训练模型
    Args:
        model_path: 模型路径
        model: 主网络
        target_model: 目标网络（可选）
    Returns:
        model: 加载后的主网络
        target_model: 加载后的目标网络（如果提供）
    """
    model.load_state_dict(torch.load(model_path))
    if target_model is not None:
        target_model.load_state_dict(model.state_dict())
        return model, target_model
    return model


def save_model(model, model_path):
    """
    保存模型
    Args:
        model: 要保存的模型
        model_path: 保存路径
    """
    torch.save(model.state_dict(), model_path)


def initialize_models(input_shape, num_actions, device, model_type='cnn'):
    """
    初始化DQN模型
    Args:
        input_shape: 输入形状
        num_actions: 动作数量
        device: 计算设备
        model_type: 模型类型（'cnn' 或 'mlp'）
    Returns:
        model: 主网络
        target_model: 目标网络
    """
    if model_type == 'cnn':
        from .model import CNNDQN
        model = CNNDQN(input_shape, num_actions).to(device)
        target_model = CNNDQN(input_shape, num_actions).to(device)
    else:
        from .model import DQN
        model = DQN(input_shape[0], num_actions).to(device)
        target_model = DQN(input_shape[0], num_actions).to(device)
    
    return model, target_model
