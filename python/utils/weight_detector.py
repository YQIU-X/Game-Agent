"""
权重文件检测器
用于检测权重文件的类型、动作空间等信息
"""

import torch
import pickle
import os
from typing import Dict, Any, Optional


class WeightDetector:
    """权重文件检测器"""
    
    @staticmethod
    def detect_model_info(weights_path: str) -> Dict[str, Any]:
        """
        检测权重文件信息
        
        Args:
            weights_path: 权重文件路径
            
        Returns:
            包含模型信息的字典
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        try:
            # 加载权重文件
            if weights_path.endswith('.pth') or weights_path.endswith('.pt') or weights_path.endswith('.dat'):
                return WeightDetector._detect_pytorch_weights(weights_path)
            elif weights_path.endswith('.pkl') or weights_path.endswith('.pickle'):
                return WeightDetector._detect_pickle_weights(weights_path)
            else:
                raise ValueError(f"不支持的权重文件格式: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"检测权重文件失败: {e}")
    
    @staticmethod
    def _detect_pytorch_weights(weights_path: str) -> Dict[str, Any]:
        """检测PyTorch权重文件"""
        # 加载权重
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # 检测模型类型和动作空间
        model_type = 'UNKNOWN'
        action_space = 'UNKNOWN'
        
        if isinstance(weights, dict):
            # 检测DQN模型 (有fc.2.weight层)
            if 'fc.2.weight' in weights:
                model_type = 'CNNDQN'
                output_size = weights['fc.2.weight'].shape[0]
                if output_size == 7:
                    action_space = 'SIMPLE'
                elif output_size == 12:
                    action_space = 'COMPLEX'
                else:
                    action_space = 'UNKNOWN'
            
            # 检测PPO模型 (有policy_head.weight和value_head.weight层)
            elif 'policy_head.weight' in weights and 'value_head.weight' in weights:
                model_type = 'CNNPPO'
                output_size = weights['policy_head.weight'].shape[0]
                if output_size == 7:
                    action_space = 'SIMPLE'
                elif output_size == 12:
                    action_space = 'COMPLEX'
                else:
                    action_space = 'UNKNOWN'
            
            # 检测A2C模型 (有actor_head.weight和critic_head.weight层)
            elif 'actor_head.weight' in weights and 'critic_head.weight' in weights:
                model_type = 'CNNA2C'
                output_size = weights['actor_head.weight'].shape[0]
                if output_size == 7:
                    action_space = 'SIMPLE'
                elif output_size == 12:
                    action_space = 'COMPLEX'
                else:
                    action_space = 'UNKNOWN'
            
            # 通过文件名推断模型类型
            else:
                filename = os.path.basename(weights_path).lower()
                if 'ppo' in filename or 'actor_critic' in filename:
                    model_type = 'CNNPPO'
                    # 尝试从文件名推断动作空间
                    if 'simple' in filename:
                        action_space = 'SIMPLE'
                    elif 'complex' in filename:
                        action_space = 'COMPLEX'
                    else:
                        action_space = 'COMPLEX'  # 默认使用COMPLEX
                elif 'a2c' in filename:
                    model_type = 'CNNA2C'
                    if 'simple' in filename:
                        action_space = 'SIMPLE'
                    elif 'complex' in filename:
                        action_space = 'COMPLEX'
                    else:
                        action_space = 'COMPLEX'
                else:
                    # 默认为DQN
                    model_type = 'CNNDQN'
                    action_space = 'COMPLEX'
        
        return {
            'model_type': model_type,
            'action_space': action_space,
            'input_shape': (4, 84, 84),  # 标准输入形状
            'weights_path': weights_path
        }
    
    @staticmethod
    def _detect_pickle_weights(weights_path: str) -> Dict[str, Any]:
        """检测Pickle权重文件"""
        with open(weights_path, 'rb') as f:
            data = pickle.load(f)
        
        # 这里可以根据pickle文件的结构进行检测
        # 暂时返回默认值
        return {
            'model_type': 'UNKNOWN',
            'action_space': 'UNKNOWN',
            'input_shape': None,
            'weights_path': weights_path
        }
