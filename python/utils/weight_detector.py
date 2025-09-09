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
        weights = torch.load(weights_path, map_location='cpu')
        
        # 假设所有模型都是CNNDQN架构，只检测动作空间
        if isinstance(weights, dict) and 'fc.2.weight' in weights:
            # 检测输出层大小来确定动作空间
            output_size = weights['fc.2.weight'].shape[0]
            if output_size == 7:
                action_space = 'SIMPLE'
            elif output_size == 12:
                action_space = 'COMPLEX'
            else:
                action_space = 'UNKNOWN'
        else:
            action_space = 'UNKNOWN'
        
        return {
            'model_type': 'CNNDQN',
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
