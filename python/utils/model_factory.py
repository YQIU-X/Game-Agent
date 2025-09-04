"""
模型工厂
用于动态创建和检测模型
"""

import torch
import os
from typing import Dict, Any
from ..models import CNNDQN, CNNPPO, CNNA2C
from .weight_detector import WeightDetector


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def detect_model_info(weights_path: str) -> Dict[str, Any]:
        """
        检测权重文件信息
        
        Args:
            weights_path: 权重文件路径
            
        Returns:
            包含模型信息的字典
        """
        return WeightDetector.detect_model_info(weights_path)
    
    @staticmethod
    def create_model(model_info: Dict[str, Any]):
        """
        根据模型信息创建模型实例
        
        Args:
            model_info: 模型信息字典
            
        Returns:
            模型实例
        """
        model_type = model_info['model_type']
        input_shape = model_info['input_shape']
        
        if isinstance(model_info['action_space'], dict):
            num_actions = model_info['action_space']['num_actions']
        else:
            # 兼容旧格式
            if model_info['action_space'] == 'SIMPLE':
                num_actions = 7
            elif model_info['action_space'] == 'COMPLEX':
                num_actions = 12
            else:
                num_actions = 7  # 默认值
        
        if model_type == 'CNNDQN':
            model = CNNDQN(input_shape, num_actions)
        elif model_type == 'CNNPPO':
            model = CNNPPO(input_shape, num_actions)
        elif model_type == 'CNNA2C':
            model = CNNA2C(input_shape, num_actions)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        if 'weights_path' in model_info:
            model.load_weights(model_info['weights_path'])
        
        return model
    
    @staticmethod
    def create_model_from_weights(weights_path: str):
        """
        从权重文件直接创建模型
        
        Args:
            weights_path: 权重文件路径
            
        Returns:
            模型实例
        """
        model_info = ModelFactory.detect_model_info(weights_path)
        return ModelFactory.create_model(model_info)
