"""
模型工厂：根据权重文件自动创建正确的模型
"""

import os
import torch
import pickle
from typing import Dict, Any, Optional, Type
from models import CNNDQN, CNNPPO, CNNA2C
from models.base import BaseModel


class ModelFactory:
    """模型工厂类"""
    
    # 模型类型映射
    MODEL_CLASSES = {
        'CNNDQN': CNNDQN,
        'CNNPPO': CNNPPO,
        'CNNA2C': CNNA2C,
    }
    
    @staticmethod
    def detect_model_info(weight_path: str) -> Dict[str, Any]:
        """
        检测权重文件信息
        
        Args:
            weight_path: 权重文件路径
            
        Returns:
            包含模型信息的字典
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        
        # 尝试加载权重文件
        try:
            state = torch.load(weight_path, map_location='cpu')
        except:
            try:
                with open(weight_path, 'rb') as f:
                    state = pickle.load(f)
            except Exception as e:
                raise ValueError(f"无法加载权重文件: {e}")
        
        # 检测模型类型
        model_type = ModelFactory._detect_model_type(state)
        
        # 检测动作空间
        action_space_info = ModelFactory._detect_action_space(state)
        
        # 检测输入形状
        input_shape = ModelFactory._detect_input_shape(state)
        
        return {
            'model_type': model_type,
            'action_space': action_space_info,
            'input_shape': input_shape,
            'state_dict': state if isinstance(state, dict) else {'state_dict': state},
            'weight_path': weight_path
        }
    
    @staticmethod
    def create_model(model_info: Dict[str, Any]) -> BaseModel:
        """
        根据模型信息创建模型
        
        Args:
            model_info: 模型信息字典
            
        Returns:
            创建的模型实例
        """
        model_type = model_info['model_type']
        input_shape = model_info['input_shape']
        num_actions = model_info['action_space']['num_actions']
        
        if model_type not in ModelFactory.MODEL_CLASSES:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_class = ModelFactory.MODEL_CLASSES[model_type]
        model = model_class(input_shape, num_actions)
        
        # 加载权重
        state_dict = model_info['state_dict']
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        return model
    
    @staticmethod
    def _detect_model_type(state: Any) -> str:
        """检测模型类型"""
        if isinstance(state, dict):
            # 检查是否有明确的模型类型标识
            if 'model_type' in state:
                return state['model_type']
            
            # 通过层结构推断
            state_dict = state.get('state_dict', state)
            if isinstance(state_dict, dict):
                # 检查是否有actor和critic网络（PPO/A2C特征）
                if any('actor' in key for key in state_dict.keys()):
                    if any('critic' in key for key in state_dict.keys()):
                        return 'CNNPPO'  # 或者CNNA2C，需要进一步区分
                
                # 检查是否有Q值输出层（DQN特征）
                if any('fc' in key and 'weight' in key for key in state_dict.keys()):
                    return 'CNNDQN'
        
        # 默认返回CNNDQN
        return 'CNNDQN'
    
    @staticmethod
    def _detect_action_space(state: Any) -> Dict[str, Any]:
        """检测动作空间"""
        if isinstance(state, dict):
            # 检查是否有保存的动作空间信息
            if 'action_space' in state:
                action_space = state['action_space']
                if action_space == 'SIMPLE':
                    return {
                        'type': 'SIMPLE',
                        'num_actions': 7
                    }
                elif action_space == 'COMPLEX':
                    return {
                        'type': 'COMPLEX',
                        'num_actions': 12
                    }
            
            # 通过模型输出层大小推断
            state_dict = state.get('state_dict', state)
            if isinstance(state_dict, dict):
                # 查找最后的线性层
                for key in state_dict.keys():
                    if 'fc.2.weight' in key and len(state_dict[key].shape) == 2:
                        num_actions = state_dict[key].shape[0]
                        if num_actions == 12:
                            return {
                                'type': 'COMPLEX',
                                'num_actions': num_actions
                            }
                        else:
                            return {
                                'type': 'SIMPLE',
                                'num_actions': num_actions
                            }
        
        # 默认返回SIMPLE
        return {
            'type': 'SIMPLE',
            'num_actions': 7
        }
    
    @staticmethod
    def _detect_input_shape(state: Any) -> tuple:
        """检测输入形状"""
        if isinstance(state, dict):
            # 检查是否有保存的输入形状信息
            if 'input_shape' in state:
                return state['input_shape']
            
            # 通过卷积层推断
            state_dict = state.get('state_dict', state)
            if isinstance(state_dict, dict):
                for key in state_dict.keys():
                    if 'conv' in key and 'weight' in key and len(state_dict[key].shape) == 4:
                        in_channels = state_dict[key].shape[1]
                        return (in_channels, 84, 84)
        
        # 默认返回(4, 84, 84)
        return (4, 84, 84)
