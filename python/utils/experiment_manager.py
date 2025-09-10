#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实验管理器
负责管理训练实验的文件组织结构
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ExperimentManager:
    """实验管理器类"""
    
    def __init__(self, base_dir: str = "experiments"):
        """
        初始化实验管理器
        
        Args:
            base_dir: 实验根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def generate_session_id(self, config: Dict[str, Any]) -> str:
        """
        生成训练会话ID
        
        Args:
            config: 训练配置参数
            
        Returns:
            会话ID字符串
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 提取关键参数
        key_params = []
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if lr == 1e-4:
                key_params.append("lr1e4")
            elif lr == 1e-3:
                key_params.append("lr1e3")
            else:
                key_params.append(f"lr{lr}")
        
        if 'gamma' in config:
            gamma = config['gamma']
            if gamma == 0.99:
                key_params.append("gamma99")
            elif gamma == 0.95:
                key_params.append("gamma95")
            else:
                key_params.append(f"gamma{int(gamma*100)}")
        
        if 'epsilon_start' in config:
            eps = config['epsilon_start']
            if eps == 1.0:
                key_params.append("eps1")
            elif eps == 0.5:
                key_params.append("eps05")
            else:
                key_params.append(f"eps{eps}")
        
        # 组合会话ID
        if key_params:
            session_id = f"{timestamp}_{'_'.join(key_params)}"
        else:
            session_id = timestamp
        
        return session_id
    
    def create_experiment_dir(self, environment: str, algorithm: str, config: Dict[str, Any]) -> str:
        """
        创建实验目录结构
        
        Args:
            environment: 环境名称
            algorithm: 算法名称
            config: 训练配置参数
            
        Returns:
            实验目录路径
        """
        session_id = self.generate_session_id(config)
        
        # 创建目录结构
        exp_dir = self.base_dir / environment / algorithm / session_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建权重子目录
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        # 保存配置参数
        config_path = exp_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 创建元数据文件
        metadata = {
            "environment": environment,
            "algorithm": algorithm,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "config": config
        }
        
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[ExperimentManager] 创建实验目录: {exp_dir}")
        return str(exp_dir)
    
    def get_metrics_path(self, exp_dir: str) -> str:
        """获取指标文件路径"""
        return os.path.join(exp_dir, "metrics.csv")
    
    def get_logs_path(self, exp_dir: str) -> str:
        """获取日志文件路径"""
        return os.path.join(exp_dir, "logs.txt")
    
    def get_weights_dir(self, exp_dir: str) -> str:
        """获取权重目录路径"""
        return os.path.join(exp_dir, "weights")
    
    def save_model(self, exp_dir: str, model_name: str, model_state_dict: Dict) -> str:
        """
        保存模型权重
        
        Args:
            exp_dir: 实验目录
            model_name: 模型名称
            model_state_dict: 模型状态字典
            
        Returns:
            保存的模型文件路径
        """
        import torch
        
        weights_dir = self.get_weights_dir(exp_dir)
        model_path = os.path.join(weights_dir, f"{model_name}.pth")
        
        torch.save(model_state_dict, model_path)
        print(f"[ExperimentManager] 模型已保存: {model_path}")
        return model_path
    
    def update_metadata(self, exp_dir: str, updates: Dict[str, Any]):
        """
        更新元数据
        
        Args:
            exp_dir: 实验目录
            updates: 要更新的字段
        """
        metadata_path = os.path.join(exp_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata.update(updates)
        metadata["updated_at"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def list_experiments(self, environment: Optional[str] = None, algorithm: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出实验
        
        Args:
            environment: 环境名称过滤
            algorithm: 算法名称过滤
            
        Returns:
            实验列表
        """
        experiments = []
        
        # 确定搜索范围
        if environment and algorithm:
            search_dir = self.base_dir / environment / algorithm
        elif environment:
            search_dir = self.base_dir / environment
        else:
            search_dir = self.base_dir
        
        if not search_dir.exists():
            return experiments
        
        # 遍历目录
        for env_dir in search_dir.iterdir():
            if not env_dir.is_dir():
                continue
            
            env_name = env_dir.name
            if environment and env_name != environment:
                continue
            
            for alg_dir in env_dir.iterdir():
                if not alg_dir.is_dir():
                    continue
                
                alg_name = alg_dir.name
                if algorithm and alg_name != algorithm:
                    continue
                
                for session_dir in alg_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    
                    session_id = session_dir.name
                    
                    # 读取元数据
                    metadata_path = session_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        except:
                            metadata = {}
                    else:
                        metadata = {}
                    
                    # 检查文件存在性
                    metrics_path = session_dir / "metrics.csv"
                    weights_dir = session_dir / "weights"
                    
                    experiment_info = {
                        "environment": env_name,
                        "algorithm": alg_name,
                        "session_id": session_id,
                        "path": str(session_dir),
                        "has_metrics": metrics_path.exists(),
                        "has_weights": weights_dir.exists() and any(weights_dir.iterdir()),
                        "created_at": metadata.get("created_at", ""),
                        "status": metadata.get("status", "unknown"),
                        "config": metadata.get("config", {})
                    }
                    
                    experiments.append(experiment_info)
        
        # 按创建时间排序
        experiments.sort(key=lambda x: x["created_at"], reverse=True)
        return experiments
    
    def get_latest_experiment(self, environment: str, algorithm: str) -> Optional[Dict[str, Any]]:
        """
        获取最新的实验
        
        Args:
            environment: 环境名称
            algorithm: 算法名称
            
        Returns:
            最新实验信息
        """
        experiments = self.list_experiments(environment, algorithm)
        return experiments[0] if experiments else None
    
    def cleanup_old_experiments(self, keep_count: int = 10):
        """
        清理旧的实验，保留最新的几个
        
        Args:
            keep_count: 保留的实验数量
        """
        experiments = self.list_experiments()
        
        # 按环境+算法分组
        groups = {}
        for exp in experiments:
            key = f"{exp['environment']}_{exp['algorithm']}"
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)
        
        # 清理每个组
        for group_experiments in groups.values():
            if len(group_experiments) > keep_count:
                # 按创建时间排序，删除旧的
                group_experiments.sort(key=lambda x: x["created_at"], reverse=True)
                to_delete = group_experiments[keep_count:]
                
                for exp in to_delete:
                    try:
                        shutil.rmtree(exp["path"])
                        print(f"[ExperimentManager] 已删除旧实验: {exp['path']}")
                    except Exception as e:
                        print(f"[ExperimentManager] 删除实验失败: {exp['path']}, 错误: {e}")


# 全局实例
experiment_manager = ExperimentManager()
