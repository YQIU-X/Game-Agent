#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
渲染服务
提供headless渲染功能，将游戏画面保存为图像文件
"""

import os
import cv2
import numpy as np
from PIL import Image
import io
import base64


class RenderService:
    """渲染服务类"""
    
    def __init__(self, output_dir="renders"):
        """
        初始化渲染服务
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.frame_count = 0
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def render_frame(self, env, episode, step, save_image=True):
        """
        渲染一帧
        Args:
            env: 游戏环境
            episode: episode编号
            step: 步数
            save_image: 是否保存图像
        Returns:
            frame_data: 帧数据（base64编码）
        """
        try:
            # 获取当前帧
            frame = env.render(mode='rgb_array')
            
            if frame is None:
                return None
            
            # 转换颜色格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if save_image:
                # 保存图像文件
                filename = f"episode_{episode}_step_{step:06d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame)
            
            # 转换为base64
            frame_data = self.frame_to_base64(frame)
            
            return frame_data
            
        except Exception as e:
            print(f"[渲染服务] 渲染失败: {e}")
            return None
    
    def frame_to_base64(self, frame):
        """
        将帧转换为base64编码
        Args:
            frame: 图像帧
        Returns:
            base64编码的字符串
        """
        try:
            # 编码为JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # 转换为base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return frame_base64
            
        except Exception as e:
            print(f"[渲染服务] 转换base64失败: {e}")
            return None
    
    def get_latest_frame(self):
        """获取最新的渲染帧"""
        try:
            # 获取最新的图像文件
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
            if not files:
                return None
            
            # 按修改时间排序
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)), reverse=True)
            latest_file = files[0]
            
            # 读取图像
            filepath = os.path.join(self.output_dir, latest_file)
            frame = cv2.imread(filepath)
            
            if frame is None:
                return None
            
            # 转换为base64
            return self.frame_to_base64(frame)
            
        except Exception as e:
            print(f"[渲染服务] 获取最新帧失败: {e}")
            return None
    
    def cleanup_old_frames(self, keep_last=100):
        """
        清理旧的渲染帧
        Args:
            keep_last: 保留最新的帧数
        """
        try:
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
            if len(files) <= keep_last:
                return
            
            # 按修改时间排序
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
            
            # 删除旧文件
            files_to_delete = files[:-keep_last]
            for file in files_to_delete:
                filepath = os.path.join(self.output_dir, file)
                os.remove(filepath)
                
        except Exception as e:
            print(f"[渲染服务] 清理旧帧失败: {e}")


# 全局渲染服务实例
render_service = RenderService()


def get_render_service():
    """获取渲染服务实例"""
    return render_service
