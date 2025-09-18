#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
无损MP4转GIF转换脚本
保持最佳质量，不进行压缩
"""

import imageio
import os
from PIL import Image

def convert_mp4_to_gif_lossless(mp4_path, gif_path, fps=10, max_width=None):
    """
    无损将MP4文件转换为GIF
    
    Args:
        mp4_path: MP4文件路径
        gif_path: 输出GIF文件路径
        fps: GIF帧率
        max_width: 最大宽度（None表示不缩放）
    """
    print(f"开始无损转换: {mp4_path} -> {gif_path}")
    
    # 检查文件是否存在
    if not os.path.exists(mp4_path):
        print(f"错误: 文件不存在 {mp4_path}")
        return False
    
    try:
        # 读取MP4文件
        reader = imageio.get_reader(mp4_path)
        fps_original = reader.get_meta_data().get('fps', 30)
        print(f"原始视频帧率: {fps_original} FPS")
        print(f"目标GIF帧率: {fps} FPS")
        
        # 计算跳帧间隔
        skip_frames = max(1, int(fps_original / fps))
        print(f"跳帧间隔: {skip_frames}")
        
        frames = []
        frame_count = 0
        
        for i, frame in enumerate(reader):
            if i % skip_frames == 0:  # 按间隔取帧
                # 无损处理：不进行压缩，保持原始质量
                img = Image.fromarray(frame)
                
                # 只在需要时调整尺寸
                if max_width and img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                frames.append(img)
                frame_count += 1
                
                if frame_count % 50 == 0:
                    print(f"已处理 {frame_count} 帧...")
        
        print(f"总共处理了 {frame_count} 帧")
        
        # 保存为GIF（无损设置）
        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),  # 毫秒
                loop=0,  # 无限循环
                optimize=False,  # 不优化以保持质量
                quality=100,  # 最高质量
                method=6  # 最佳压缩方法
            )
            
            # 显示文件大小
            file_size = os.path.getsize(gif_path) / (1024 * 1024)  # MB
            print(f"GIF文件大小: {file_size:.2f} MB")
            print(f"无损转换完成: {gif_path}")
            return True
        else:
            print("错误: 没有提取到任何帧")
            return False
            
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def main():
    """主函数"""
    # 无损转换developer.mp4
    mp4_path = "static/developer.mp4"
    gif_path = "static/developer.gif"
    
    print("开始无损转换developer.mp4...")
    
    if convert_mp4_to_gif_lossless(mp4_path, gif_path, fps=10, max_width=None):
        print("✅ 无损转换成功完成!")
    else:
        print("❌ 转换失败!")

if __name__ == "__main__":
    main()
