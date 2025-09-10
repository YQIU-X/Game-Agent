#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pyglet兼容性补丁 - 修复Python 3.8+中time.clock()被移除的问题
"""

import sys
import time
import os

# 设置环境变量
os.environ['PYGLET_HEADLESS'] = '1'

def patch_pyglet():
    """应用pyglet兼容性补丁"""
    try:
        # 修复time.clock问题
        if not hasattr(time, 'clock'):
            time.clock = time.perf_counter
        
        # 设置pyglet为无头模式
        import pyglet
        pyglet.options['headless'] = True
        
        # 设置更多环境变量确保无头模式
        os.environ['PYGLET_HEADLESS'] = '1'
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['DISPLAY'] = ''
        
        print("[Pyglet补丁] 已应用pyglet兼容性补丁和无头模式设置", flush=True)
        return True
    except Exception as e:
        print(f"[Pyglet补丁] 应用补丁失败: {e}", flush=True)
        return False

# 自动应用补丁
if __name__ == '__main__':
    patch_pyglet()
else:
    # 当作为模块导入时自动应用补丁
    patch_pyglet()
