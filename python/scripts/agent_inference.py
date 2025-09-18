#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用推理脚本：支持多种游戏和算法
"""

import argparse
import sys
import time
import json
import base64
import numpy as np
import torch
from PIL import Image
import io
import os

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from python.utils.model_factory import ModelFactory
from python.utils.environment_factory import EnvironmentFactory


def frame_to_base64(frame: np.ndarray) -> str:
    """将帧转换为base64编码"""
    # 确保帧是uint8格式
    if frame.dtype != np.uint8:
        # 如果帧是浮点数，假设范围是[0,1]，转换为[0,255]
        if frame.dtype in [np.float32, np.float64]:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        else:
            # 如果是其他整数类型，直接裁剪到uint8范围
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # 转换为PIL图像
    img = Image.fromarray(frame)
    
    # 转换为JPEG格式的base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=70)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str


def load_experiment_metadata(weights_path: str) -> dict:
    """从experiments目录中加载metadata信息"""
    try:
        # 如果是experiments目录中的模型，尝试读取metadata
        if 'experiments/' in weights_path:
            # 找到experiment目录路径
            parts = weights_path.split('/')
            experiment_path = None
            for i, part in enumerate(parts):
                if part == 'experiments':
                    experiment_path = '/'.join(parts[:i+3])  # experiments/env/algorithm/session
                    break
            
            if experiment_path:
                metadata_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), experiment_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        print(f"[INFO] 从metadata文件读取信息: {metadata_file}", file=sys.stderr)
                        return metadata
    except Exception as e:
        print(f"[WARNING] 无法读取metadata文件: {e}", file=sys.stderr)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='通用游戏推理脚本')
    parser.add_argument('--env', type=str, required=True, help='游戏环境名称')
    parser.add_argument('--weights', type=str, required=True, help='权重文件路径')
    parser.add_argument('--fps', type=int, default=5, help='帧率')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--algorithm', type=str, help='算法类型 (dqn, ppo, a2c)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 获取权重文件路径
    if os.path.isabs(args.weights):
        wpath = args.weights
    else:
        # 相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        wpath = os.path.join(project_root, args.weights)
    print(f"[INFO] 检测权重文件: {wpath}", file=sys.stderr)
    
    try:
        # 检测模型信息
        model_info = ModelFactory.detect_model_info(wpath)
        
        # 如果指定了算法类型，覆盖检测结果
        if args.algorithm:
            algorithm_map = {
                'dqn': 'CNNDQN',
                'ppo': 'CNNPPO', 
                'a2c': 'CNNA2C'
            }
            if args.algorithm.lower() in algorithm_map:
                model_info['model_type'] = algorithm_map[args.algorithm.lower()]
                print(f"[INFO] 使用指定的算法类型: {args.algorithm}", file=sys.stderr)
        
        # 尝试从experiments目录读取metadata信息来增强模型信息
        metadata = load_experiment_metadata(args.weights)
        if metadata:
            # 使用metadata中的信息来增强模型信息
            if 'algorithm' in metadata:
                print(f"[INFO] 从metadata检测到算法: {metadata['algorithm']}", file=sys.stderr)
            if 'config' in metadata and 'action_space' in metadata['config']:
                model_info['action_space'] = metadata['config']['action_space'].upper()
                print(f"[INFO] 从metadata检测到动作空间: {model_info['action_space']}", file=sys.stderr)
        
        print(f"[INFO] 检测到模型类型: {model_info['model_type']}", file=sys.stderr)
        print(f"[INFO] 检测到动作空间: {model_info['action_space']}", file=sys.stderr)
        
        # 创建模型
        model = ModelFactory.create_model(model_info)
        model = model.to(device)
        model.eval()
        print(f"[INFO] 模型加载成功", file=sys.stderr)
        
        # 创建环境
        print(f"[DEBUG] 正在创建环境，模型类型: {model_info['model_type']}", file=sys.stderr)
        if model_info['model_type'] == 'CNNDQN':
            # CNNDQN模型使用专用环境包装器
            print(f"[DEBUG] 使用DQN专用环境包装器", file=sys.stderr)
            from python.wrappers.mario_wrappers import wrap_environment
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
            
            if model_info['action_space'] == 'SIMPLE':
                actions = SIMPLE_MOVEMENT
            else:
                actions = COMPLEX_MOVEMENT
            
            env = wrap_environment(args.env, actions, monitor=False)
            print(f"[INFO] 使用DQN专用环境包装器", file=sys.stderr)
        elif model_info['model_type'] == 'CNNPPO':
            # PPO模型使用原始PPO环境包装器
            import gym
            from nes_py.wrappers import JoypadSpace
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
            from gym.wrappers import Monitor
            import cv2
            from collections import deque
            
            # 设置环境变量
            os.environ['PYGLET_HEADLESS'] = '1'
            cv2.ocl.setUseOpenCL(False)
            
            # 定义PPO专用环境包装器
            class NoopResetEnv(gym.Wrapper):
                def __init__(self, env, noop_max=30):
                    super().__init__(env)
                    self.noop_max = noop_max
                    self.override_num_noops = None
                    self.noop_action = 0
                    assert env.unwrapped.get_action_meanings()[0] == "NOOP"

                def reset(self, **kwargs):
                    self.env.reset(**kwargs)
                    if self.override_num_noops is not None:
                        noops = self.override_num_noops
                    else:
                        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
                    assert noops > 0
                    obs = None
                    for _ in range(noops):
                        obs, _, done, info = self.env.step(self.noop_action)
                        if done:
                            obs = self.env.reset(**kwargs)
                    return obs

                def step(self, ac):
                    return self.env.step(ac)

            class EpisodicLifeMario(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.lives = 0
                    self.was_real_done = True

                def step(self, action):
                    obs, reward, done, info = self.env.step(action)
                    self.was_real_done = done
                    lives = info.get("life", 0)
                    if lives < self.lives and lives > 0:
                        done = True
                    self.lives = lives
                    return obs, reward, done, info

                def reset(self, **kwargs):
                    if self.was_real_done:
                        obs = self.env.reset(**kwargs)
                        obs, _, _, info = self.env.step(0)
                        self.lives = info.get("life", 0)
                    else:
                        obs, _, _, info = self.env.step(0)
                        self.lives = info.get("life", 0)
                    return obs

            class MaxAndSkipEnv(gym.Wrapper):
                def __init__(self, env, skip=4):
                    super().__init__(env)
                    self._obs_buffer = deque(maxlen=2)
                    self._skip = skip

                def step(self, action):
                    total_reward = 0.0
                    done = None
                    for _ in range(self._skip):
                        obs, reward, done, info = self.env.step(action)
                        self._obs_buffer.append(obs)
                        total_reward += reward
                        if done:
                            break
                    max_frame = np.max(np.stack(self._obs_buffer), axis=0)
                    return max_frame, total_reward, done, info

                def reset(self, **kwargs):
                    self._obs_buffer.clear()
                    obs = self.env.reset(**kwargs)
                    self._obs_buffer.append(obs)
                    return obs

            class WarpFrame(gym.ObservationWrapper):
                def __init__(self, env, width=84, height=84):
                    super().__init__(env)
                    self._width = width
                    self._height = height
                    self.observation_space = gym.spaces.Box(
                        low=0, high=255, shape=(self._height, self._width, 1), dtype=np.uint8
                    )

                def observation(self, obs):
                    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                    frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
                    return frame[:, :, None]

            class ScaledFloatFrame(gym.ObservationWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.observation_space = gym.spaces.Box(
                        low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
                    )

                def observation(self, observation):
                    return np.array(observation).astype(np.float32) / 255.0

            class FrameStack(gym.Wrapper):
                def __init__(self, env, k):
                    super().__init__(env)
                    self.k = k
                    self.frames = deque([], maxlen=k)
                    shp = env.observation_space.shape
                    self.observation_space = gym.spaces.Box(
                        low=0, high=1, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype
                    )

                def reset(self, **kwargs):
                    ob = self.env.reset(**kwargs)
                    for _ in range(self.k):
                        self.frames.append(ob)
                    return self._get_ob()

                def step(self, action):
                    ob, reward, done, info = self.env.step(action)
                    self.frames.append(ob)
                    return self._get_ob(), reward, done, info

                def _get_ob(self):
                    assert len(self.frames) == self.k
                    return LazyFrames(list(self.frames))

            class LazyFrames(object):
                def __init__(self, frames):
                    self._frames = frames
                    self._out = None

                def _force(self):
                    if self._out is None:
                        self._out = np.concatenate(self._frames, axis=-1)
                        self._frames = None
                    return self._out

                def __array__(self, dtype=None):
                    out = self._force()
                    if dtype is not None:
                        out = out.astype(dtype)
                    return out

                def __len__(self):
                    return len(self._force())

                def __getitem__(self, i):
                    return self._force()[i]

                def count(self):
                    frames = self._force()
                    return frames.shape[frames.ndim - 1]

                def frame(self, i):
                    return self._force()[..., i]

            def make_env_mario(environment, monitor=False, monitor_path=None):
                """Creates and wraps a single Super Mario Bros environment with all wrappers."""
                env = gym.make(environment)  # 使用gym.make而不是gym_super_mario_bros.make
                if monitor:
                    env = Monitor(env, monitor_path, force=True, video_callable=lambda episode_id: True)
                env = JoypadSpace(env, COMPLEX_MOVEMENT)
                env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)
                env = EpisodicLifeMario(env)
                env = WarpFrame(env, width=84, height=84)
                env = ScaledFloatFrame(env)
                env = FrameStack(env, k=4)
                return env
            
            if model_info['action_space'] == 'SIMPLE':
                actions = SIMPLE_MOVEMENT
            else:
                actions = COMPLEX_MOVEMENT
            
            env = make_env_mario(args.env, monitor=False)
            print(f"[INFO] 使用PPO原始环境包装器", file=sys.stderr)
        else:
            # 其他模型使用通用环境工厂
            env = EnvironmentFactory.create_environment(
                args.env, 
                action_space_type=model_info['action_space']
            )
            print(f"[INFO] 使用通用环境工厂", file=sys.stderr)
        
        # 获取动作列表
        if model_info['model_type'] in ['CNNDQN', 'CNNPPO']:
            # CNNDQN和PPO模型使用动作空间
            actions = actions  # 已经在上面定义了
        else:
            # 其他模型使用环境工厂
            actions = env.get_actions()
        
        # 初始化环境 - 完全按照你的代码逻辑
        print(f"[DEBUG] 正在初始化环境...", file=sys.stderr)
        state = env.reset()
        print(f"[DEBUG] 环境初始化完成，state shape: {state.shape if hasattr(state, 'shape') else type(state)}", file=sys.stderr)
        
        # 兼容旧版Gym：确保state是numpy数组
        if isinstance(state, tuple):
            state = state[0]  # 如果是tuple，取第一个元素
            print(f"[DEBUG] 从tuple中提取state，新shape: {state.shape}", file=sys.stderr)
        
        total_reward = 0.0
        frame_count = 0
        
        print(f"[INFO] 开始推理，FPS: {args.fps}", file=sys.stderr)
        
        while True:
            start_time = time.time()
            
            # 完全按照你的代码逻辑：直接使用环境包装器处理后的观察
            state_v = torch.tensor(np.array([state], copy=True))
            state_v = state_v.to(device)
            
            # 模型推理
            with torch.no_grad():
                if model_info['model_type'] == 'CNNPPO':
                    # PPO模型推理
                    logits, value = model(state_v)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.probs.argmax(dim=-1).item()
                else:
                    # DQN/A2C模型推理
                    q_vals = model(state_v).data.cpu().numpy()[0]
                    action = np.argmax(q_vals)
            
            # 执行动作
            action_name = actions[action]
            
            # 环境步进 - 兼容旧版Gym
            step_result = env.step(action)
            
            # 处理不同版本的返回值
            if len(step_result) == 4:
                # 旧版Gym: (obs, reward, done, info)
                state, reward, done, info = step_result
            elif len(step_result) == 5:
                # 新版Gym: (obs, reward, terminated, truncated, info)
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
            total_reward += reward
            
            # 渲染帧用于显示
            frame = env.render(mode='rgb_array')
            frame_base64 = frame_to_base64(frame)
            
            # 输出JSON
            current_time = time.time()
            output = {
                "type": "frame",
                "t": frame_count,
                "timestamp": current_time,
                "agent_action": action_name,
                "reward": float(reward),
                "frame": frame_base64
            }
            print(json.dumps(output, ensure_ascii=False), flush=True)
            
            # 检查是否获得旗帜
            if info.get('flag_get', False):
                print(f"[INFO] 获得旗帜！", file=sys.stderr)
            
            # 游戏结束后停止循环
            if done:
                print(f"[INFO] 游戏结束，总奖励: {total_reward}", file=sys.stderr)
                break
            
            frame_count += 1
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"[ERROR] 推理失败: {e}", file=sys.stderr)
        return 1
    
    finally:
        if 'env' in locals():
            env.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
