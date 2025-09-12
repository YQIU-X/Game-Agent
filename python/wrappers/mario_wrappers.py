import cv2
import numpy as np
from collections import deque
from gym import make, ObservationWrapper, wrappers, Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace


class FrameDownsample(ObservationWrapper):
    def __init__(self, env):
        super(FrameDownsample, self).__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(84, 84, 1),
                                     dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self._width, self._height),
                           interpolation=cv2.INTER_AREA)
        # 确保返回的帧是uint8格式
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame[:, :, None]


class MaxAndSkipEnv(Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            step_result = self.env.step(action)
            
            # 处理不同版本的返回值
            if len(step_result) == 4:
                # 旧版Gym: (obs, reward, done, info)
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                # 新版Gym: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
                
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        # 确保max_frame是uint8格式
        max_frame = np.clip(max_frame, 0, 255).astype(np.uint8)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        reset_result = self.env.reset()
        
        # 兼容旧版Gym：确保obs是numpy数组
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # 如果是tuple，取第一个元素
        else:
            obs = reset_result
            
        self._obs_buffer.append(obs)
        return obs


class FireResetEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        if len(env.unwrapped.get_action_meanings()) < 3:
            raise ValueError('Expected an action space of at least 3!')

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        
        # 兼容旧版Gym：确保obs是numpy数组
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # 如果是tuple，取第一个元素
        else:
            obs = reset_result
        
        # 第一次step
        step_result = self.env.step(1)
        if len(step_result) == 4:
            obs, _, done, _ = step_result
        elif len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
            
        if done:
            reset_result = self.env.reset(**kwargs)
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            
        # 第二次step
        step_result = self.env.step(2)
        if len(step_result) == 4:
            obs, _, done, _ = step_result
        elif len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            raise ValueError(f"未知的环境步进返回值格式: {len(step_result)}")
            
        if done:
            reset_result = self.env.reset(**kwargs)
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
        return obs

    def step(self, action):
        return self.env.step(action)


class FrameBuffer(ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBuffer, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),
                                     obs_space.high.repeat(num_steps, axis=0),
                                     dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
                                    dtype=self._dtype)
        reset_result = self.env.reset()
        
        # 兼容旧版Gym：确保obs是numpy数组
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # 如果是tuple，取第一个元素
        else:
            obs = reset_result
            
        return self.observation(obs)

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(obs_shape[::-1]),
                                     dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class NormalizeFloats(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        step_result = self.env.step(action)
        
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
            
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info
        

def wrap_environment(environment, action_space, monitor=False, iteration=0):
    env = make(environment)
    if monitor:
        env = wrappers.Monitor(env, '%s_recording/run%s' %(environment, iteration), force=True)
    env = JoypadSpace(env, action_space)
    env = MaxAndSkipEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = FrameDownsample(env)
    env = ImageToPyTorch(env)
    env = FrameBuffer(env, 4)
    env = NormalizeFloats(env)
    env = CustomReward(env)
    return env
