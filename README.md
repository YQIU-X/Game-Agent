# 通用游戏智能体平台

这是一个支持多种游戏和强化学习算法的通用智能体平台。

## 架构特点

### 🎯 模块化设计
- **模型层**: 支持多种神经网络架构（DQN、PPO、A2C等）
- **环境层**: 支持多种游戏环境（Mario、Atari等）
- **工厂模式**: 自动检测和创建正确的模型和环境

### 🔧 自动检测
- 自动检测权重文件的模型类型
- 自动检测动作空间（SIMPLE/COMPLEX）
- 自动检测输入形状和网络结构

### 🚀 易于扩展
- 添加新模型：只需继承 `BaseModel` 类
- 添加新环境：只需继承 `BaseGameEnv` 类
- 添加新游戏：只需在配置文件中注册

## 文件结构

```
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── base.py            # 基础模型类
│   ├── cnndqn.py          # CNN DQN模型
│   ├── cnnppo.py          # CNN PPO模型
│   └── cnna2c.py          # CNN A2C模型
├── environments/           # 环境定义
│   ├── __init__.py
│   ├── base.py            # 基础环境类
│   ├── mario.py           # 马里奥环境
│   └── atari.py           # Atari环境
├── model_factory.py        # 模型工厂
├── environment_factory.py  # 环境工厂
├── config.py              # 配置文件
├── infer_stream_v2.py     # 通用推理脚本
└── pretrained_models/     # 预训练模型
```

## 支持的模型

| 模型类型 | 算法 | 描述 |
|---------|------|------|
| CNNDQN | DQN, DDQN, DuelingDQN | 深度Q网络 |
| CNNPPO | PPO | 近端策略优化 |
| CNNA2C | A2C, A3C | 优势演员评论家 |

## 支持的环境

| 游戏类型 | 环境 | 动作空间 |
|---------|------|----------|
| Mario | SuperMarioBros | SIMPLE (7), COMPLEX (12) |
| Atari | ALE | DISCRETE |

## 使用方法

### 1. 基本推理
```bash
python infer_stream_v2.py --env "SuperMarioBros-1-1-v0" --weights "mario_dqn.dat"
```

### 2. 指定设备
```bash
python infer_stream_v2.py --env "SuperMarioBros-1-1-v0" --weights "mario_dqn.dat" --device cuda
```

### 3. 控制帧率
```bash
python infer_stream_v2.py --env "SuperMarioBros-1-1-v0" --weights "mario_dqn.dat" --fps 30
```

## 扩展指南

### 添加新模型

1. 在 `models/` 目录下创建新模型文件
2. 继承 `BaseModel` 类
3. 实现必要的方法：`forward()`, `act()`
4. 在 `model_factory.py` 中注册新模型

```python
# models/my_model.py
from .base import BaseModel

class MyModel(BaseModel):
    def __init__(self, input_shape, num_actions):
        super().__init__(input_shape, num_actions)
        # 定义网络结构
        
    def forward(self, x):
        # 前向传播
        pass
        
    def act(self, state_tensor):
        # 选择动作
        pass
```

### 添加新环境

1. 在 `environments/` 目录下创建新环境文件
2. 继承 `BaseGameEnv` 类
3. 实现必要的方法：`_setup_environment()`, `get_actions()`, `preprocess_frame()`
4. 在 `environment_factory.py` 中注册新环境

```python
# environments/my_game.py
from .base import BaseGameEnv

class MyGameEnv(BaseGameEnv):
    def _setup_environment(self):
        # 设置游戏环境
        pass
        
    def get_actions(self):
        # 返回动作列表
        pass
        
    def preprocess_frame(self, frame):
        # 预处理帧
        pass
```

### 添加新游戏

1. 在 `config.py` 中注册新游戏
2. 创建对应的环境类
3. 更新工厂类中的检测逻辑

## 权重文件格式

系统支持以下权重文件格式：

1. **PyTorch格式** (.pth, .pt, .dat)
2. **Pickle格式** (.pkl)

权重文件应包含：
- 模型权重（state_dict）
- 模型类型信息（可选）
- 动作空间信息（可选）
- 输入形状信息（可选）

## 自动检测机制

### 模型类型检测
1. 检查权重文件中的模型类型标识
2. 分析网络层结构（actor/critic网络等）
3. 根据特征推断模型类型

### 动作空间检测
1. 检查权重文件中的动作空间标识
2. 分析输出层维度
3. 根据动作数量推断动作空间类型

### 输入形状检测
1. 检查权重文件中的输入形状标识
2. 分析第一个卷积层的输入通道数
3. 默认使用 (4, 84, 84)

## 性能优化

- 支持GPU加速（CUDA）
- 帧率控制
- 内存优化
- 批处理推理

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查权重文件路径
   - 确认文件格式正确
   - 检查模型类型是否支持

2. **动作空间不匹配**
   - 检查权重文件的输出维度
   - 确认动作空间类型正确

3. **环境创建失败**
   - 检查环境名称
   - 确认依赖包已安装

### 调试模式

启用详细日志：
```bash
python infer_stream_v2.py --env "SuperMarioBros-1-1-v0" --weights "mario_dqn.dat" 2>&1 | tee debug.log
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License
