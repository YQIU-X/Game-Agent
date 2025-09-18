## 🎮 项目演示

### 平台功能演示
#### 玩家端
<video src="static/player.mp4" controls muted autoplay loop></video>


##### 开发者端
<video src="static/developer.mp4" controls muted autoplay loop></video>

*平台界面和功能演示 - 开发者模式*

### 智能体游戏演示

The following table shows the current progress of the model on various levels
and the settings used to achieve the indicated performance:

| Level | Status | Actions | DQN | PPO |
|:---|:---|:---|:---|:---|
| **World 1-1** | Optimal | Complex | ![][1-1] | N/A |
| **World 1-2** | Optimal | Simple | ![][1-2] | N/A |
| **World 1-3** | Optimal | Simple | ![][1-3] | N/A |
| **World 1-4** | Optimal | Simple | ![][1-4] | N/A |
| **World 2-1** | Untested | N/A | N/A | N/A |
| **World 2-2** | Untested | N/A | N/A | N/A |
| **World 2-3** | Optimal | Simple | ![][2-3] | N/A |
| **World 2-4** | Optimal | Simple | ![][2-4] | N/A |
| **World 3-1** | Untested | N/A | N/A | N/A |
| **World 3-2** | Optimal | Simple | ![][3-2] | N/A |
| **World 3-3** | Untested | N/A | N/A | ![][3-3PPO] |
| **World 3-4** | Optimal | Simple | ![][3-4] | ![][3-4PPO] |
| **World 4-1** | Untested | N/A | N/A | N/A |
| **World 4-2** | Untested | N/A | N/A | N/A |
| **World 4-3** | Optimal | Simple | ![][4-3] | N/A |
| **World 4-4** | Untested | N/A | N/A | N/A |
| **World 5-1** | Untested | N/A | N/A | N/A |
| **World 5-2** | Untested | N/A | N/A | N/A |
| **World 5-3** | Untested | N/A | N/A | N/A |
| **World 5-4** | Optimal | Simple | ![][5-4] | N/A |
| **World 6-1** | Optimal | Simple | ![][6-1] | N/A |
| **World 6-2** | Untested | N/A | N/A | N/A |
| **World 6-3** | Untested | N/A | N/A | N/A |
| **World 6-4** | Optimal | Simple | ![][6-4] | N/A |
| **World 7-1** | Untested | N/A | N/A | N/A |
| **World 7-2** | Untested | N/A | N/A | N/A |
| **World 7-3** | Optimal | Simple | ![][7-3] | N/A |

[1-1]: static/smb-1-1-complete.gif
[1-2]: static/smb-1-2-complete.gif
[1-3]: static/smb-1-3-complete.gif
[1-4]: static/smb-1-4-complete.gif
[2-3]: static/smb-2-3-complete.gif
[2-4]: static/smb-2-4-complete.gif
[3-2]: static/smb-3-2-complete.gif
[3-4]: static/smb-3-4-complete.gif
[4-3]: static/smb-4-3-complete.gif
[5-4]: static/smb-5-4-complete.gif
[6-1]: static/smb-6-1-complete.gif
[6-4]: static/smb-6-4-complete.gif
[7-3]: static/smb-7-3-complete.gif
[3-3PPO]: static/PPO-3-3.gif
[3-4PPO]: static/PPO-3-4.gif

### 部分智能体训练日志



## 📖 项目介绍

### 🎯 项目概述
游戏代理平台是一个基于强化学习的智能游戏训练实验管理系统，旨在为研究人员和开发者提供一个完整的游戏AI训练、测试和部署平台。平台支持多种强化学习算法（DQN、PPO、A2C等）和游戏环境（Super Mario Bros、Atari等），提供可视化的训练过程监控和智能体性能评估。

### 🚀 核心功能
- **智能实验管理**: 自动组织和管理训练实验，支持超参数对比和结果分析
- **多算法支持**: 集成DQN、PPO、A2C等主流强化学习算法
- **多环境兼容**: 支持Super Mario Bros、Atari等多种游戏环境
- **可视化界面**: 提供Web界面进行训练监控、模型管理和智能体测试
- **模型推理**: 支持训练好的模型进行游戏推理和性能评估
- **实验对比**: 自动记录训练指标，支持不同实验结果的对比分析

### 🎨 技术特色
- **模块化设计**: 采用模块化架构，易于扩展新算法和游戏环境
- **智能文件管理**: 自动检测和组织模型文件，支持新旧格式兼容
- **实时监控**: 训练过程中实时显示损失、奖励等关键指标
- **用户友好**: 提供开发者模式和玩家模式，满足不同用户需求


## 环境介绍

### Action spaces
This repository allows users to specify a custom set of actions that Mario can
use with various degrees of complexity. Choosing a simpler action space makes it
quicker and easier for Mario to learn, but prevents him from trying more complex
movements which can include entering pipes and making advanced jumps which might
be required to solve some levels. If Mario appears to struggle with a particular
level, try simplifying the action space to see if he makes further progress.

Currently, the following options are supported:

#### Right only
Mario can effectively only go right. This simplifies the training process, but
prevents Mario from trying more complex actions. The following buttons are
supported:
  * Nothing
  * Right
  * Right + A
  * Right + B
  * Right + A + B

#### Simple movement
In addition to moving right and running/jumping, Mario can now walk left and
jump in place. The following buttons are supported:
  * Nothing
  * Right
  * Right + A
  * Right + B
  * Right + A + B
  * A
  * Left

#### Complex movement
This action allows Mario to try nearly any of his possible actions from the
game. This option should be chosen by default for the most realistic exploration
of a level, but can increase the time and complexity of learning a level. This
is the only provided action space that allows Mario to enter vertically-oriented
pipes. The following buttons are supported:
  * Nothing
  * Right
  * Right + A
  * Right + B
  * Right + A + B
  * A
  * Left
  * Left + A
  * Left + B
  * Left + A + B
  * Down
  * Up


#### Version
The version of the environment that was tested. See the Environments section of
[gym-super-mario-bros' README](https://github.com/Kautenja/gym-super-mario-bros/blob/master/README.md#environments)
for examples of the various environment versions.

#### Status
The current status of training for the indicated level. The status can take on
the following values:
  * **Untested**: No attempts or progress has been made on training for the
given level yet.
  * **Training**: Training has begun for the indicated level, but Mario has not
yet completed the level. If a model is provided, it will correspond to the most
recent training pass achieved, and not necessarily the best run so far.
  * **Satisfactory**: Mario can successfully complete the level, but is
currently unable to do so in an optimal manner for any reason, including
standing in place, losing health, not making forward progress, or others.
  * **Optimal**: Mario has trained enough that he can beat the level at
near-optimal performance. This does not necessarily mean the run is perfect, but
he can complete the level with only a couple minor interruptions at most. In
this state, further progress will likely not be made.

#### Actions
The action-space that Mario has been trained to use. See "Action spaces" above
for more details on the various action spaces.

#### GIF
An animated GIF of the run that corresponds to the saved model provided in the
repository.




## 环境
```bash
conda create -n [env name] python=3.8
conda activate [env name]
pip install -r requirement.txt
```

## 启动
```bash
conda activate [env name]
npm install concurrently --save-dev
npm run dev
```

## 登录
```javascript
// 模拟用户数据
const users = [
  { username: 'developer', password: 'dev123', role: 'developer' },
  { username: 'player', password: 'play123', role: 'player' }
];
```




## ✨ 核心特性

### 🎯 **智能实验管理**
- **结构化存储**: 按环境、算法、超参数自动组织训练实验
- **完整记录**: 每个实验包含配置、指标、日志、权重等所有文件
- **易于比较**: 轻松对比不同超参数设置的效果
- **向后兼容**: 支持旧格式文件，平滑迁移

### 🚀 **多算法支持**
- **DQN**: 深度Q网络，适合离散动作空间
- **PPO**: 近端策略优化，适合连续和离散动作空间
- **A2C**: 优势演员评论家，适合策略梯度方法

### 🎮 **多游戏环境**
- **Super Mario Bros**: 经典平台游戏
- **Atari Games**: 经典街机游戏
- **可扩展**: 易于添加新游戏类型

## 📁 项目结构

```
game-agent-platform/
├── 📁 experiments/               # 🆕 智能实验管理系统
│   ├── 📁 {environment}/        # 环境目录 (如: SuperMarioBros-1-1-v0)
│   │   ├── 📁 {algorithm}/      # 算法目录 (如: DQN)
│   │   │   ├── 📁 {session_id}/ # 训练会话目录 (如: 20241225_143022_lr1e4_gamma99)
│   │   │   │   ├── config.json  # 训练配置参数
│   │   │   │   ├── metrics.csv  # 训练指标
│   │   │   │   ├── logs.txt     # 训练日志
│   │   │   │   ├── weights/     # 权重文件目录
│   │   │   │   │   ├── best_model.pth      # 最佳模型
│   │   │   │   │   ├── checkpoint_100.pth  # 检查点
│   │   │   │   │   └── final_model.pth     # 最终模型
│   │   │   │   └── metadata.json # 元数据信息
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── 📁 python/                   # Python后端代码
│   ├── 📁 scripts/              # 可执行脚本
│   │   ├── agent_inference.py   # 智能体推理脚本
│   │   ├── player_control.py    # 玩家控制脚本
│   │   ├── train_dqn.py         # DQN训练脚本
│   │   └── train_ppo.py         # PPO训练脚本
│   ├── 📁 algorithms/           # 算法实现
│   │   ├── 📁 dqn/             # DQN算法
│   │   │   ├── 📁 core/        # 核心组件
│   │   │   └── trainer.py      # 训练器
│   │   └── 📁 ppo/             # PPO算法
│   ├── 📁 games/               # 游戏环境
│   │   ├── 📁 mario/           # Mario游戏
│   │   └── 📁 atari/           # Atari游戏
│   ├── 📁 utils/               # 工具模块
│   │   ├── experiment_manager.py # 🆕 实验管理器
│   │   ├── model_factory.py    # 模型工厂
│   │   ├── environment_factory.py # 环境工厂
│   │   └── weight_detector.py  # 权重检测器
│   ├── 📁 configs/             # 配置文件
│   │   ├── algorithm_configs.py # 算法配置
│   │   └── config.py           # 主配置文件
│   ├── 📁 models/              # 模型定义
│   │   ├── base.py             # 基础模型类
│   │   ├── cnndqn.py          # CNN DQN模型
│   │   ├── cnnppo.py          # CNN PPO模型
│   │   └── cnna2c.py          # CNN A2C模型
│   ├── 📁 environments/        # 环境定义
│   │   ├── base.py             # 基础环境类
│   │   ├── mario.py            # Mario环境
│   │   └── atari.py            # Atari环境
│   └── 📁 wrappers/            # 环境包装器
│       ├── base_wrapper.py     # 基础包装器类
│       ├── mario_wrappers.py   # Mario游戏包装器
│       └── atari_wrappers.py   # Atari游戏包装器
├── 📁 pretrained_models/       # 预训练模型 (向后兼容)
│   ├── 📁 mario/              # Mario游戏模型
│   │   └── 📁 dqn/           # DQN算法模型
│   └── 📁 atari/              # Atari游戏模型
├── 📁 training_metrics/        # 训练指标 (向后兼容)
├── 📁 src/                     # Vue.js前端代码
│   ├── 📁 api/                # API客户端
│   ├── 📁 components/         # Vue组件
│   ├── 📁 services/           # 服务模块
│   ├── 📁 views/              # 页面视图
│   │   ├── Developer.vue      # 开发者界面
│   │   ├── Player.vue         # 玩家界面
│   │   └── Login.vue          # 登录界面
│   └── 📁 router/             # 路由配置
├── 📁 public/                  # 静态资源
├── server.js                   # Node.js服务器
├── package.json                # Node.js依赖
└── requirements.txt            # Python依赖
```

## 🎯 设计原则

### 1. **智能实验管理** 🆕
- **结构化存储**: 按环境、算法、超参数自动组织训练实验
- **完整记录**: 每个实验包含配置、指标、日志、权重等所有文件
- **易于比较**: 轻松对比不同超参数设置的效果
- **向后兼容**: 支持旧格式文件，平滑迁移

### 2. **模块化设计**
- **脚本模块** (`python/scripts/`): 独立的可执行脚本
- **算法模块** (`python/algorithms/`): 按算法类型组织
- **游戏模块** (`python/games/`): 按游戏类型组织
- **工具模块** (`python/utils/`): 通用工具和工厂类
- **配置模块** (`python/configs/`): 集中配置管理

### 3. **实验目录结构** 🆕
```
experiments/
├── {environment}/                    # 环境目录
│   ├── {algorithm}/                  # 算法目录
│   │   ├── {session_id}/             # 训练会话目录
│   │   │   ├── config.json           # 训练配置参数
│   │   │   ├── metrics.csv           # 训练指标
│   │   │   ├── logs.txt              # 训练日志
│   │   │   ├── weights/              # 权重文件目录
│   │   │   └── metadata.json         # 元数据信息
│   │   └── ...
│   └── ...
└── ...
```

### 4. **会话ID生成规则** 🆕
- **格式**: `{timestamp}_{key_params}`
- **示例**: `20241225_143022_lr1e4_gamma99_eps1e4`
- **优势**: 从文件名就能看出训练参数和时间

## 🔧 扩展指南

### 添加新游戏类型
1. 在 `python/games/` 创建新游戏目录
2. 在 `python/environments/` 创建新环境类
3. 在 `python/wrappers/` 创建专用包装器
4. 在 `python/configs/` 添加配置
5. 在 `experiments/` 创建对应目录结构

### 添加新算法
1. 在 `python/algorithms/` 创建新算法目录
2. 在 `python/models/` 创建新模型类
3. 在 `python/configs/` 添加算法配置
4. 更新实验管理器以支持新算法

### 添加新模型文件
1. **新格式**: 将模型文件放入对应的 `experiments/{env}/{alg}/{session}/weights/` 目录
2. **旧格式**: 将模型文件放入对应的 `pretrained_models/{game}/{algorithm}/` 目录
3. 更新 `python/utils/weight_detector.py` 以支持新格式
4. 测试模型加载和推理

## 🚀 使用方法

### 启动服务器
```bash
npm run serve
```

### 开始训练实验 🆕
```bash
# DQN训练 - 自动创建实验目录
python python/scripts/train_dqn.py --environment SuperMarioBros-1-1-v0 --episodes 1000

# PPO训练 - 自动创建实验目录
python python/scripts/train_ppo.py --environment SuperMarioBros-1-1-v0 --episodes 1000
```

### 运行Python脚本
```bash
# 智能体推理 - 支持新实验目录
python python/scripts/agent_inference.py --env SuperMarioBros-1-1-v0 --weights experiments/SuperMarioBros-1-1-v0/DQN/20241225_143022_lr1e4_gamma99/weights/best_model.pth

# 玩家控制
python python/scripts/player_control.py --env SuperMarioBros-1-1-v0 --action-space SIMPLE
```

### 实验管理 🆕
```python
from python.utils.experiment_manager import experiment_manager

# 列出所有实验
experiments = experiment_manager.list_experiments()

# 获取最新实验
latest = experiment_manager.get_latest_experiment('SuperMarioBros-1-1-v0', 'DQN')

# 清理旧实验
experiment_manager.cleanup_old_experiments(keep_count=10)
```

## 📝 注意事项

1. **环境一致性**: 确保推理环境与训练环境完全一致
2. **模型兼容性**: 不同算法的模型不能混用
3. **文件路径**: 使用相对路径，确保跨平台兼容
4. **依赖管理**: 保持Python和Node.js依赖的同步更新
5. **实验管理** 🆕: 定期清理旧实验，避免磁盘空间不足
6. **向后兼容**: 系统同时支持新旧文件格式，无需手动迁移

## 🎉 新功能亮点

### 🆕 **智能实验管理系统**
- **自动组织**: 训练开始时自动创建结构化目录
- **完整记录**: 配置、指标、日志、权重统一管理
- **易于比较**: 从文件名就能看出训练参数
- **向后兼容**: 支持旧格式文件，平滑迁移

### 🆕 **增强的前端界面**
- **训练可视化**: 自动检测和显示最新训练记录
- **模型选择**: 支持实验目录中的模型文件
- **参数配置**: 可编辑的代码编辑器，支持语法高亮
- **实时更新**: 训练过程中自动更新可视化图表

### 🆕 **改进的API接口**
- **实验列表**: 获取所有训练实验的详细信息
- **文件管理**: 支持新旧文件格式的统一管理
- **模型检测**: 自动识别和分类不同类型的模型文件

## 🔮 未来规划

- [ ] 支持更多强化学习算法 (A3C, SAC, TD3等)
- [ ] 添加实验对比分析功能
- [ ] 实现分布式训练支持
- [ ] 添加模型性能评估工具
- [ ] 支持超参数自动调优

## 📚 参考项目

本项目基于以下优秀的开源项目构建：

- **[Super Mario Bros DQN](https://github.com/roclark/super-mario-bros-dqn)** - 使用PyTorch训练的DQN模型来通关超级马里奥兄弟
- **[Super Mario RL](https://github.com/jiseongHAN/Super-Mario-RL)** - 基于强化学习的超级马里奥游戏代理

感谢这些项目的开源贡献，为我们的游戏代理平台提供了重要的技术基础。
