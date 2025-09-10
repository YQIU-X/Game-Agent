# 游戏代理平台 - 智能训练实验管理系统

## 📚 参考项目

本项目基于以下优秀的开源项目构建：

- **[Super Mario Bros DQN](https://github.com/roclark/super-mario-bros-dqn)** - 使用PyTorch训练的DQN模型来通关超级马里奥兄弟
- **[Super Mario RL](https://github.com/jiseongHAN/Super-Mario-RL)** - 基于强化学习的超级马里奥游戏代理

感谢这些项目的开源贡献，为我们的游戏代理平台提供了重要的技术基础。

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
