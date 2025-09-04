# 游戏代理平台 - 项目结构

## 📚 参考项目

本项目基于以下优秀的开源项目构建：

- **[Super Mario Bros DQN](https://github.com/roclark/super-mario-bros-dqn)** - 使用PyTorch训练的DQN模型来通关超级马里奥兄弟
- **[Super Mario RL](https://github.com/jiseongHAN/Super-Mario-RL)** - 基于强化学习的超级马里奥游戏代理

感谢这些项目的开源贡献，为我们的游戏代理平台提供了重要的技术基础。

## 📁 项目结构

```
game-agent-platform/
├── 📁 python/                    # Python后端代码
│   ├── 📁 scripts/              # 可执行脚本
│   │   ├── agent_inference.py   # 智能体推理脚本
│   │   └── player_control.py    # 玩家控制脚本
│   ├── 📁 wrappers/             # 环境包装器
│   │   ├── base_wrapper.py      # 基础包装器类
│   │   ├── mario_wrappers.py    # Mario游戏包装器
│   │   └── atari_wrappers.py    # Atari游戏包装器
│   ├── 📁 utils/                # 工具模块
│   │   ├── model_factory.py     # 模型工厂
│   │   ├── environment_factory.py # 环境工厂
│   │   └── weight_detector.py   # 权重检测器
│   ├── 📁 configs/              # 配置文件
│   │   └── config.py            # 主配置文件
│   ├── 📁 models/               # 模型定义
│   │   ├── base.py              # 基础模型类
│   │   ├── cnndqn.py           # CNN DQN模型
│   │   ├── cnnppo.py           # CNN PPO模型
│   │   └── cnna2c.py           # CNN A2C模型
│   └── 📁 environments/         # 环境定义
│       ├── base.py              # 基础环境类
│       ├── mario.py             # Mario环境
│       └── atari.py             # Atari环境
├── 📁 pretrained_models/        # 预训练模型
│   ├── 📁 mario/               # Mario游戏模型
│   │   ├── 📁 dqn/            # DQN算法模型
│   │   ├── 📁 ppo/            # PPO算法模型
│   │   └── 📁 a2c/            # A2C算法模型
│   └── 📁 atari/               # Atari游戏模型
├── 📁 src/                      # Vue.js前端代码
│   ├── 📁 api/                 # API客户端
│   ├── 📁 components/         # Vue组件
│   ├── 📁 services/           # 服务模块
│   ├── 📁 views/              # 页面视图
│   └── 📁 router/             # 路由配置
├── 📁 public/                   # 静态资源
├── server.js                    # Node.js服务器
├── package.json                 # Node.js依赖
└── requirements.txt             # Python依赖
```

## 🎯 设计原则

### 1. **模块化设计**
- **脚本模块** (`python/scripts/`): 独立的可执行脚本
- **包装器模块** (`python/wrappers/`): 环境包装器，按游戏类型分类
- **工具模块** (`python/utils/`): 通用工具和工厂类
- **配置模块** (`python/configs/`): 集中配置管理

### 2. **游戏类型分离**
- **Mario游戏**: 专用包装器和配置
- **Atari游戏**: 标准化的包装器
- **未来扩展**: 易于添加新游戏类型

### 3. **算法类型分离**
- **DQN**: 深度Q网络，适合离散动作空间
- **PPO**: 近端策略优化，适合连续和离散动作空间
- **A2C**: 优势演员评论家，适合策略梯度方法

### 4. **模型文件组织**
```
pretrained_models/
├── mario/
│   ├── dqn/          # DQN算法训练的Mario模型
│   ├── ppo/          # PPO算法训练的Mario模型
│   └── a2c/          # A2C算法训练的Mario模型
└── atari/
    ├── dqn/          # DQN算法训练的Atari模型
    ├── ppo/          # PPO算法训练的Atari模型
    └── a2c/          # A2C算法训练的Atari模型
```

## 🔧 扩展指南

### 添加新游戏类型
1. 在 `python/environments/` 创建新环境类
2. 在 `python/wrappers/` 创建专用包装器
3. 在 `python/configs/config.py` 添加配置
4. 在 `pretrained_models/` 创建对应目录

### 添加新算法
1. 在 `python/models/` 创建新模型类
2. 在 `python/configs/config.py` 添加算法配置
3. 更新工厂类以支持新算法

### 添加新模型文件
1. 将模型文件放入对应的 `pretrained_models/[game]/[algorithm]/` 目录
2. 更新 `python/utils/weight_detector.py` 以支持新格式
3. 测试模型加载和推理

## 🚀 使用方法

### 启动服务器
```bash
npm run dev
```

### 运行Python脚本
```bash
# 智能体推理
python python/scripts/agent_inference.py --env SuperMarioBros-v0 --weights pretrained_models/mario/dqn/SuperMarioBros-1-1-v0.dat

# 玩家控制
python python/scripts/player_control.py --env SuperMarioBros-v0 --action-space SIMPLE
```

## 📝 注意事项

1. **环境一致性**: 确保推理环境与训练环境完全一致
2. **模型兼容性**: 不同算法的模型不能混用
3. **文件路径**: 使用相对路径，确保跨平台兼容
4. **依赖管理**: 保持Python和Node.js依赖的同步更新
