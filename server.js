const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { exec, spawn } = require('child_process');
const WebSocket = require('ws');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_BIN = process.env.PYTHON_BIN || 'python';

// 中间件
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'dist')));

// 模拟用户数据
const users = [
  { username: 'developer', password: 'dev123', role: 'developer' },
  { username: 'player', password: 'play123', role: 'player' }
];

// 创建WebSocket服务器
const wss = new WebSocket.Server({ noServer: true });

// 存储WebSocket连接
const wsConnections = new Map();
// 存储运行中的进程
const runningProcesses = {};

// 登录接口
app.post('/api/login', (req, res) => {
  const { username, password } = req.body;
  const user = users.find(u => u.username === username && u.password === password);
  
  if (user) {
    res.json({ success: true, user: { username: user.username, role: user.role } });
  } else {
    res.status(401).json({ success: false, message: '用户名或密码错误' });
  }
});

// 配置文件目录
const configDir = path.join(__dirname, 'configs');

// 确保配置目录存在
if (!fs.existsSync(configDir)) {
  fs.mkdirSync(configDir, { recursive: true });
}

// 获取配置文件列表
app.get('/api/configs', (req, res) => {
  fs.readdir(configDir, (err, files) => {
    if (err) {
      return res.status(500).json({ success: false, message: '无法读取配置文件' });
    }
    
    const configFiles = files.filter(file => file.endsWith('.json'));
    res.json({ success: true, configs: configFiles });
  });
});

// 读取配置文件
app.get('/api/configs/:filename', (req, res) => {
  const filePath = path.join(configDir, req.params.filename);
  
  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      return res.status(404).json({ success: false, message: '配置文件不存在' });
    }
    
    try {
      const config = JSON.parse(data);
      res.json({ success: true, config });
    } catch (e) {
      res.status(500).json({ success: false, message: '配置文件格式错误' });
    }
  });
});

// 保存配置文件
app.post('/api/configs/:filename', (req, res) => {
  const filePath = path.join(configDir, req.params.filename);
  const config = req.body;
  
  fs.writeFile(filePath, JSON.stringify(config, null, 2), 'utf8', (err) => {
    if (err) {
      return res.status(500).json({ success: false, message: '保存配置文件失败' });
    }
    
    res.json({ success: true, message: '配置文件保存成功' });
  });
});

// 执行Python脚本并获取输出
function runPythonScript(args) {
  return new Promise((resolve, reject) => {
    console.log(`[Python] 启动脚本: ${PYTHON_BIN} ${args.join(' ')}`);
    const pythonProcess = spawn(PYTHON_BIN, args);
    let output = '';
    let errorOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      // 实时显示错误信息
      console.error(`[Python Error] ${data.toString()}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`[Python] 脚本退出，退出码: ${code}`);
      if (code !== 0) {
        console.error(`[Python] 脚本执行失败，完整错误输出: ${errorOutput}`);
        reject(new Error(`Python脚本执行失败: ${errorOutput}`));
      } else {
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (e) {
          resolve(output);
        }
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error(`[Python] 启动脚本失败: ${error.message}`);
      reject(new Error(`启动Python脚本失败: ${error.message}`));
    });
  });
}

// 获取可用游戏列表
app.get('/api/games', async (req, res) => {
  try {
    const result = await runPythonScript(['agent_integration.py', 'list-games']);
    const games = JSON.parse(result).map(gameId => {
      return { 
        id: gameId, 
        name: gameId === 'super-mario-bros' ? '超级马里奥(PPO)' : '超级马里奥(DQN)'
      };
    });
    res.json({ success: true, games });
  } catch (error) {
    console.error('获取游戏列表失败:', error);
    // 如果Python脚本执行失败，返回模拟数据
    const games = [
      { id: 'super-mario-bros', name: '超级马里奥(PPO)' },
      { id: 'super-mario-dqn', name: '超级马里奥(DQN)' }
    ];
    res.json({ success: true, games });
  }
});

// 获取游戏关卡列表
app.get('/api/levels', async (req, res) => {
  const gameId = req.query.game;
  if (!gameId) {
    return res.status(400).json({ success: false, message: '缺少游戏ID参数' });
  }
  
  try {
    const result = await runPythonScript(['agent_integration.py', 'list-levels', '--game', gameId]);
    const levels = JSON.parse(result).map(levelId => {
      return { id: levelId, name: levelId.replace('SuperMarioBros-', '关卡') };
    });
    res.json({ success: true, levels });
  } catch (error) {
    console.error('获取关卡列表失败:', error);
    // 如果Python脚本执行失败，返回模拟数据
    const levels = [
      { id: 'SuperMarioBros-1-1-v0', name: '关卡1-1' },
      { id: 'SuperMarioBros-1-2-v0', name: '关卡1-2' },
      { id: 'SuperMarioBros-1-3-v0', name: '关卡1-3' }
    ];
    res.json({ success: true, levels });
  }
});

// 获取可用智能体列表
app.get('/api/agents', async (req, res) => {
  const gameId = req.query.game;
  if (!gameId) {
    return res.status(400).json({ success: false, message: '缺少游戏ID参数' });
  }
  
  try {
    const result = await runPythonScript(['agent_integration.py', 'list-agents', '--game', gameId]);
    const agents = JSON.parse(result).map(agentId => {
      return { 
        id: agentId, 
        name: agentId === 'ppo' ? 'PPO智能体' : 'DQN智能体',
        game: gameId
      };
    });
    res.json({ success: true, agents });
  } catch (error) {
    console.error('获取智能体列表失败:', error);
    // 如果Python脚本执行失败，返回模拟数据
    const agents = [
      { id: 'ppo', name: 'PPO智能体', game: gameId },
      { id: 'dqn', name: 'DQN智能体', game: gameId }
    ];
    res.json({ success: true, agents });
  }
});

// 执行训练命令
app.post('/api/train', (req, res) => {
  const { game, level, agent, config } = req.body;
  
  if (!game || !level || !agent) {
    return res.status(400).json({ success: false, message: '缺少必要参数' });
  }
  
  // 保存配置到临时文件
  let configPath = null;
  if (config) {
    configPath = path.join(configDir, `${game}_${level}_${agent}_config.json`);
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf8');
  }
  
  // 构建命令参数
  const args = ['agent_integration.py', 'train', '--game', game, '--level', level, '--agent', agent];
  if (configPath) {
    args.push('--config', configPath);
  }
  
  // 启动训练进程
  const trainProcess = spawn(PYTHON_BIN, args, {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  
  const processId = Date.now().toString();
  runningProcesses[processId] = {
    process: trainProcess,
    logs: [],
    type: 'train'
  };
  
  // 收集输出
  trainProcess.stdout.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(log);
    console.log(`[训练日志] ${log}`);
  });
  
  trainProcess.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    console.error(`[训练错误] ${log}`);
  });
  
  trainProcess.on('close', (code) => {
    console.log(`训练进程退出，退出码: ${code}`);
    runningProcesses[processId].logs.push(`[完成] 训练进程退出，退出码: ${code}`);
    runningProcesses[processId].completed = true;
  });
  
  res.json({ 
    success: true, 
    message: '训练已启动', 
    processId,
    game,
    level,
    agent
  });
});

// 获取训练日志
app.get('/api/logs/:processId', (req, res) => {
  const { processId } = req.params;
  const process = runningProcesses[processId];
  
  if (!process) {
    return res.status(404).json({ success: false, message: '进程不存在' });
  }
  
  res.json({ 
    success: true, 
    logs: process.logs,
    completed: !!process.completed
  });
});

// 停止训练
app.post('/api/stop/:processId', (req, res) => {
  const { processId } = req.params;
  const process = runningProcesses[processId];
  
  if (!process) {
    return res.status(404).json({ success: false, message: '进程不存在' });
  }
  
  try {
    process.process.kill();
    process.logs.push('[停止] 进程已手动终止');
    process.completed = true;
    
    res.json({ success: true, message: '进程已停止' });
  } catch (error) {
    res.status(500).json({ success: false, message: `停止进程失败: ${error.message}` });
  }
});

// 复制预训练模型
app.post('/api/copy-models', async (req, res) => {
  try {
    const result = await runPythonScript(['agent_integration.py', 'copy-models']);
    res.json({ success: true, message: '预训练模型已复制', result });
  } catch (error) {
    console.error('复制预训练模型失败:', error);
    res.status(500).json({ success: false, message: `复制预训练模型失败: ${error.message}` });
  }
});

// 同步预训练模型到本服务目录（最小可用：直接从上级目录复制）
app.post('/api/sync-models', (req, res) => {
  try {
    const sourceDir = path.join(__dirname, '..', 'pretrained_models');
    const destDir = path.join(__dirname, 'pretrained_models');
    if (!fs.existsSync(sourceDir)) {
      return res.status(404).json({ success: false, message: '源模型目录不存在: ' + sourceDir });
    }
    if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true });
    const files = fs.readdirSync(sourceDir).filter(f => f.endsWith('.dat') || f.endsWith('.pth'));
    files.forEach(f => {
      const src = path.join(sourceDir, f);
      const dst = path.join(destDir, f);
      fs.copyFileSync(src, dst);
    });
    res.json({ success: true, message: '模型已同步', files });
  } catch (error) {
    res.status(500).json({ success: false, message: '同步失败: ' + error.message });
  }
});

// 列举本地可用模型文件
app.get('/api/models', (req, res) => {
  try {
    const models = [];
    
    // 扫描新的实验目录结构
    const experimentsDir = path.join(__dirname, 'experiments');
    if (fs.existsSync(experimentsDir)) {
      const scanExperiments = (dir, prefix = '') => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        
        for (const entry of entries) {
          if (entry.isDirectory()) {
            const subDir = path.join(dir, entry.name);
            const newPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
            
            // 检查是否有weights目录
            const weightsDir = path.join(subDir, 'weights');
            if (fs.existsSync(weightsDir)) {
              const weightFiles = fs.readdirSync(weightsDir)
                .filter(f => f.endsWith('.pth') || f.endsWith('.dat'));
              
              weightFiles.forEach(file => {
                models.push({
                  path: `experiments/${newPrefix}/weights/${file}`,
                  name: `${newPrefix}/${file}`,
                  type: 'experiment',
                  experiment_path: `experiments/${newPrefix}`
                });
              });
            }
            
            // 递归扫描子目录
            scanExperiments(subDir, newPrefix);
          }
        }
      };
      
      scanExperiments(experimentsDir);
    }
    
    // 扫描旧的目录结构（兼容性）
    const marioDqnDir = path.join(__dirname, 'pretrained_models', 'mario', 'dqn');
    if (fs.existsSync(marioDqnDir)) {
      const files = fs.readdirSync(marioDqnDir).filter(f => f.endsWith('.dat') || f.endsWith('.pth'));
      files.forEach(file => {
        models.push({
          path: `pretrained_models/mario/dqn/${file}`,
          name: file,
          type: 'pretrained'
        });
      });
    }
    
    const oldDir = path.join(__dirname, 'pretrained_models');
    if (fs.existsSync(oldDir)) {
      const files = fs.readdirSync(oldDir).filter(f => f.endsWith('.dat') || f.endsWith('.pth'));
      files.forEach(file => {
        models.push({
          path: file,
          name: file,
          type: 'legacy'
        });
      });
    }
    
    // 按类型和名称排序
    models.sort((a, b) => {
      if (a.type !== b.type) {
        const typeOrder = { 'experiment': 0, 'pretrained': 1, 'legacy': 2 };
        return typeOrder[a.type] - typeOrder[b.type];
      }
      return a.name.localeCompare(b.name);
    });
    
    res.json({ success: true, models });
  } catch (error) {
    res.status(500).json({ success: false, message: '读取失败: ' + error.message });
  }
});

// 获取实验列表
app.get('/api/experiments', (req, res) => {
  try {
    const experimentsDir = path.join(__dirname, 'experiments');
    const experiments = [];
    
    if (!fs.existsSync(experimentsDir)) {
      return res.json({ success: true, experiments: [] });
    }
    
    const scanExperiments = (dir, prefix = '') => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        if (entry.isDirectory()) {
          const subDir = path.join(dir, entry.name);
          const newPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
          
          // 检查是否有metadata.json文件
          const metadataFile = path.join(subDir, 'metadata.json');
          if (fs.existsSync(metadataFile)) {
            try {
              const metadata = JSON.parse(fs.readFileSync(metadataFile, 'utf8'));
              const stats = fs.statSync(subDir);
              
              experiments.push({
                path: `experiments/${newPrefix}`,
                name: newPrefix,
                metadata: metadata,
                created_at: stats.birthtime,
                modified_at: stats.mtime,
                has_metrics: fs.existsSync(path.join(subDir, 'metrics.csv')),
                has_weights: fs.existsSync(path.join(subDir, 'weights')) && 
                           fs.readdirSync(path.join(subDir, 'weights')).length > 0
              });
            } catch (e) {
              console.warn(`无法读取元数据文件: ${metadataFile}`, e);
            }
          }
          
          // 递归扫描子目录
          scanExperiments(subDir, newPrefix);
        }
      }
    };
    
    scanExperiments(experimentsDir);
    
    // 按创建时间排序
    experiments.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
    res.json({ success: true, experiments });
  } catch (error) {
    console.error('获取实验列表失败:', error);
    res.status(500).json({ success: false, message: '获取实验列表失败' });
  }
});

// 启动 DQN 评估（最小可用）：调用本项目内 test.py 并回传日志
app.post('/api/start-dqn', (req, res) => {
  const { level, weights } = req.body;
  if (!level || !weights) {
    return res.status(400).json({ success: false, message: '缺少必要参数 level 或 weights' });
  }
  const scriptPath = path.join(__dirname, 'test.py');
  const weightsArg = path.isAbsolute(weights) ? weights : weights; // 让 test.py 自行解析到 ./pretrained_models
  const args = [scriptPath, '--env', level, '--weights', weightsArg, '--render', 'human'];

  const evalProcess = spawn(PYTHON_BIN, args, {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  const processId = Date.now().toString();
  runningProcesses[processId] = {
    process: evalProcess,
    logs: [],
    type: 'dqn-eval'
  };

  evalProcess.stdout.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(log);
    console.log(`[DQN日志] ${log}`);
  });

  evalProcess.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    console.error(`[DQN错误] ${log}`);
  });

  evalProcess.on('close', (code) => {
    console.log(`DQN进程退出，退出码: ${code}`);
    runningProcesses[processId].logs.push(`[完成] DQN进程退出，退出码: ${code}`);
    runningProcesses[processId].completed = true;
  });

  res.json({ success: true, message: 'DQN 启动', processId, level, weights: path.basename(weightsArg) });
});

// 启动帧流：调用 render_stream.py，输出包含 base64 JPEG 的 JSON 行
app.post('/api/start-stream', (req, res) => {
  const { level, weights, fps } = req.body;
  if (!level) {
    return res.status(400).json({ success: false, message: '缺少必要参数 level' });
  }
  // 根据是否提供权重，选择纯渲染或推理渲染
  const scriptPath = weights ? path.join(__dirname, 'python/scripts/agent_inference.py') : path.join(__dirname, 'render_stream.py');
  const args = [scriptPath, '--env', level];
  if (weights) args.push('--weights', weights);
  if (fps) args.push('--fps', String(fps));

  const streamProc = spawn(PYTHON_BIN, args, {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  const processId = Date.now().toString();
  runningProcesses[processId] = {
    process: streamProc,
    logs: [],
    frames: [],
    lastMeta: null,
    type: 'frame-stream'
  };

  streamProc.stdout.on('data', (data) => {
    const line = data.toString();
    // 逐行处理，尝试解析 JSON
    line.split('\n').forEach((ln) => {
      if (!ln.trim()) return;
      try {
        const obj = JSON.parse(ln);
        if (obj.type === 'frame' && obj.frame) {
          runningProcesses[processId].frames = [obj.frame]; // 仅保留最新一帧
          runningProcesses[processId].lastMeta = { agent_action: obj.agent_action, reward: obj.reward, t: obj.t };
          // 同时也把原始 JSON 压入日志，便于前端解析
          runningProcesses[processId].logs.push(JSON.stringify(obj));
          
          // 通过WebSocket广播帧数据
          broadcastFrame(processId, obj);
        } else {
          runningProcesses[processId].logs.push(ln);
        }
      } catch (e) {
        runningProcesses[processId].logs.push(ln);
      }
    });
  });

  streamProc.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    // 添加这一行，让错误信息显示在服务器控制台中
    console.error(`[Stream Error] processId ${processId}: ${log}`);
  });

  streamProc.on('close', (code) => {
    runningProcesses[processId].logs.push(`[完成] 帧流退出，退出码: ${code}`);
    runningProcesses[processId].completed = true;
    // 添加这一行，让进程退出信息也显示在控制台中
    console.log(`[Stream Process] processId ${processId} 退出，退出码: ${code}`);
  });

  res.json({ success: true, processId });
});

// 启动通用训练
app.post('/api/start-training', (req, res) => {
  const { 
    algorithm = 'DQN',
    game = 'mario',
    environment = 'SuperMarioBros-1-1-v0',
    action_space = 'complex',
    episodes = 1000,
    ...algorithmParams
  } = req.body;

  const processId = Date.now().toString();
  
  // 根据算法选择对应的训练脚本
  const algorithmScripts = {
    'DQN': 'python/scripts/train_dqn.py',
    'PPO': 'python/scripts/train_ppo.py',
    'A2C': 'python/scripts/train_a2c.py'
  };
  
  const scriptPath = algorithmScripts[algorithm];
  if (!scriptPath) {
    return res.status(400).json({ 
      success: false, 
      message: `不支持的算法: ${algorithm}` 
    });
  }
  
  // 构建训练命令参数
  const args = [
    path.join(__dirname, scriptPath),
    '--environment', environment,
    '--action-space', action_space,
    '--episodes', String(episodes)
  ];

  // 参数名映射：将下划线格式转换为连字符格式
  const paramMapping = {
    'learning_rate': 'learning-rate',
    'epsilon_start': 'epsilon-start',
    'epsilon_final': 'epsilon-final',
    'epsilon_decay': 'epsilon-decay',
    'batch_size': 'batch-size',
    'memory_capacity': 'memory-capacity',
    'target_update_frequency': 'target-update-frequency',
    'initial_learning': 'initial-learning',
    'beta_start': 'beta-start',
    'beta_frames': 'beta-frames',
    'action_space': 'action-space',
    'save_frequency': 'save-frequency',
    'log_frequency': 'log-frequency',
    'save_model': 'save-model',
    'use_gpu': 'force-cpu' // 注意：use_gpu=true时应该不传force-cpu参数
    // 注意：max_steps_per_episode 和 verbose 参数在 train_dqn.py 中不支持，已移除
  };

  // 添加算法特定参数
  Object.entries(algorithmParams).forEach(([key, value]) => {
    // 跳过不支持的参数
    if (key === 'max_steps_per_episode' || key === 'verbose') {
      return;
    }
    
    // 使用映射后的参数名
    const mappedKey = paramMapping[key] || key;
    
    if (typeof value === 'boolean') {
      if (value) {
        // 特殊处理use_gpu参数
        if (key === 'use_gpu' && value) {
          // use_gpu=true时不添加任何参数（默认使用GPU）
          return;
        }
        args.push(`--${mappedKey}`);
      }
    } else {
      args.push(`--${mappedKey}`, String(value));
    }
  });

  console.log(`[${algorithm}训练] 启动训练进程 ${processId}:`, args.join(' '));

  const trainProcess = spawn(PYTHON_BIN, args, {
    cwd: path.join(__dirname),
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  runningProcesses[processId] = {
    process: trainProcess,
    logs: [],
    metrics: [],
    completed: false,
    type: 'training',
    algorithm,
    game,
    config: req.body
  };

  // 处理训练输出
  trainProcess.stdout.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(log);
    console.log(`[${algorithm}训练] ${log}`);
    
    // 解析训练指标
    const lines = log.split('\n');
    lines.forEach(line => {
      if (line.includes('Episode') && line.includes('Reward')) {
        try {
          // 解析类似 "Episode 100 - Reward: 150.5, Best: 200.0, Average: 120.3" 的日志
          const episodeMatch = line.match(/Episode (\d+)/);
          const rewardMatch = line.match(/Reward: ([\d.-]+)/);
          const bestMatch = line.match(/Best: ([\d.-]+)/);
          const avgMatch = line.match(/Average: ([\d.-]+)/);
          
          if (episodeMatch && rewardMatch) {
            const metric = {
              episode: parseInt(episodeMatch[1]),
              reward: parseFloat(rewardMatch[1]),
              best: bestMatch ? parseFloat(bestMatch[1]) : 0,
              average: avgMatch ? parseFloat(avgMatch[1]) : 0,
              timestamp: Date.now()
            };
            runningProcesses[processId].metrics.push(metric);
            
            // 通过WebSocket广播训练指标
            broadcastTrainingMetrics(processId, metric);
          }
        } catch (e) {
          // 忽略解析错误
        }
      }
    });
  });

  trainProcess.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    console.error(`[${algorithm}训练错误] ${log}`);
  });

  trainProcess.on('close', (code) => {
    console.log(`${algorithm}训练进程退出，退出码: ${code}`);
    runningProcesses[processId].logs.push(`[完成] ${algorithm}训练完成，退出码: ${code}`);
    runningProcesses[processId].completed = true;
    
    // 通过WebSocket广播训练完成
    broadcastTrainingComplete(processId, code);
  });

  res.json({ 
    success: true, 
    message: `${algorithm}训练已启动`, 
    processId,
    algorithm,
    game,
    config: req.body
  });
});

// 获取算法配置
app.get('/api/algorithm-configs', (req, res) => {
  try {
    // 直接读取Python配置文件
    const fs = require('fs');
    const configPath = path.join(__dirname, 'python/configs/algorithm_configs.py');
    
    if (!fs.existsSync(configPath)) {
      return res.status(404).json({ success: false, message: '配置文件不存在' });
    }
    
    // 读取配置文件内容
    const configContent = fs.readFileSync(configPath, 'utf8');
    
    // 提取配置数据（简单解析）
    const algorithmsMatch = configContent.match(/ALGORITHM_CONFIGS\s*=\s*({[\s\S]*?^})/m);
    const gamesMatch = configContent.match(/GAME_ENVIRONMENTS\s*=\s*({[\s\S]*?^})/m);
    
    if (!algorithmsMatch || !gamesMatch) {
      return res.status(500).json({ success: false, message: '配置文件格式错误' });
    }
    
    // 返回硬编码的配置（避免Python执行问题）
    const configs = {
      algorithms: {
        'DQN': {
          name: 'Deep Q-Network',
          script: 'python/scripts/train_dqn.py',
          description: '基于深度Q网络的强化学习算法',
          parameters: {
            'learning_rate': {
              type: 'float',
              default: 1e-4,
              min: 1e-6,
              max: 1e-2,
              step: 1e-5,
              precision: 6,
              description: '学习率'
            },
            'gamma': {
              type: 'float',
              default: 0.99,
              min: 0.1,
              max: 1.0,
              step: 0.01,
              precision: 2,
              description: '折扣因子'
            },
            'epsilon_start': {
              type: 'float',
              default: 1.0,
              min: 0.1,
              max: 1.0,
              step: 0.1,
              precision: 1,
              description: '初始探索率'
            },
            'epsilon_final': {
              type: 'float',
              default: 0.01,
              min: 0.001,
              max: 0.1,
              step: 0.001,
              precision: 3,
              description: '最终探索率'
            },
            'epsilon_decay': {
              type: 'int',
              default: 100000,
              min: 10000,
              max: 1000000,
              step: 10000,
              description: '探索率衰减步数'
            },
            'batch_size': {
              type: 'int',
              default: 32,
              min: 4,
              max: 128,
              step: 4,
              description: '批次大小'
            },
            'memory_capacity': {
              type: 'int',
              default: 20000,
              min: 1000,
              max: 100000,
              step: 1000,
              description: '经验回放缓冲区大小'
            },
            'target_update_frequency': {
              type: 'int',
              default: 1000,
              min: 100,
              max: 10000,
              step: 100,
              description: '目标网络更新频率'
            },
            'initial_learning': {
              type: 'int',
              default: 10000,
              min: 1000,
              max: 50000,
              step: 1000,
              description: '初始学习步数'
            },
            'beta_start': {
              type: 'float',
              default: 0.4,
              min: 0.1,
              max: 1.0,
              step: 0.1,
              precision: 1,
              description: '初始Beta值'
            },
            'beta_frames': {
              type: 'int',
              default: 10000,
              min: 1000,
              max: 100000,
              step: 1000,
              description: 'Beta更新帧数'
            }
          },
          flags: {
            'render': {
              type: 'boolean',
              default: false,
              description: '启用渲染'
            },
            'save_model': {
              type: 'boolean',
              default: true,
              description: '保存模型'
            }
          }
        },
        'PPO': {
          name: 'Proximal Policy Optimization',
          script: 'python/scripts/train_ppo.py',
          description: '近端策略优化算法',
          parameters: {
            'learning_rate': {
              type: 'float',
              default: 3e-4,
              min: 1e-6,
              max: 1e-2,
              step: 1e-5,
              precision: 6,
              description: '学习率'
            },
            'gamma': {
              type: 'float',
              default: 0.99,
              min: 0.1,
              max: 1.0,
              step: 0.01,
              precision: 2,
              description: '折扣因子'
            },
            'clip_ratio': {
              type: 'float',
              default: 0.2,
              min: 0.1,
              max: 0.5,
              step: 0.01,
              precision: 2,
              description: '裁剪比例'
            },
            'value_loss_coef': {
              type: 'float',
              default: 0.5,
              min: 0.1,
              max: 1.0,
              step: 0.1,
              precision: 1,
              description: '价值损失系数'
            },
            'entropy_coef': {
              type: 'float',
              default: 0.01,
              min: 0.001,
              max: 0.1,
              step: 0.001,
              precision: 3,
              description: '熵系数'
            },
            'max_grad_norm': {
              type: 'float',
              default: 0.5,
              min: 0.1,
              max: 2.0,
              step: 0.1,
              precision: 1,
              description: '最大梯度范数'
            },
            'ppo_epochs': {
              type: 'int',
              default: 4,
              min: 1,
              max: 20,
              step: 1,
              description: 'PPO更新轮数'
            },
            'batch_size': {
              type: 'int',
              default: 64,
              min: 16,
              max: 256,
              step: 16,
              description: '批次大小'
            }
          },
          flags: {
            'render': {
              type: 'boolean',
              default: false,
              description: '启用渲染'
            },
            'save_model': {
              type: 'boolean',
              default: true,
              description: '保存模型'
            }
          }
        }
      },
      games: {
        'mario': {
          name: 'Super Mario Bros',
          environments: [
            'SuperMarioBros-1-1-v0',
            'SuperMarioBros-1-2-v0',
            'SuperMarioBros-1-3-v0',
            'SuperMarioBros-1-4-v0',
            'SuperMarioBros-2-1-v0',
            'SuperMarioBros-2-2-v0',
            'SuperMarioBros-2-3-v0',
            'SuperMarioBros-2-4-v0',
            'SuperMarioBros-3-1-v0',
            'SuperMarioBros-3-2-v0',
            'SuperMarioBros-3-3-v0'
          ],
          action_spaces: {
            'simple': 'SIMPLE_MOVEMENT',
            'complex': 'COMPLEX_MOVEMENT',
            'right_only': 'RIGHT_ONLY'
          }
        },
        'atari': {
          name: 'Atari Games',
          environments: [
            'Breakout-v4',
            'Pong-v4',
            'SpaceInvaders-v4',
            'MsPacman-v4',
            'Qbert-v4',
            'Seaquest-v4'
          ],
          action_spaces: {
            'discrete': 'DISCRETE',
            'minimal': 'MINIMAL'
          }
        }
      }
    };
    
    res.json({ success: true, ...configs });
  } catch (error) {
    console.error('获取算法配置失败:', error);
    res.status(500).json({ success: false, message: '获取算法配置失败' });
  }
});

// 获取算法参数文件内容
app.get('/api/algorithm-config-file/:algorithm', (req, res) => {
  try {
    const { algorithm } = req.params;
    const fs = require('fs');
    
    // 根据算法类型确定配置文件路径
    let configPath;
    switch (algorithm.toLowerCase()) {
      case 'dqn':
        configPath = path.join(__dirname, 'python/algorithms/dqn/core/constants.py');
        break;
      case 'ppo':
        configPath = path.join(__dirname, 'python/algorithms/ppo/core/constants.py');
        break;
      case 'a2c':
        configPath = path.join(__dirname, 'python/algorithms/a2c/core/constants.py');
        break;
      default:
        return res.status(404).json({ success: false, message: '不支持的算法类型' });
    }
    
    if (!fs.existsSync(configPath)) {
      return res.status(404).json({ success: false, message: '配置文件不存在' });
    }
    
    const content = fs.readFileSync(configPath, 'utf8');
    res.json({ 
      success: true, 
      content: content,
      algorithm: algorithm,
      path: configPath
    });
  } catch (error) {
    console.error('读取算法配置文件失败:', error);
    res.status(500).json({ success: false, message: '读取配置文件失败' });
  }
});

// 保存算法参数文件
app.post('/api/save-algorithm-config-file', (req, res) => {
  try {
    const { algorithm, content } = req.body;
    const fs = require('fs');
    
    if (!algorithm || !content) {
      return res.status(400).json({ success: false, message: '缺少必要参数' });
    }
    
    // 根据算法类型确定配置文件路径
    let configPath;
    switch (algorithm.toLowerCase()) {
      case 'dqn':
        configPath = path.join(__dirname, 'python/algorithms/dqn/core/constants.py');
        break;
      case 'ppo':
        configPath = path.join(__dirname, 'python/algorithms/ppo/core/constants.py');
        break;
      case 'a2c':
        configPath = path.join(__dirname, 'python/algorithms/a2c/core/constants.py');
        break;
      default:
        return res.status(404).json({ success: false, message: '不支持的算法类型' });
    }
    
    // 备份原文件
    const backupPath = configPath + '.backup.' + Date.now();
    if (fs.existsSync(configPath)) {
      fs.copyFileSync(configPath, backupPath);
    }
    
    // 保存新内容
    fs.writeFileSync(configPath, content, 'utf8');
    
    res.json({ 
      success: true, 
      message: '配置文件保存成功',
      backupPath: backupPath
    });
  } catch (error) {
    console.error('保存算法配置文件失败:', error);
    res.status(500).json({ success: false, message: '保存配置文件失败' });
  }
});

// 获取算法训练脚本内容
app.get('/api/algorithm-script/:algorithm', (req, res) => {
  try {
    const { algorithm } = req.params;
    const fs = require('fs');
    
    // 根据算法类型确定脚本路径
    let scriptPath;
    switch (algorithm.toLowerCase()) {
      case 'dqn':
        scriptPath = path.join(__dirname, 'python/scripts/train_dqn.py');
        break;
      case 'ppo':
        scriptPath = path.join(__dirname, 'python/scripts/train_ppo.py');
        break;
      case 'a2c':
        scriptPath = path.join(__dirname, 'python/scripts/train_a2c.py');
        break;
      default:
        return res.status(404).json({ success: false, message: '不支持的算法类型' });
    }
    
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ success: false, message: '训练脚本不存在' });
    }
    
    const content = fs.readFileSync(scriptPath, 'utf8');
    res.json({ 
      success: true, 
      content: content,
      algorithm: algorithm,
      path: scriptPath
    });
  } catch (error) {
    console.error('读取算法训练脚本失败:', error);
    res.status(500).json({ success: false, message: '读取训练脚本失败' });
  }
});

// 启动DQN训练
app.post('/api/start-dqn-training', (req, res) => {
  const { 
    environment = 'SuperMarioBros-1-1-v0',
    action_space = 'complex',
    num_episodes = 1000,
    learning_rate = 1e-4,
    gamma = 0.99,
    epsilon_start = 1.0,
    epsilon_final = 0.01,
    epsilon_decay = 100000,
    batch_size = 32,
    buffer_capacity = 20000,
    target_update_frequency = 1000,
    initial_learning = 10000,
    beta_start = 0.4,
    beta_frames = 10000,
    render = false,
    transfer = false,
    force_cpu = false
  } = req.body;

  const processId = Date.now().toString();
  
  // 构建训练命令参数
  const args = [
    path.join(__dirname, 'super-mario-bros-dqn', 'train.py'),
    '--environment', environment,
    '--action-space', action_space,
    '--num-episodes', String(num_episodes),
    '--learning-rate', String(learning_rate),
    '--gamma', String(gamma),
    '--epsilon-start', String(epsilon_start),
    '--epsilon-final', String(epsilon_final),
    '--epsilon-decay', String(epsilon_decay),
    '--batch-size', String(batch_size),
    '--buffer-capacity', String(buffer_capacity),
    '--target-update-frequency', String(target_update_frequency),
    '--initial-learning', String(initial_learning),
    '--beta-start', String(beta_start),
    '--beta-frames', String(beta_frames)
  ];

  if (render) args.push('--render');
  if (transfer) args.push('--transfer');
  if (force_cpu) args.push('--force-cpu');

  console.log(`[DQN训练] 启动训练进程 ${processId}:`, args.join(' '));

  const trainProcess = spawn(PYTHON_BIN, args, {
    cwd: path.join(__dirname, 'super-mario-bros-dqn'),
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  runningProcesses[processId] = {
    process: trainProcess,
    logs: [],
    metrics: [],
    completed: false,
    type: 'dqn-training',
    config: req.body
  };

  // 处理训练输出
  trainProcess.stdout.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(log);
    console.log(`[DQN训练] ${log}`);
    
    // 解析训练指标
    const lines = log.split('\n');
    lines.forEach(line => {
      if (line.includes('Episode') && line.includes('Reward')) {
        try {
          // 解析类似 "Episode 100 - Reward: 150.5, Best: 200.0, Average: 120.3" 的日志
          const episodeMatch = line.match(/Episode (\d+)/);
          const rewardMatch = line.match(/Reward: ([\d.-]+)/);
          const bestMatch = line.match(/Best: ([\d.-]+)/);
          const avgMatch = line.match(/Average: ([\d.-]+)/);
          
          if (episodeMatch && rewardMatch) {
            const metric = {
              episode: parseInt(episodeMatch[1]),
              reward: parseFloat(rewardMatch[1]),
              best: bestMatch ? parseFloat(bestMatch[1]) : 0,
              average: avgMatch ? parseFloat(avgMatch[1]) : 0,
              timestamp: Date.now()
            };
            runningProcesses[processId].metrics.push(metric);
            
            // 通过WebSocket广播训练指标
            broadcastTrainingMetrics(processId, metric);
          }
        } catch (e) {
          // 忽略解析错误
        }
      }
    });
  });

  trainProcess.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    console.error(`[DQN训练错误] ${log}`);
  });

  trainProcess.on('close', (code) => {
    console.log(`DQN训练进程退出，退出码: ${code}`);
    runningProcesses[processId].logs.push(`[完成] DQN训练完成，退出码: ${code}`);
    runningProcesses[processId].completed = true;
    
    // 通过WebSocket广播训练完成
    broadcastTrainingComplete(processId, code);
  });

  res.json({ 
    success: true, 
    message: 'DQN训练已启动', 
    processId,
    config: req.body
  });
});

// 启动玩家控制
app.post('/api/start-player', (req, res) => {
  const { level, actionSpace, fps } = req.body;
  if (!level) {
    return res.status(400).json({ success: false, message: '缺少必要参数 level' });
  }
  
  // 停止旧的玩家进程
  Object.keys(runningProcesses).forEach(pid => {
    const proc = runningProcesses[pid];
    if (proc.type === 'player-control') {
      try {
        console.log(`[停止旧玩家进程] ${pid}`);
        if (proc.process && typeof proc.process.kill === 'function') {
          proc.process.kill();
        }
        delete runningProcesses[pid];
      } catch (error) {
        console.log(`[停止旧玩家进程错误] ${pid}: ${error.message}`);
      }
    }
  });
  
  const scriptPath = path.join(__dirname, 'python/scripts/player_control.py');
  const args = [scriptPath, '--env', level];
  if (actionSpace) args.push('--action-space', actionSpace);
  if (fps) args.push('--fps', String(fps));

  const playerProc = spawn(PYTHON_BIN, args, {
    stdio: ['pipe', 'pipe', 'pipe'] // 需要stdin来接收键盘输入
  });
  
  // 检查进程是否成功创建
  if (!playerProc) {
    return res.status(500).json({ success: false, message: '无法创建玩家进程' });
  }
  
  const processId = Date.now().toString();
  runningProcesses[processId] = {
    process: playerProc,
    logs: [],
    frames: [],
    lastMeta: null,
    type: 'player-control'
  };

  playerProc.stdout.on('data', (data) => {
    const line = data.toString();
    // 逐行处理，尝试解析 JSON
    line.split('\n').forEach((ln) => {
      if (!ln.trim()) return;
      try {
        const obj = JSON.parse(ln);
        if (obj.type === 'player_frame' && obj.frame) {
          runningProcesses[processId].frames = [obj.frame]; // 仅保留最新一帧
          runningProcesses[processId].lastMeta = { agent_action: obj.agent_action, reward: obj.reward, t: obj.t };
          // 同时也把原始 JSON 压入日志，便于前端解析
          runningProcesses[processId].logs.push(JSON.stringify(obj));
          
          // 通过WebSocket广播帧数据
          broadcastFrame(processId, obj);
        } else {
          runningProcesses[processId].logs.push(ln);
        }
      } catch (e) {
        runningProcesses[processId].logs.push(ln);
      }
    });
  });

  playerProc.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    console.error(`[Player Error] processId ${processId}: ${log}`);
  });

  playerProc.on('close', (code) => {
    runningProcesses[processId].logs.push(`[完成] 玩家控制退出，退出码: ${code}`);
    runningProcesses[processId].completed = true;
    console.log(`[Player Process] processId ${processId} 退出，退出码: ${code}`);
  });

  playerProc.on('error', (error) => {
    console.error(`[Player Process Error] processId ${processId}: ${error.message}`);
    runningProcesses[processId].logs.push(`[错误] 进程启动失败: ${error.message}`);
    runningProcesses[processId].completed = true;
  });

  res.json({ success: true, processId });
});

// 发送玩家动作
app.post('/api/player-action/:processId', (req, res) => {
  const { processId } = req.params;
  const { action } = req.body;
  
  const proc = runningProcesses[processId];
  if (!proc) {
    return res.status(404).json({ success: false, message: '进程不存在' });
  }
  
  if (proc.type !== 'player-control') {
    return res.status(400).json({ success: false, message: '不是玩家控制进程' });
  }
  
  try {
    // 发送动作到Python进程的stdin
    proc.process.stdin.write(JSON.stringify({ action }) + '\n');
    res.json({ success: true, message: '动作已发送' });
  } catch (error) {
    res.status(500).json({ success: false, message: `发送动作失败: ${error.message}` });
  }
});

// 获取玩家帧
app.get('/api/player-frame/:processId', (req, res) => {
  const { processId } = req.params;
  const proc = runningProcesses[processId];
  if (!proc) {
    console.log(`[API] 玩家进程 ${processId} 不存在，可能已退出`);
    return res.status(404).json({ 
      success: false, 
      message: '进程不存在或已退出',
      completed: true 
    });
  }
  const frame = (proc.frames && proc.frames[0]) || null;
  res.json({ success: true, frame, meta: proc.lastMeta || null, completed: !!proc.completed });
});

// 获取最新一帧（base64 JPEG）
app.get('/api/frame/:processId', (req, res) => {
  const { processId } = req.params;
  const proc = runningProcesses[processId];
  if (!proc) {
    console.log(`[API] 智能体进程 ${processId} 不存在，可能已退出`);
    return res.status(404).json({ 
      success: false, 
      message: '进程不存在或已退出',
      completed: true 
    });
  }
  const frame = (proc.frames && proc.frames[0]) || null;
  res.json({ success: true, frame, meta: proc.lastMeta || null, completed: !!proc.completed });
});

// 设置游戏显示
app.post('/api/setup-display', async (req, res) => {
  const { game, level, agent } = req.body;
  
  if (!game || !level || !agent) {
    return res.status(400).json({ success: false, message: '缺少必要参数' });
  }
  
  try {
    const result = await runPythonScript(['agent_integration.py', 'setup-display', '--game', game, '--level', level, '--agent', agent]);
    res.json({ success: true, message: '游戏显示已设置', result });
  } catch (error) {
    console.error('设置游戏显示失败:', error);
    res.status(500).json({ success: false, message: `设置游戏显示失败: ${error.message}` });
  }
});

// 启动游戏
app.post('/api/start-game', async (req, res) => {
  const { game, level, agent } = req.body;
  
  if (!game || !level || !agent) {
    return res.status(400).json({ success: false, message: '缺少必要参数' });
  }
  
  // 构建命令参数
  const args = ['agent_integration.py', 'eval', '--game', game, '--level', level, '--agent', agent];
  
  // 启动评估进程
  const evalProcess = spawn(PYTHON_BIN, args, {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  
  const processId = Date.now().toString();
  runningProcesses[processId] = {
    process: evalProcess,
    logs: [],
    type: 'eval'
  };
  
  // 收集输出
  evalProcess.stdout.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(log);
    console.log(`[评估日志] ${log}`);
    
    // 尝试解析JSON输出并广播
    try {
      const lines = log.split('\n').filter(line => line.trim());
      for (const line of lines) {
        const jsonData = JSON.parse(line);
        if (jsonData.type === 'frame' || jsonData.type === 'player_frame') {
          broadcastFrame(processId, jsonData);
        }
      }
    } catch (e) {
      // 不是JSON数据，忽略
    }
  });
  
  evalProcess.stderr.on('data', (data) => {
    const log = data.toString();
    runningProcesses[processId].logs.push(`[错误] ${log}`);
    console.error(`[评估错误] ${log}`);
  });
  
  evalProcess.on('close', (code) => {
    console.log(`评估进程退出，退出码: ${code}`);
    runningProcesses[processId].logs.push(`[完成] 评估进程退出，退出码: ${code}`);
    runningProcesses[processId].completed = true;
  });
  
  res.json({ 
    success: true, 
    message: '游戏已启动', 
    processId,
    game,
    level,
    agent,
    gameUrl: '/game-screen', // 游戏画面URL
    agentUrl: '/agent-screen' // 智能体画面URL
  });
});

// WebSocket连接处理
wss.on('connection', (ws, req) => {
  const clientId = Date.now().toString();
  wsConnections.set(clientId, ws);
  console.log(`[WebSocket] 客户端 ${clientId} 已连接`);
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      console.log(`[WebSocket] 收到消息:`, data);
      
      // 处理不同类型的消息
      switch (data.type) {
        case 'player_action':
          handlePlayerAction(data.processId, data.action);
          break;
        case 'ping':
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
          break;
        default:
          console.log(`[WebSocket] 未知消息类型: ${data.type}`);
      }
    } catch (error) {
      console.error('[WebSocket] 消息解析错误:', error);
    }
  });
  
  ws.on('close', () => {
    wsConnections.delete(clientId);
    console.log(`[WebSocket] 客户端 ${clientId} 已断开`);
  });
  
  ws.on('error', (error) => {
    console.error(`[WebSocket] 客户端 ${clientId} 错误:`, error);
    wsConnections.delete(clientId);
  });
});

// 处理玩家动作
function handlePlayerAction(processId, action) {
  const proc = runningProcesses[processId];
  if (proc && proc.process && !proc.completed) {
    try {
      proc.process.stdin.write(JSON.stringify({ action }) + '\n');
      console.log(`[WebSocket] 发送玩家动作: ${action} 到进程 ${processId}`);
    } catch (error) {
      console.error(`[WebSocket] 发送玩家动作失败:`, error);
    }
  }
}

// 广播帧数据到所有连接的客户端
function broadcastFrame(processId, frameData) {
  const message = JSON.stringify({
    processId,
    ...frameData
  });
  
  wsConnections.forEach((ws, clientId) => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(message);
      } catch (error) {
        console.error(`[WebSocket] 发送到客户端 ${clientId} 失败:`, error);
        wsConnections.delete(clientId);
      }
    }
  });
}

// 广播训练指标到所有连接的客户端
function broadcastTrainingMetrics(processId, metrics) {
  const message = JSON.stringify({
    type: 'training_metrics',
    processId,
    metrics
  });
  
  wsConnections.forEach((ws, clientId) => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(message);
      } catch (error) {
        console.error(`[WebSocket] 发送训练指标到客户端 ${clientId} 失败:`, error);
        wsConnections.delete(clientId);
      }
    }
  });
}

// 广播训练完成到所有连接的客户端
function broadcastTrainingComplete(processId, exitCode) {
  const message = JSON.stringify({
    type: 'training_complete',
    processId,
    exitCode,
    completed: true
  });
  
  wsConnections.forEach((ws, clientId) => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(message);
      } catch (error) {
        console.error(`[WebSocket] 发送训练完成到客户端 ${clientId} 失败:`, error);
        wsConnections.delete(clientId);
      }
    }
  });
}

// 获取训练指标数据 - 从CSV文件读取
app.get('/api/training-metrics/:environment', (req, res) => {
  const { environment } = req.params
  
  try {
    let csvPath = null
    
    // 首先尝试新的实验目录结构
    const experimentsDir = path.join(__dirname, 'experiments')
    if (fs.existsSync(experimentsDir)) {
      const envDir = path.join(experimentsDir, environment)
      if (fs.existsSync(envDir)) {
        // 查找所有算法目录
        const algorithmDirs = fs.readdirSync(envDir, { withFileTypes: true })
          .filter(e => e.isDirectory())
          .map(e => e.name)
        
        for (const algorithmDir of algorithmDirs) {
          const algorithmPath = path.join(envDir, algorithmDir)
          const sessionDirs = fs.readdirSync(algorithmPath, { withFileTypes: true })
            .filter(e => e.isDirectory())
            .map(e => e.name)
            .sort()
            .reverse() // 最新的在前面
          
          if (sessionDirs.length > 0) {
            const latestSession = sessionDirs[0]
            const metricsFile = path.join(algorithmPath, latestSession, 'metrics.csv')
            if (fs.existsSync(metricsFile)) {
              csvPath = metricsFile
              break // 找到第一个就退出
            }
          }
        }
      }
    }
    
    // 如果没找到，尝试旧的目录结构
    if (!csvPath) {
      const oldPath = path.join(__dirname, 'training_metrics', `${environment}.csv`)
      if (fs.existsSync(oldPath)) {
        csvPath = oldPath
      }
    }
    
    if (!csvPath || !fs.existsSync(csvPath)) {
      return res.json({ success: true, metrics: [] })
    }
    
    const csvContent = fs.readFileSync(csvPath, 'utf8')
    const lines = csvContent.trim().split('\n')
    
    if (lines.length <= 1) {
      return res.json({ success: true, metrics: [] })
    }
    
    // 解析CSV头部
    const headers = lines[0].split(',').map(h => h.trim())
    const metrics = []
    
    // 解析数据行
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim())
      if (values.length === headers.length) {
        const metric = {}
        headers.forEach((header, index) => {
          const value = values[index]
          // 尝试转换为数字
          if (!isNaN(value) && value !== '') {
            metric[header] = parseFloat(value)
          } else {
            metric[header] = value
          }
        })
        metrics.push(metric)
      }
    }
    
    res.json({ success: true, metrics })
  } catch (error) {
    console.error('读取训练指标CSV文件失败:', error)
    res.status(500).json({ success: false, message: '读取训练指标失败' })
  }
})

// 获取可用的训练指标文件列表
app.get('/api/training-metrics-files', (req, res) => {
  const experimentsDir = path.join(__dirname, 'experiments')
  const oldMetricsDir = path.join(__dirname, 'training_metrics')
  
  try {
    const files = []
    
    // 扫描新的实验目录结构
    if (fs.existsSync(experimentsDir)) {
      const scanExperiments = (dir, prefix = '') => {
        const entries = fs.readdirSync(dir, { withFileTypes: true })
        
        for (const entry of entries) {
          if (entry.isDirectory()) {
            const subDir = path.join(dir, entry.name)
            const newPrefix = prefix ? `${prefix}/${entry.name}` : entry.name
            
            // 检查是否有metrics.csv文件
            const metricsFile = path.join(subDir, 'metrics.csv')
            if (fs.existsSync(metricsFile)) {
              const stats = fs.statSync(metricsFile)
              files.push({
                name: newPrefix,
                path: `experiments/${newPrefix}/metrics.csv`,
                size: stats.size,
                modified: stats.mtime,
                type: 'experiment'
              })
            }
            
            // 递归扫描子目录
            scanExperiments(subDir, newPrefix)
          }
        }
      }
      
      scanExperiments(experimentsDir)
    }
    
    // 扫描旧的training_metrics目录（兼容性）
    if (fs.existsSync(oldMetricsDir)) {
      const oldFiles = fs.readdirSync(oldMetricsDir)
        .filter(file => file.endsWith('.csv'))
        .map(file => ({
          name: file.replace('.csv', ''),
          path: file,
          size: fs.statSync(path.join(oldMetricsDir, file)).size,
          modified: fs.statSync(path.join(oldMetricsDir, file)).mtime,
          type: 'legacy'
        }))
      
      files.push(...oldFiles)
    }
    
    // 按修改时间排序
    files.sort((a, b) => new Date(b.modified) - new Date(a.modified))
    
    res.json({ success: true, files })
  } catch (error) {
    console.error('获取训练指标文件列表失败:', error)
    res.status(500).json({ success: false, message: '获取文件列表失败' })
  }
})

// 统一的配置管理API
app.get('/api/config-manager/:algorithm/:game', (req, res) => {
  try {
    const { algorithm, game } = req.params;
    const fs = require('fs');
    
    // 获取算法相关文件
    const algorithmFiles = {
      constants: path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/constants.py`),
      trainer: path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/trainer.py`),
      model: path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/model.py`),
      replay_buffer: path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/replay_buffer.py`),
      helpers: path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/helpers.py`),
      script: path.join(__dirname, `python/scripts/train_${algorithm.toLowerCase()}.py`)
    };
    
    // 获取游戏相关文件
    const gameFiles = {
      constants: path.join(__dirname, `python/games/${game.toLowerCase()}/core/constants.py`),
      wrappers: path.join(__dirname, `python/games/${game.toLowerCase()}/core/wrappers.py`)
    };
    
    const result = {
      success: true,
      algorithm: algorithm.toUpperCase(),
      game: game.toLowerCase(),
      files: {}
    };
    
    // 读取算法文件
    Object.entries(algorithmFiles).forEach(([type, filePath]) => {
      if (fs.existsSync(filePath)) {
        result.files[`algorithm_${type}`] = {
          path: filePath,
          content: fs.readFileSync(filePath, 'utf8'),
          type: type,
          category: 'algorithm'
        };
      }
    });
    
    // 读取游戏文件
    Object.entries(gameFiles).forEach(([type, filePath]) => {
      if (fs.existsSync(filePath)) {
        result.files[`game_${type}`] = {
          path: filePath,
          content: fs.readFileSync(filePath, 'utf8'),
          type: type,
          category: 'game'
        };
      }
    });
    
    res.json(result);
  } catch (error) {
    console.error('读取配置失败:', error);
    res.status(500).json({ success: false, message: '读取配置失败' });
  }
});

// 保存配置文件
app.post('/api/config-manager/save', (req, res) => {
  try {
    const { algorithm, game, fileType, category, content } = req.body;
    const fs = require('fs');
    
    if (!algorithm || !fileType || !category || !content) {
      return res.status(400).json({ success: false, message: '缺少必要参数' });
    }
    
    // 确定文件路径
    let filePath;
    if (category === 'algorithm') {
      if (fileType === 'trainer') {
        filePath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/trainer.py`);
      } else if (fileType === 'script') {
        filePath = path.join(__dirname, `python/scripts/train_${algorithm.toLowerCase()}.py`);
      } else {
        filePath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/${fileType}.py`);
      }
    } else if (category === 'game') {
      filePath = path.join(__dirname, `python/games/${game.toLowerCase()}/core/${fileType}.py`);
    } else {
      return res.status(400).json({ success: false, message: '无效的文件类别' });
    }
    
    // 确保不会修改原始备份
    const originalBackupDir = path.join(__dirname, 'backups', `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Original`);
    if (filePath.startsWith(originalBackupDir)) {
      return res.status(400).json({ 
        success: false, 
        message: '不能修改原始备份文件' 
      });
    }
    
    console.log('保存文件请求:', { algorithm, game, fileType, category, contentLength: typeof content === 'string' ? content.length : 'not string' });
    console.log('文件路径:', filePath);
    console.log('内容类型:', typeof content);
    
    // 确保content是字符串
    if (typeof content !== 'string') {
      console.error('内容不是字符串类型:', typeof content, content);
      return res.status(400).json({ success: false, message: '文件内容必须是字符串类型' });
    }
    
    // 先保存新内容到实际文件
    fs.writeFileSync(filePath, content, 'utf8');
    console.log('文件已保存到:', filePath);
    
    // 然后创建版本备份
    const now = new Date();
    const timestamp = now.toISOString().slice(0, 19).replace(/[:.]/g, '-');
    const backupBaseDir = path.join(__dirname, 'backups');
    const versionDir = path.join(backupBaseDir, `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Versions`);
    const fileDir = path.join(versionDir, `${fileType}_${timestamp}`);
    
    // 确保备份目录存在
    if (!fs.existsSync(fileDir)) {
      fs.mkdirSync(fileDir, { recursive: true });
    }
    
    const backupPath = path.join(fileDir, `${fileType}.py`);
    
    // 先保存新内容到实际文件
    fs.writeFileSync(filePath, content, 'utf8');
    
    // 备份刚保存的文件
    fs.copyFileSync(filePath, backupPath);
    
    res.json({
      success: true,
      message: '文件保存成功',
      backupPath: backupPath,
      filePath: filePath
    });
  } catch (error) {
    console.error('保存文件失败:', error);
    res.status(500).json({ success: false, message: '保存文件失败' });
  }
});

// 重置为默认配置
app.post('/api/config-manager/reset', (req, res) => {
  try {
    const { algorithm, game, fileType, category } = req.body;
    const fs = require('fs');
    
    if (!algorithm || !fileType || !category) {
      return res.status(400).json({ success: false, message: '缺少必要参数' });
    }
    
    // 确定文件路径
    let filePath;
    if (category === 'algorithm') {
      if (fileType === 'script') {
        filePath = path.join(__dirname, `python/scripts/train_${algorithm.toLowerCase()}.py`);
      } else {
        filePath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/${fileType}.py`);
      }
    } else if (category === 'game') {
      filePath = path.join(__dirname, `python/games/${game.toLowerCase()}/core/${fileType}.py`);
    }
    
    // 确保不会修改原始备份
    const originalBackupDir = path.join(__dirname, 'backups', `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Original`);
    if (filePath.startsWith(originalBackupDir)) {
      return res.status(400).json({ 
        success: false, 
        message: '不能修改原始备份文件' 
      });
    }
    
    // 从原始备份恢复
    const originalDir = path.join(__dirname, 'backups', `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Original`);
    let sourcePath;
    
    if (category === 'algorithm') {
      sourcePath = path.join(originalDir, 'algorithm', `${fileType}.py`);
    } else if (category === 'game') {
      sourcePath = path.join(originalDir, 'game', `${fileType}.py`);
    } else {
      sourcePath = filePath; // 使用当前版本作为默认
    }
    
    if (sourcePath && fs.existsSync(sourcePath)) {
      // 创建版本备份 - 重置前备份当前状态
      const now = new Date();
      const timestamp = now.toISOString().slice(0, 19).replace(/[:.]/g, '-');
      const backupBaseDir = path.join(__dirname, 'backups');
      const versionDir = path.join(backupBaseDir, `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Versions`);
      const fileDir = path.join(versionDir, `${fileType}_reset_${timestamp}`);
      
      // 确保备份目录存在
      if (!fs.existsSync(fileDir)) {
        fs.mkdirSync(fileDir, { recursive: true });
      }
      
      const backupPath = path.join(fileDir, `${fileType}.py`);
      
      if (fs.existsSync(filePath)) {
        fs.copyFileSync(filePath, backupPath);
      }
      
      // 恢复默认内容
      if (sourcePath !== filePath) {
        fs.copyFileSync(sourcePath, filePath);
      }
      
      res.json({
        success: true,
        message: '文件已重置为默认值',
        backupPath: backupPath,
        filePath: filePath
      });
    } else {
      // 如果没有找到默认文件，说明该文件类型不支持重置
      res.status(400).json({ 
        success: false, 
        message: `文件类型 ${fileType} 不支持重置为默认值，因为没有找到原始版本` 
      });
    }
  } catch (error) {
    console.error('重置文件失败:', error);
    res.status(500).json({ success: false, message: '重置文件失败' });
  }
});

// 创建原始备份
app.post('/api/config-manager/create-original-backup', (req, res) => {
  try {
    const { algorithm, game } = req.body;
    const fs = require('fs');
    
    if (!algorithm || !game) {
      return res.status(400).json({ success: false, message: '缺少必要参数' });
    }
    
    // 只支持DQN+Mario的原始备份
    if (algorithm.toLowerCase() !== 'dqn' || game.toLowerCase() !== 'mario') {
      return res.status(400).json({ 
        success: false, 
        message: '目前只支持DQN+Mario的原始备份' 
      });
    }
    
    const backupBaseDir = path.join(__dirname, 'backups');
    const originalDir = path.join(backupBaseDir, `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Original`);
    
    // 创建目录结构
    const algorithmDir = path.join(originalDir, 'algorithm');
    const gameDir = path.join(originalDir, 'game');
    
    if (!fs.existsSync(algorithmDir)) {
      fs.mkdirSync(algorithmDir, { recursive: true });
    }
    if (!fs.existsSync(gameDir)) {
      fs.mkdirSync(gameDir, { recursive: true });
    }
    
    // 复制算法文件
    const algorithmFiles = {
      constants: 'super-mario-bros-dqn/core/constants.py',
      model: 'super-mario-bros-dqn/core/model.py',
      replay_buffer: 'super-mario-bros-dqn/core/replay_buffer.py',
      helpers: 'super-mario-bros-dqn/core/helpers.py'
    };
    
    // 复制游戏文件
    const gameFiles = {
      wrappers: 'super-mario-bros-dqn/core/wrappers.py'
    };
    
    let copiedFiles = [];
    
    // 复制算法文件
    Object.entries(algorithmFiles).forEach(([fileType, sourcePath]) => {
      const fullSourcePath = path.join(__dirname, sourcePath);
      const targetPath = path.join(algorithmDir, `${fileType}.py`);
      
      if (fs.existsSync(fullSourcePath)) {
        fs.copyFileSync(fullSourcePath, targetPath);
        copiedFiles.push(`algorithm/${fileType}.py`);
      }
    });
    
    // 复制游戏文件
    Object.entries(gameFiles).forEach(([fileType, sourcePath]) => {
      const fullSourcePath = path.join(__dirname, sourcePath);
      const targetPath = path.join(gameDir, `${fileType}.py`);
      
      if (fs.existsSync(fullSourcePath)) {
        fs.copyFileSync(fullSourcePath, targetPath);
        copiedFiles.push(`game/${fileType}.py`);
      }
    });
    
    res.json({
      success: true,
      message: '原始备份创建成功',
      backupPath: originalDir,
      copiedFiles: copiedFiles
    });
    
  } catch (error) {
    console.error('创建原始备份失败:', error);
    res.status(500).json({ success: false, message: '创建原始备份失败' });
  }
});

// 重置为原始备份
app.post('/api/config-manager/reset-to-original', (req, res) => {
  try {
    const { algorithm, game, fileType, category } = req.body;
    const fs = require('fs');
    
    if (!algorithm || !game || !fileType || !category) {
      return res.status(400).json({ success: false, message: '缺少必要参数' });
    }
    
    // 确定文件路径
    let filePath;
    if (category === 'algorithm') {
      if (fileType === 'trainer') {
        filePath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/trainer.py`);
      } else if (fileType === 'script') {
        filePath = path.join(__dirname, `python/scripts/train_${algorithm.toLowerCase()}.py`);
      } else {
        filePath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/${fileType}.py`);
      }
    } else if (category === 'game') {
      filePath = path.join(__dirname, `python/games/${game.toLowerCase()}/core/${fileType}.py`);
    } else {
      return res.status(400).json({ success: false, message: '无效的文件类别' });
    }
    
    // 确保不会修改原始备份
    const originalBackupDir = path.join(__dirname, 'backups', `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Original`);
    if (filePath.startsWith(originalBackupDir)) {
      return res.status(400).json({ 
        success: false, 
        message: '不能修改原始备份文件' 
      });
    }
    
    // 原始备份路径
    const originalDir = path.join(__dirname, 'backups', `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Original`);
    const originalFilePath = path.join(originalDir, category === 'algorithm' ? 'algorithm' : 'game', `${fileType}.py`);
    
    if (!fs.existsSync(originalFilePath)) {
      return res.status(404).json({ 
        success: false, 
        message: '未找到原始备份文件，请先创建原始备份' 
      });
    }
    
    // 直接从原始备份恢复到实际文件
    console.log('恢复文件:', { originalFilePath, filePath });
    console.log('原始文件存在:', fs.existsSync(originalFilePath));
    console.log('目标文件存在:', fs.existsSync(filePath));
    
    fs.copyFileSync(originalFilePath, filePath);
    console.log('文件复制完成');
    
    res.json({
      success: true,
      message: '文件已重置为原始状态',
      originalPath: originalFilePath
    });
    
  } catch (error) {
    console.error('重置为原始状态失败:', error);
    res.status(500).json({ success: false, message: '重置为原始状态失败' });
  }
});

app.get('/api/config-manager/backups/:algorithm/:game/:fileType', (req, res) => {
  try {
    const { algorithm, game, fileType } = req.params;
    const fs = require('fs');
    
    // 构建版本备份目录路径
    const versionDir = path.join(__dirname, 'backups', `${algorithm.toUpperCase()}_${game.charAt(0).toUpperCase() + game.slice(1)}_Versions`);
    
    if (!fs.existsSync(versionDir)) {
      return res.json({
        success: true,
        backups: []
      });
    }
    
    // 查找该文件类型的所有版本备份
    const files = fs.readdirSync(versionDir).filter(dir => 
      dir.startsWith(`${fileType}_`) && fs.statSync(path.join(versionDir, dir)).isDirectory()
    );
    
    const backups = files.map(dir => {
      const backupFile = path.join(versionDir, dir, `${fileType}.py`);
      const stats = fs.statSync(backupFile);
      return {
        filename: dir,
        path: backupFile,
        created: stats.birthtime,
        size: stats.size,
        operation: 'save'
      };
    }).sort((a, b) => b.created - a.created);
    
    res.json({
      success: true,
      backups: backups
    });
  } catch (error) {
    console.error('获取备份列表失败:', error);
    res.status(500).json({ success: false, message: '获取备份列表失败' });
  }
});

// 删除备份
app.post('/api/config-manager/delete-backup', (req, res) => {
  try {
    const { backupPath } = req.body;
    const fs = require('fs');
    
    if (!backupPath) {
      return res.status(400).json({ success: false, message: '缺少备份路径' });
    }
    
    // 确保不会删除原始备份
    const originalBackupDir = path.join(__dirname, 'backups');
    const allOriginalDirs = fs.readdirSync(originalBackupDir).filter(dir => 
      dir.includes('_Original') && fs.statSync(path.join(originalBackupDir, dir)).isDirectory()
    );
    
    const backupDir = path.dirname(backupPath);
    const isOriginalBackup = allOriginalDirs.some(originalDir => 
      backupDir.includes(originalDir)
    );
    
    if (isOriginalBackup) {
      return res.status(400).json({ 
        success: false, 
        message: '不能删除原始备份' 
      });
    }
    
    // 删除整个备份目录
    if (fs.existsSync(backupDir)) {
      fs.rmSync(backupDir, { recursive: true, force: true });
      res.json({
        success: true,
        message: '备份删除成功'
      });
    } else {
      res.status(404).json({ success: false, message: '备份不存在' });
    }
    
  } catch (error) {
    console.error('删除备份失败:', error);
    res.status(500).json({ success: false, message: '删除备份失败' });
  }
});

// 恢复备份
app.post('/api/config-manager/restore', (req, res) => {
  try {
    const { algorithm, game, fileType, category, backupPath } = req.body;
    const fs = require('fs');
    
    if (!algorithm || !game || !fileType || !category || !backupPath) {
      return res.status(400).json({ success: false, message: '缺少必要参数' });
    }
    
    // 确定目标文件路径
    let targetPath;
    if (category === 'algorithm') {
      if (fileType === 'trainer') {
        targetPath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/trainer.py`);
      } else if (fileType === 'script') {
        targetPath = path.join(__dirname, `python/scripts/train_${algorithm.toLowerCase()}.py`);
      } else {
        targetPath = path.join(__dirname, `python/algorithms/${algorithm.toLowerCase()}/core/${fileType}.py`);
      }
    } else if (category === 'game') {
      targetPath = path.join(__dirname, `python/games/${game.toLowerCase()}/core/${fileType}.py`);
    } else {
      return res.status(400).json({ success: false, message: '无效的文件类别' });
    }
    
    console.log('恢复备份请求:', { algorithm, game, fileType, category, backupPath });
    
    // 使用Node.js path模块正确处理路径
    const normalizedBackupPath = path.resolve(backupPath);
    console.log('标准化后的备份路径:', normalizedBackupPath);
    console.log('备份文件存在:', fs.existsSync(normalizedBackupPath));
    
    if (!fs.existsSync(normalizedBackupPath)) {
      return res.status(404).json({ success: false, message: '备份文件不存在' });
    }
    
    // 直接恢复备份文件
    fs.copyFileSync(normalizedBackupPath, targetPath);
    
    res.json({
      success: true,
      message: '备份恢复成功',
      restoredPath: targetPath
    });
  } catch (error) {
    console.error('恢复备份失败:', error);
    res.status(500).json({ success: false, message: '恢复备份失败' });
  }
});

// 启动服务器
const server = app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
});

// 将WebSocket服务器附加到HTTP服务器
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});