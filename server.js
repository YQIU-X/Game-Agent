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
    const destDir = path.join(__dirname, 'pretrained_models');
    if (!fs.existsSync(destDir)) return res.json({ success: true, models: [] });
    const files = fs.readdirSync(destDir).filter(f => f.endsWith('.dat') || f.endsWith('.pth'));
    res.json({ success: true, models: files });
  } catch (error) {
    res.status(500).json({ success: false, message: '读取失败: ' + error.message });
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
  const scriptPath = weights ? path.join(__dirname, 'infer_stream.py') : path.join(__dirname, 'render_stream.py');
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
  
  const scriptPath = path.join(__dirname, 'player_control.py');
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