<template>
  <div class="player-container">
    <el-container>
      <el-header>
        <div class="header-content">
          <h2>游戏智能体开发平台 - 玩家模式</h2>
          <div class="user-info">
            <span>{{ username }}</span>
            <el-button type="text" @click="logout">退出登录</el-button>
          </div>
        </div>
      </el-header>
      
      <el-container>
        <el-aside width="250px">
          <div class="game-selection">
            <h3>游戏选择</h3>
            <el-select v-model="selectedGame" placeholder="选择游戏" class="select-full-width" @change="onGameChange">
              <el-option 
                v-for="game in availableGames" 
                :key="game.id || game.value" 
                :label="game.name || game.label" 
                :value="game.id || game.value"
              ></el-option>
            </el-select>

            <h3>关卡选择</h3>
            <el-select 
              v-model="selectedLevel" 
              placeholder="选择关卡" 
              class="select-full-width"
              :disabled="!selectedGame"
            >
              <el-option 
                v-for="lv in availableLevels" 
                :key="lv.id || lv.value" 
                :label="lv.name || lv.label" 
                :value="lv.id || lv.value"
              ></el-option>
            </el-select>
            
            <h3>智能体选择</h3>
            <el-select 
              v-model="selectedAgent" 
              placeholder="选择智能体" 
              class="select-full-width"
              :disabled="!selectedGame"
            >
              <el-option 
                v-for="agent in availableAgents" 
                :key="agent.id || agent.value" 
                :label="agent.name || agent.label" 
                :value="agent.id || agent.value"
              ></el-option>
            </el-select>

            <h3>权重文件</h3>
            <div class="weights-row">
              <el-select 
                v-model="selectedWeights"
                placeholder="选择权重文件"
                class="select-full-width"
                :disabled="!selectedGame"
              >
                <el-option 
                  v-for="m in availableModels" 
                  :key="m" 
                  :label="m" 
                  :value="m"
                />
              </el-select>
            </div>
            
            <div class="game-controls">
              <el-button 
                type="primary" 
                @click="startGame" 
                :disabled="!canStartGame || isGameRunning"
                class="control-button"
              >
                开始游戏
              </el-button>
              <el-button 
                type="danger" 
                @click="stopGame" 
                :disabled="!isGameRunning"
                class="control-button"
              >
                结束游戏
              </el-button>
            </div>
          </div>
          
          <div class="game-stats" v-if="isGameRunning">
            <h3>游戏数据</h3>
            <el-card class="stats-card">
              <div class="stat-item">
                <span class="stat-label">游戏时间:</span>
                <span class="stat-value">{{ formatTime(gameTime) }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">玩家奖励:</span>
                <span class="stat-value">{{ playerReward.toFixed(2) }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">智能体奖励:</span>
                <span class="stat-value">{{ agentReward.toFixed(2) }}</span>
              </div>
            </el-card>
          </div>
        </el-aside>
        
        <el-main>
          <div class="game-area">
            <el-empty v-if="!isGameRunning" description="请选择游戏和智能体开始游戏"></el-empty>
            
            <div v-else class="game-screens">
              <div class="game-screen">
                <div class="screen-header">
                  <h3>玩家视图</h3>
                </div>
                <div class="screen-content player-screen" ref="playerScreen" tabindex="0" @keydown="handleKeyDown" @keyup="handleKeyUp">
                  <img v-if="playerFrameSrc" :src="playerFrameSrc" alt="player" style="max-width:100%;max-height:100%;object-fit:contain" />
                  <div v-else class="game-placeholder">
                    <el-icon class="game-icon"><Monitor /></el-icon>
                    <p>玩家游戏界面</p>
                  </div>
                </div>
              </div>
              
              <div class="game-screen">
                <div class="screen-header">
                  <h3>智能体视图</h3>
                </div>
                <div class="screen-content agent-screen">
                  <img v-if="agentFrameSrc" :src="agentFrameSrc" alt="agent" style="max-width:100%;max-height:100%;object-fit:contain" />
                  <div v-else class="game-placeholder">
                    <el-icon class="game-icon"><Monitor /></el-icon>
                    <p>智能体游戏界面</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div v-if="isGameRunning" class="game-logs">
              <el-tabs type="border-card">
                <el-tab-pane label="操作日志">
                  <div class="action-logs">
                    <div class="log-columns">
                      <div class="log-column">
                        <h4>玩家操作</h4>
                        <el-scrollbar height="200px">
                          <div v-for="(action, index) in playerActions" :key="'p'+index" class="action-item">
                            <span class="action-time">{{ formatActionTime(index) }}</span>
                            <span class="action-content">{{ action }}</span>
                          </div>
                          <div v-if="playerActions.length === 0" class="empty-actions">
                            暂无操作记录
                          </div>
                        </el-scrollbar>
                      </div>
                      
                      <div class="log-column">
                        <h4>智能体操作</h4>
                        <el-scrollbar height="200px">
                          <div v-for="(action, index) in agentActions" :key="'a'+index" class="action-item">
                            <span class="action-time">{{ formatActionTime(index) }}</span>
                            <span class="action-content">{{ action }}</span>
                          </div>
                          <div v-if="agentActions.length === 0" class="empty-actions">
                            暂无操作记录
                          </div>
                        </el-scrollbar>
                      </div>
                    </div>
                  </div>
                </el-tab-pane>
                
                <el-tab-pane label="游戏控制">
                  <div class="game-controls-panel">
                    <h4>键盘控制</h4>
                    <div class="keyboard-controls">
                      <div class="key-row">
                        <div class="key">W</div>
                      </div>
                      <div class="key-row">
                        <div class="key">A</div>
                        <div class="key">S</div>
                        <div class="key">D</div>
                      </div>
                      <div class="key-row">
                        <div class="key space-key">空格 (跳跃)</div>
                      </div>
                    </div>
                    
                    <h4>游戏设置</h4>
                    <div class="game-settings">
                      <el-slider v-model="gameSpeed" :min="1" :max="5" :step="1" show-stops>
                        <template #default="{ value }">
                          <span>速度: {{ value }}x</span>
                        </template>
                      </el-slider>
                    </div>
                  </div>
                </el-tab-pane>
              </el-tabs>
            </div>
          </div>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import { ref, computed, onBeforeUnmount } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import websocketService from '../services/websocket'

export default {
  name: 'PlayerView',
  setup() {
    const store = useStore()
    const router = useRouter()
    
    // 游戏选择
    const selectedGame = ref('')
    const selectedLevel = ref('')
    const selectedAgent = ref('')
    const isGameRunning = ref(false)
    const gameSpeed = ref(1)
    
    // 游戏数据
    const gameTime = ref(0)
    const playerReward = ref(0)
    const agentReward = ref(0)
    const playerActions = ref([])
    const agentActions = ref([])
    
    // 游戏定时器
    let gameInterval = null
    let gameCleanup = null
    let agentFramePoll = null
    let playerFramePoll = null
    const agentFrameSrc = ref('')
    const playerFrameSrc = ref('')
    
    // 进程ID
    let agentProcessId = null
    let playerProcessId = null
    
    // 从store获取状态
    const username = computed(() => store.getters.username)
    
    // 可用游戏列表
    const availableGames = ref([])
    const availableLevels = ref([])
    const availableAgents = ref([])
    const availableModels = ref([])
    
    // 是否可以开始游戏
    const selectedWeights = ref('')
    const canStartGame = computed(() => selectedGame.value && selectedLevel.value && selectedAgent.value)

    // 玩家键盘控制
    const keyToActionIndex = (key) => {
      const keyMap = {
        'a': 6,           // 左
        'd': 1,           // 右
        's': 0,           // 下（无动作）
        'w': 5,           // 上（跳跃）
        ' ': 5,           // 跳跃
        'x': 3,           // 攻击（右+攻击）
        'z': 4,           // 特殊（右+跳跃+攻击）
        'ArrowLeft': 6,   // 箭头键也支持
        'ArrowRight': 1,
        'ArrowDown': 0,
        'ArrowUp': 5
      }
      return keyMap[key] || 0
    }

    const handleKeyDown = (e) => {
      if (!isGameRunning.value) return
      
      // 防止重复按键
      if (e.repeat) return
      
      const actionIndex = keyToActionIndex(e.key)
      if (actionIndex > 0) {
        // 立即发送动作，不等待
        sendPlayerAction(actionIndex)
        const actionName = getActionName(actionIndex)
        playerActions.value.push(actionName)
        playerReward.value += actionIndex === 5 ? 2 : 1
      }
    }
    
    const handleKeyUp = () => {
      if (!isGameRunning.value) return
      
      // 按键释放时发送无动作
      sendPlayerAction(0)
      playerActions.value.push('无动作')
    }
    
    const getActionName = (actionIndex) => {
      const actions = ['无动作', '右移', '右跳', '右攻', '右跳攻', '跳跃', '左移']
      return actions[actionIndex] || '未知动作'
    }
    
    // 发送玩家动作
    const sendPlayerAction = async (actionIndex) => {
      if (!playerProcessId) return
      
      try {
        // 使用WebSocket发送动作
        websocketService.sendPlayerAction(playerProcessId, actionIndex)
      } catch (error) {
        console.error('发送玩家动作失败:', error)
      }
    }
    
    // 开始游戏
    const startGame = async () => {
      if (!canStartGame.value) {
        ElMessage.warning('请先选择游戏和智能体')
        return
      }
      
      ElMessageBox.confirm(
        `确定要开始游戏吗？\n游戏: ${getGameLabel(selectedGame.value)}\n关卡: ${getLevelLabel(selectedLevel.value)}\n智能体: ${getAgentLabel(selectedAgent.value)}`,
        '开始游戏',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'info'
        }
      ).then(async () => {
        isGameRunning.value = true
        gameTime.value = 0
        playerReward.value = 0
        agentReward.value = 0
        playerActions.value = []
        agentActions.value = []
        agentFrameSrc.value = ''
        playerFrameSrc.value = ''
        
        // 键盘监听
        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)

        // 启动游戏定时器
        gameInterval = setInterval(() => {
          gameTime.value++
        }, 1000) // 固定1秒间隔，与智能体同步
        
        // 同时启动智能体和玩家
        try {
          // 连接WebSocket
          await websocketService.connect()
          console.log('[前端] WebSocket连接成功')
          
          // 监听帧数据
          websocketService.on('frame', (data) => {
            console.log('[WebSocket] 收到帧数据:', data);
            if (data.processId === agentProcessId && data.type === 'frame') {
              if (data.frame) {
                agentFrameSrc.value = `data:image/jpeg;base64,${data.frame}`
              }
              if (data.agent_action) {
                agentActions.value.push(data.agent_action)
              }
              if (typeof data.reward === 'number') {
                agentReward.value += data.reward
              }
            }
          })
          
          // 监听玩家帧数据
          websocketService.on('player_frame', (data) => {
            console.log('[WebSocket] 收到玩家帧数据:', data);
            if (data.processId === playerProcessId) {
              if (data.frame) {
                playerFrameSrc.value = `data:image/jpeg;base64,${data.frame}`
              }
              if (data.agent_action) {
                const actionName = data.agent_action
                if (actionName !== 'waiting' && actionName !== '无动作') {
                  playerActions.value.push(actionName)
                }
              }
              if (typeof data.reward === 'number') {
                playerReward.value += data.reward
              }
            }
          })
          
          // 启动智能体
          const { startStreamApi } = await import('../api')
          const agentPayload = { level: selectedLevel.value, fps: 8 }
          if (selectedWeights.value) agentPayload.weights = selectedWeights.value
          const agentResponse = await startStreamApi(agentPayload)
          
          if (agentResponse.data && agentResponse.data.success) {
            agentProcessId = agentResponse.data.processId
          }
          
          // 启动玩家
          const { startPlayerApi } = await import('../api')
          const playerPayload = { level: selectedLevel.value, actionSpace: 'SIMPLE', fps: 30 }
          const playerResponse = await startPlayerApi(playerPayload)
          
          if (playerResponse.data && playerResponse.data.success) {
            playerProcessId = playerResponse.data.processId
          }
          
        } catch (e) {
          console.error('启动游戏失败:', e)
          ElMessage.error('启动游戏失败')
        }
        
        ElMessage.success('游戏已开始，请点击玩家画面并使用键盘控制')
        
        ElMessage.success('游戏已开始，请点击玩家画面并使用键盘控制')
      }).catch(() => { void 0 })
    }
    
    // 停止游戏
    const stopGame = () => {
      ElMessageBox.confirm(
        '确定要结束当前游戏吗？',
        '结束游戏',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }
      ).then(async () => {
        // 停止智能体
        if (agentProcessId) {
          try {
            await fetch(`/api/stop/${agentProcessId}`, { method: 'POST' })
          } catch (_) { /* ignore */ }
        }
        
        // 停止玩家
        if (playerProcessId) {
          try {
            await fetch(`/api/stop/${playerProcessId}`, { method: 'POST' })
          } catch (_) { /* ignore */ }
        }
        
        if (gameInterval) {
          clearInterval(gameInterval)
          gameInterval = null
        }
        
        if (typeof gameCleanup === 'function') {
          gameCleanup()
          gameCleanup = null
        }
        if (agentFramePoll) { clearInterval(agentFramePoll); agentFramePoll = null }
        if (playerFramePoll) { clearInterval(playerFramePoll); playerFramePoll = null }
        
        // 断开WebSocket连接
        websocketService.disconnect()
        
        isGameRunning.value = false
        agentFrameSrc.value = ''
        playerFrameSrc.value = ''
        agentProcessId = null
        playerProcessId = null
        window.removeEventListener('keydown', handleKeyDown)
        window.removeEventListener('keyup', handleKeyUp)
        ElMessage.info('游戏已结束')
      }).catch(() => { void 0 })
    }
    
    // 格式化时间
    const formatTime = (seconds) => {
      const mins = Math.floor(seconds / 60)
      const secs = seconds % 60
      return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }
    
    // 格式化操作时间
    const formatActionTime = (index) => {
      const actionTime = index + 1
      return formatTime(actionTime)
    }
    
    // 获取标签
    const getGameLabel = (value) => {
      const game = availableGames.value.find(g => g.id === value || g.value === value)
      return game ? (game.name || game.label) : value
    }
    const getLevelLabel = (value) => {
      const level = availableLevels.value.find(l => l.id === value || l.value === value)
      return level ? (level.name || level.label) : value
    }
    const getAgentLabel = (value) => {
      const agent = availableAgents.value.find(a => a.id === value || a.value === value)
      return agent ? (agent.name || agent.label) : value
    }
    
    // 登出
    const logout = () => {
      if (isGameRunning.value) {
        ElMessage.warning('请先结束当前游戏再退出')
        return
      }
      
      ElMessageBox.confirm(
        '确定要退出登录吗？',
        '退出登录',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }
      ).then(() => {
        store.commit('logout')
        router.push('/')
      }).catch(() => { void 0 })
    }
    
    // 组件销毁前清理
    onBeforeUnmount(() => {
      if (gameInterval) {
        clearInterval(gameInterval)
      }
      if (typeof gameCleanup === 'function') {
        gameCleanup()
      }
      if (agentFramePoll) { clearInterval(agentFramePoll); agentFramePoll = null }
      if (playerFramePoll) { clearInterval(playerFramePoll); playerFramePoll = null }
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    })
    
    // 初始化：仅一个游戏(super-mario-bros)，关卡1-1至3-3，智能体仅dqn
    availableGames.value = [{ id: 'super-mario-bros', name: '超级马里奥' }]
    const levelIds = ['1-1','1-2','1-3','1-4','2-1','2-2','2-3','2-4','3-1','3-2','3-3']
    availableLevels.value = levelIds.map(v => ({ id: `SuperMarioBros-${v}-v0`, name: `关卡 ${v}` }))
    availableAgents.value = [{ id: 'dqn', name: 'DQN 智能体' }]

    const onGameChange = async () => {
      selectedLevel.value = ''
      selectedAgent.value = ''
      selectedWeights.value = ''
    }

    // 页面加载时自动列举可用模型
    ;(async () => {
      try {
        const { listModelsApi } = await import('../api')
        const r = await listModelsApi()
        if (r.data && r.data.success) availableModels.value = r.data.models || []
      } catch (_) { /* ignore */ }
    })()

    return {
      username,
      selectedGame,
      selectedLevel,
      selectedAgent,
      availableGames,
      availableLevels,
      availableAgents,
      availableModels,
      isGameRunning,
      gameSpeed,  
      gameTime,
      playerReward,
      agentReward,
      playerActions,
      agentActions,
      selectedWeights,
      canStartGame,
      agentFrameSrc,
      playerFrameSrc,
      startGame,
      stopGame,
      formatTime,
      formatActionTime,
      onGameChange,
      handleKeyDown,
      handleKeyUp,
      logout
    }
  }
}
</script>

<style scoped>
.player-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.el-header {
  background-color: #409EFF;
  color: white;
  line-height: 60px;
  padding: 0 20px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.el-aside {
  background-color: #f5f7fa;
  border-right: 1px solid #e6e6e6;
  padding: 20px;
}

.game-selection h3 {
  margin-top: 20px;
  margin-bottom: 10px;
}

.select-full-width {
  width: 100%;
  margin-bottom: 15px;
}

.game-controls {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.control-button {
  width: 100%;
}

.game-stats {
  margin-top: 30px;
}

.stats-card {
  margin-top: 10px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.stat-label {
  font-weight: bold;
}

.stat-value {
  font-family: monospace;
}

.game-area {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.game-screens {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  flex: 1;
}

.game-screen {
  flex: 1;
  display: flex;
  flex-direction: column;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  overflow: hidden;
}

.screen-header {
  background-color: #f5f7fa;
  padding: 10px;
  border-bottom: 1px solid #dcdfe6;
}

.screen-header h3 {
  margin: 0;
}

.screen-content {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #000;
}

.game-placeholder {
  text-align: center;
  color: #909399;
}

.game-icon {
  font-size: 48px;
  margin-bottom: 10px;
  color: #dcdfe6;
}

.game-logs {
  margin-top: auto;
}

.action-logs {
  padding: 10px;
}

.log-columns {
  display: flex;
  gap: 20px;
}

.log-column {
  flex: 1;
}

.log-column h4 {
  margin-top: 0;
  margin-bottom: 10px;
  text-align: center;
}

.action-item {
  display: flex;
  margin-bottom: 5px;
  padding: 5px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.action-time {
  font-family: monospace;
  margin-right: 10px;
  color: #909399;
}

.empty-actions {
  text-align: center;
  color: #909399;
  padding: 20px;
}

.game-controls-panel {
  padding: 10px;
}

.keyboard-controls {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 20px;
}

.key-row {
  display: flex;
  gap: 5px;
  margin-bottom: 5px;
}

.key {
  width: 40px;
  height: 40px;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f5f7fa;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-weight: bold;
}

.space-key {
  width: 200px;
}

.game-settings {
  padding: 0 20px;
}
</style>