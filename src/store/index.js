import { createStore } from 'vuex'

export default createStore({
  state: {
    isAuthenticated: localStorage.getItem('isAuthenticated') === 'true',
    userRole: localStorage.getItem('userRole') || '',
    username: localStorage.getItem('username') || '',
    userPermissions: JSON.parse(localStorage.getItem('userPermissions') || '{}'),
    // 开发者相关状态
    configFile: null,
    trainingLogs: [],
    isTraining: false,
    // 玩家相关状态
    selectedGame: '',
    selectedAgent: '',
    gameRewards: 0,
    gameTime: 0,
    playerActions: [],
    agentActions: []
  },
  mutations: {
    setAuthentication(state, { isAuthenticated, username, role, permissions }) {
      state.isAuthenticated = isAuthenticated
      state.username = username
      state.userRole = role
      state.userPermissions = permissions || {}
      
      // 保存到本地存储
      localStorage.setItem('isAuthenticated', isAuthenticated)
      localStorage.setItem('username', username)
      localStorage.setItem('userRole', role)
      localStorage.setItem('userPermissions', JSON.stringify(permissions || {}))
    },
    logout(state) {
      state.isAuthenticated = false
      state.username = ''
      state.userRole = ''
      state.userPermissions = {}
      
      // 清除本地存储
      localStorage.removeItem('isAuthenticated')
      localStorage.removeItem('username')
      localStorage.removeItem('userRole')
      localStorage.removeItem('userPermissions')
      // 注意：不清除记住的密码，让用户下次可以选择是否记住
    },
    // 开发者相关mutations
    setConfigFile(state, configFile) {
      state.configFile = configFile
    },
    addTrainingLog(state, log) {
      state.trainingLogs.push(log)
    },
    clearTrainingLogs(state) {
      state.trainingLogs = []
    },
    setTrainingStatus(state, isTraining) {
      state.isTraining = isTraining
    },
    // 玩家相关mutations
    setSelectedGame(state, game) {
      state.selectedGame = game
    },
    setSelectedAgent(state, agent) {
      state.selectedAgent = agent
    },
    updateGameRewards(state, rewards) {
      state.gameRewards = rewards
    },
    updateGameTime(state, time) {
      state.gameTime = time
    },
    addPlayerAction(state, action) {
      state.playerActions.push(action)
    },
    addAgentAction(state, action) {
      state.agentActions.push(action)
    },
    clearGameData(state) {
      state.gameRewards = 0
      state.gameTime = 0
      state.playerActions = []
      state.agentActions = []
    }
  },
  actions: {
    async login({ commit }, { username, password, role, rememberMe }) {
      const { loginApi } = await import('../api')
      const res = await loginApi({ username, password })
      const data = res.data
      if (!data || !data.success) {
        throw new Error((data && data.message) || '登录失败')
      }
      const serverUser = data.user || { username, role }
      // 统一账户系统：develop和player使用同一个账户，根据选择的角色决定访问权限
      const finalRole = role || serverUser.role || 'player'
      
      // 提取权限信息
      const permissions = {
        canAccessDeveloper: serverUser.canAccessDeveloper || false,
        canAccessPlayer: serverUser.canAccessPlayer || true
      }
      
      commit('setAuthentication', { 
        isAuthenticated: true, 
        username: serverUser.username || username, 
        role: finalRole,
        permissions: permissions
      })
      
      // 记住密码功能
      if (rememberMe) {
        localStorage.setItem('rememberedUsername', username)
        localStorage.setItem('rememberedPassword', password)
        localStorage.setItem('rememberedRole', finalRole)
        localStorage.setItem('rememberMe', 'true')
      } else {
        localStorage.removeItem('rememberedUsername')
        localStorage.removeItem('rememberedPassword')
        localStorage.removeItem('rememberedRole')
        localStorage.removeItem('rememberMe')
      }
      return true
    },
    // 开发者相关actions
    startTraining({ commit }, command) {
      // 这里应该有实际的训练启动逻辑
      commit('clearTrainingLogs')
      commit('setTrainingStatus', true)
      // 使用command参数记录当前执行的命令
      commit('addTrainingLog', `执行命令: ${command}`)
      // 模拟训练日志
      const interval = setInterval(() => {
        commit('addTrainingLog', `[${new Date().toLocaleTimeString()}] Training in progress...`)
      }, 2000)
      
      // 模拟训练结束
      setTimeout(() => {
        clearInterval(interval)
        commit('setTrainingStatus', false)
        commit('addTrainingLog', `[${new Date().toLocaleTimeString()}] Training completed.`)
      }, 10000)
    },
    // 玩家相关actions
    startGame({ commit }, { game, agent }) {
      commit('setSelectedGame', game)
      commit('setSelectedAgent', agent)
      commit('clearGameData')
      
      // 模拟游戏进行中的数据更新
      const interval = setInterval(() => {
        commit('updateGameRewards', Math.floor(Math.random() * 100))
        commit('updateGameTime', state => state.gameTime + 1)
        
        // 模拟玩家和智能体动作
        const actions = ['向左移动', '向右移动', '跳跃', '加速', '射击']
        const randomAction = actions[Math.floor(Math.random() * actions.length)]
        
        commit('addPlayerAction', randomAction)
        commit('addAgentAction', actions[Math.floor(Math.random() * actions.length)])
      }, 1000)
      
      // 返回清除函数，用于停止游戏
      return () => clearInterval(interval)
    }
  },
  getters: {
    isAuthenticated: state => state.isAuthenticated,
    userRole: state => state.userRole,
    username: state => state.username,
    userPermissions: state => state.userPermissions,
    canAccessDeveloper: state => state.userPermissions.canAccessDeveloper || false,
    canAccessPlayer: state => state.userPermissions.canAccessPlayer || true,
    rememberedCredentials: () => ({
      username: localStorage.getItem('rememberedUsername') || '',
      password: localStorage.getItem('rememberedPassword') || '',
      role: localStorage.getItem('rememberedRole') || ''
    }),
    // 开发者相关getters
    configFile: state => state.configFile,
    trainingLogs: state => state.trainingLogs,
    isTraining: state => state.isTraining,
    // 玩家相关getters
    selectedGame: state => state.selectedGame,
    selectedAgent: state => state.selectedAgent,
    gameRewards: state => state.gameRewards,
    gameTime: state => state.gameTime,
    playerActions: state => state.playerActions,
    agentActions: state => state.agentActions
  }
})