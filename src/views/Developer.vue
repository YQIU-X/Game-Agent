<template>
  <div class="developer-container">
    <el-container>
      <el-header>
        <div class="header-content">
          <h2>游戏智能体开发平台 - 开发者模式</h2>
          <div class="user-info">
            <span>{{ username }}</span>
            <el-button type="text" @click="logout">退出登录</el-button>
          </div>
        </div>
      </el-header>
      
      <el-container>
        <el-aside width="300px">
          <el-menu
            :default-active="activeMenu"
            class="el-menu-vertical"
            @select="handleMenuSelect"
          >
            <el-menu-item index="config">
              <el-icon><Document /></el-icon>
              <span>配置文件</span>
            </el-menu-item>
            <el-menu-item index="training">
              <el-icon><VideoPlay /></el-icon>
              <span>训练控制</span>
            </el-menu-item>
            <el-menu-item index="preview">
              <el-icon><Monitor /></el-icon>
              <span>训练预览</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
        
        <el-main>
          <!-- 配置文件编辑 -->
          <div v-if="activeMenu === 'config'" class="config-section">
            <div class="section-header">
              <h3>配置文件编辑</h3>
              <div class="section-actions">
                <el-select v-model="selectedConfig" placeholder="选择/创建配置文件" filterable style="width: 260px" @change="handleSelectConfig">
                  <el-option v-for="f in configFiles" :key="f" :label="f" :value="f" />
                </el-select>
                <el-button @click="createConfigPrompt">新建</el-button>
                <el-button type="primary" @click="loadConfigFile" :disabled="!selectedConfig">打开</el-button>
                <el-button type="success" @click="saveConfigFile" :disabled="!configContent || !selectedConfig">保存</el-button>
              </div>
            </div>
            
            <el-input
              v-model="configContent"
              type="textarea"
              :rows="20"
              placeholder="请打开或创建配置文件..."
              class="config-editor"
            ></el-input>
          </div>
          
          <!-- 训练控制 -->
          <div v-if="activeMenu === 'training'" class="training-section">
            <div class="section-header">
              <h3>训练控制</h3>
              <div class="section-actions">
                <el-button 
                  type="primary" 
                  @click="startTraining" 
                  :disabled="isTraining"
                  :loading="isTraining"
                >
                  {{ isTraining ? '训练中...' : '开始训练' }}
                </el-button>
                <el-button 
                  type="danger" 
                  @click="stopTraining" 
                  :disabled="!isTraining"
                >
                  停止训练
                </el-button>
                <el-button 
                  type="info" 
                  @click="clearLogs"
                >
                  清空日志
                </el-button>
              </div>
            </div>
            
            <div class="command-input">
              <el-input
                v-model="trainingCommand"
                placeholder="输入训练命令..."
                :disabled="isTraining"
              >
                <template #prepend>$</template>
                <template #append>
                  <el-button @click="startTraining" :disabled="isTraining || !trainingCommand">
                    执行
                  </el-button>
                </template>
              </el-input>
            </div>
            
            <div class="training-logs">
              <h4>训练日志</h4>
              <el-card class="log-container">
                <div v-for="(log, index) in trainingLogs" :key="index" class="log-entry">
                  {{ log }}
                </div>
                <div v-if="trainingLogs.length === 0" class="empty-logs">
                  暂无日志信息
                </div>
              </el-card>
            </div>
          </div>
          
          <!-- 训练预览 -->
          <div v-if="activeMenu === 'preview'" class="preview-section">
            <div class="section-header">
              <h3>训练预览</h3>
            </div>
            
            <el-empty v-if="!isTraining" description="当前没有训练进行中，请先开始训练">
              <el-button type="primary" @click="activeMenu = 'training'">去训练</el-button>
            </el-empty>
            
            <div v-else class="preview-container">
              <div class="preview-placeholder">
                <el-icon class="preview-icon"><VideoPlay /></el-icon>
                <p>训练预览区域</p>
                <p class="preview-note">这里将显示训练中的智能体行为</p>
              </div>
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

export default {
  name: 'DeveloperView',
  setup() {
    const store = useStore()
    const router = useRouter()
    const activeMenu = ref('config')
    const configContent = ref('')
    const selectedConfig = ref('')
    const configFiles = ref([])
    const trainingCommand = ref('')
    let trainingInterval = null
    
    // 从store获取状态
    const username = computed(() => store.getters.username)
    const isTraining = computed(() => store.getters.isTraining)
    const trainingLogs = computed(() => store.getters.trainingLogs)
    
    // 菜单选择处理
    const handleMenuSelect = (index) => {
      activeMenu.value = index
    }
    
    // 配置文件处理
    const refreshConfigFiles = async () => {
      const { listConfigsApi } = await import('../api')
      const res = await listConfigsApi()
      if (res.data && res.data.success) {
        configFiles.value = res.data.configs || []
      }
    }

    const handleSelectConfig = () => {
      configContent.value = ''
    }

    const loadConfigFile = async () => {
      if (!selectedConfig.value) return
      const { readConfigApi } = await import('../api')
      try {
        ElMessage.info('正在加载配置文件...')
        const res = await readConfigApi(selectedConfig.value)
        if (res.data && res.data.success) {
          configContent.value = JSON.stringify(res.data.config, null, 2)
          ElMessage.success('配置文件加载成功')
        } else {
          throw new Error(res.data && res.data.message)
        }
      } catch (e) {
        ElMessage.error('加载失败: ' + (e.message || '未知错误'))
      }
    }

    const saveConfigFile = async () => {
      if (!configContent.value || !selectedConfig.value) {
        ElMessage.warning('请选择配置并填写内容')
        return
      }
      try {
        const payload = JSON.parse(configContent.value)
        const { saveConfigApi } = await import('../api')
        ElMessage.info('正在保存配置文件...')
        const res = await saveConfigApi(selectedConfig.value, payload)
        if (res.data && res.data.success) {
          store.commit('setConfigFile', configContent.value)
          ElMessage.success('配置文件保存成功')
          refreshConfigFiles()
        } else {
          throw new Error(res.data && res.data.message)
        }
      } catch (e) {
        ElMessage.error('保存失败: ' + (e.message || 'JSON格式错误'))
      }
    }

    const createConfigPrompt = async () => {
      try {
        const { value } = await ElMessageBox.prompt('请输入新配置文件名(以.json结尾)', '新建配置', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          inputPattern: /^.+\.json$/,
          inputErrorMessage: '文件名必须以.json结尾'
        })
        selectedConfig.value = value
        configContent.value = JSON.stringify({
          ENVIRONMENT: { env_name: 'SuperMarioBros-1-1-v0', render_mode: 'human', frame_skip: 4 },
          AGENT: { model_type: 'PPO', learning_rate: 0.0003, gamma: 0.99 },
          TRAINING: { max_episodes: 1000, steps_per_epoch: 2048 }
        }, null, 2)
      } catch (_) { void 0 }
    }
    
    // 训练控制
    const startTraining = async () => {
      if (!trainingCommand.value && activeMenu.value === 'training') {
        ElMessage.warning('请输入训练命令')
        return
      }
      
      const command = trainingCommand.value || 'python train_ppo_with_log.py --env SuperMarioBros-1-1-v0'
      
      ElMessageBox.confirm(
        `确定要开始训练吗？\n命令: ${command}`,
        '开始训练',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'info'
        }
      ).then(async () => {
        // 通过后端接口启动训练并开始轮询日志
        try {
          const { startTrainApi, fetchLogsApi } = await import('../api')
          store.commit('clearTrainingLogs')
          store.commit('setTrainingStatus', true)
          const payload = { game: 'super-mario-bros', level: 'SuperMarioBros-1-1-v0', agent: 'ppo', config: selectedConfig.value ? JSON.parse(configContent.value || '{}') : undefined }
          const res = await startTrainApi(payload)
          if (!res.data || !res.data.success) throw new Error((res.data && res.data.message) || '启动失败')
          const pid = res.data.processId
          store.commit('addTrainingLog', `执行命令: ${command}`)
          // 日志轮询
          if (trainingInterval) clearInterval(trainingInterval)
          trainingInterval = setInterval(async () => {
            try {
              const logRes = await fetchLogsApi(pid)
              if (logRes.data && logRes.data.success) {
                const logs = logRes.data.logs || []
                // 简单刷新所有日志
                // 清空并追加
                store.commit('clearTrainingLogs')
                logs.forEach(l => store.commit('addTrainingLog', l))
                if (logRes.data.completed) {
                  clearInterval(trainingInterval)
                  trainingInterval = null
                  store.commit('setTrainingStatus', false)
                }
              }
            } catch (_) { void 0 }
          }, 1500)
          ElMessage.success('训练已开始')
          if (activeMenu.value !== 'training') {
            activeMenu.value = 'training'
          }
        } catch (e) {
          store.commit('setTrainingStatus', false)
          ElMessage.error('启动训练失败: ' + (e.message || '未知错误'))
        }
      }).catch(() => { void 0 })
    }
    
    const stopTraining = () => {
      ElMessageBox.confirm(
        '确定要停止当前训练吗？',
        '停止训练',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }
      ).then(async () => {
        try {
          if (trainingInterval) {
            clearInterval(trainingInterval)
            trainingInterval = null
          }
          const lastPid = null // 简化：如需跟踪PID可在状态中存储
          if (lastPid) {
            const { stopProcessApi } = await import('../api')
            await stopProcessApi(lastPid)
          }
        } catch (_) { void 0 }
        store.commit('setTrainingStatus', false)
        store.commit('addTrainingLog', `[${new Date().toLocaleTimeString()}] 训练已手动停止`)
        ElMessage.info('训练已停止')
      }).catch(() => { void 0 })
    }
    
    const clearLogs = () => {
      store.commit('clearTrainingLogs')
      ElMessage.info('日志已清空')
    }
    
    // 登出
    const logout = () => {
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
      if (trainingInterval) {
        clearInterval(trainingInterval)
      }
    })
    
    refreshConfigFiles()

    return {
      activeMenu,
      username,
      configContent,
      selectedConfig,
      configFiles,
      trainingCommand,
      isTraining,
      trainingLogs,
      handleMenuSelect,
      loadConfigFile,
      saveConfigFile,
      refreshConfigFiles,
      createConfigPrompt,
      handleSelectConfig,
      startTraining,
      stopTraining,
      clearLogs,
      logout
    }
  }
}
</script>

<style scoped>
.developer-container {
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
}

.el-menu-vertical {
  height: 100%;
  border-right: none;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-actions {
  display: flex;
  gap: 10px;
}

.config-editor {
  font-family: monospace;
}

.command-input {
  margin-bottom: 20px;
}

.log-container {
  height: 400px;
  overflow-y: auto;
  padding: 10px;
  background-color: #1e1e1e;
  color: #f0f0f0;
  font-family: monospace;
}

.log-entry {
  margin-bottom: 5px;
  white-space: pre-wrap;
  word-break: break-all;
}

.empty-logs {
  text-align: center;
  color: #909399;
  padding: 20px;
}

.preview-container {
  height: 500px;
  display: flex;
  justify-content: center;
  align-items: center;
  border: 2px dashed #ccc;
  border-radius: 4px;
}

.preview-placeholder {
  text-align: center;
  color: #909399;
}

.preview-icon {
  font-size: 48px;
  margin-bottom: 10px;
}

.preview-note {
  font-size: 14px;
  margin-top: 10px;
}
</style>