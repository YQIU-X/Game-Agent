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
            <el-menu-item index="training">
              <el-icon><VideoPlay /></el-icon>
              <span>训练控制</span>
            </el-menu-item>
            <el-menu-item index="config">
              <el-icon><Setting /></el-icon>
              <span>参数配置</span>
            </el-menu-item>
            <el-menu-item index="visualization">
              <el-icon><TrendCharts /></el-icon>
              <span>训练可视化</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
        
        <el-main>
          <!-- 通用训练控制 -->
          <div v-if="activeMenu === 'training'" class="training-section training-scrollable">
            <div class="section-header">
              <h3>强化学习训练控制</h3>
              <div class="section-actions">
                <el-button 
                  type="primary" 
                  @click="startTraining" 
                  :disabled="isTraining"
                  :loading="isTraining"
                >
                  {{ isTraining ? '训练中...' : `开始${selectedAlgorithm}训练` }}
                </el-button>
                <el-button 
                  type="danger" 
                  @click="stopTraining" 
                  :disabled="!isTraining"
                >
                  停止训练
                </el-button>
                <el-button 
                  type="warning"
                  @click="resetTrainingConfig"
                  :disabled="isTraining"
                >
                  重置配置
                </el-button>
                <el-button 
                  type="success"
                  @click="syncConstantsFromTrainingForm"
                  :disabled="isTraining"
                >
                  保存到配置文件
                </el-button>
                <el-button 
                  type="info" 
                  @click="clearLogs"
                >
                  清空日志
                </el-button>
              </div>
            </div>
            
            <!-- 算法和游戏选择 -->
            <el-card class="training-config">
              <template #header>
                <div class="card-header">
                  <span>算法和游戏配置</span>
                </div>
              </template>
              
              <el-form :model="trainingForm" label-width="120px" :disabled="isTraining">
                <el-row :gutter="20">
                  <el-col :span="12">
                    <el-form-item label="算法">
                      <el-select v-model="selectedAlgorithm" placeholder="选择算法" @change="onAlgorithmChange">
                        <el-option 
                          v-for="(config, name) in algorithmConfigs" 
                          :key="name" 
                          :label="config.name" 
                          :value="name"
                        >
                          <span>{{ config.name }}</span>
                          <span style="float: right; color: #8492a6; font-size: 13px">{{ config.description }}</span>
                        </el-option>
                      </el-select>
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="游戏">
                      <el-select v-model="selectedGame" placeholder="选择游戏" @change="onGameChange">
                        <el-option 
                          v-for="(config, name) in gameConfigs" 
                          :key="name" 
                          :label="config.name" 
                          :value="name"
                        />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>
                
                <el-row :gutter="20">
                  <el-col :span="12">
                    <el-form-item label="环境">
                      <el-select v-model="trainingForm.environment" placeholder="选择环境" :disabled="!selectedGame">
                        <el-option 
                          v-for="env in availableEnvironments" 
                          :key="env" 
                          :label="env" 
                          :value="env"
                        />
                      </el-select>
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="动作空间">
                      <el-select v-model="trainingForm.action_space" placeholder="选择动作空间" :disabled="!selectedGame">
                        <el-option 
                          v-for="(label, value) in availableActionSpaces" 
                          :key="value" 
                          :label="label" 
                          :value="value"
                        />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>
                
                <el-row :gutter="20">
                  <el-col :span="12">
                    <el-form-item label="训练轮数">
                      <el-input-number v-model="trainingForm.episodes" :min="100" :max="100000" :step="100" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="每轮最大步数">
                      <el-input-number v-model="trainingForm.max_steps_per_episode" :min="1000" :max="50000" :step="1000" />
                    </el-form-item>
                  </el-col>
                </el-row>
              </el-form>
            </el-card>
            
            <!-- 算法特定参数 -->
            <el-card v-if="selectedAlgorithm && algorithmParameters.length > 0" class="algorithm-params">
              <template #header>
                <div class="card-header">
                  <span>{{ algorithmConfigs[selectedAlgorithm]?.name }} 参数配置</span>
                </div>
                </template>
              
              <el-form :model="trainingForm" label-width="140px" :disabled="isTraining">
                <template v-for="(param, index) in algorithmParameters" :key="param.name">
                  <el-row v-if="index % 2 === 0" :gutter="20">
                    <el-col :span="12">
                      <el-form-item :label="param.description">
                        <el-input-number 
                          v-if="param.type === 'int'"
                          v-model="trainingForm[param.name]"
                          :min="param.min"
                          :max="param.max"
                          :step="param.step"
                          style="width: 100%"
                        />
                        <el-input-number 
                          v-else-if="param.type === 'float'"
                          v-model="trainingForm[param.name]"
                          :min="param.min"
                          :max="param.max"
                          :step="param.step"
                          :precision="param.precision"
                          style="width: 100%"
                        />
                      </el-form-item>
                    </el-col>
                    <el-col v-if="algorithmParameters[index + 1]" :span="12">
                      <el-form-item :label="algorithmParameters[index + 1].description">
                        <el-input-number 
                          v-if="algorithmParameters[index + 1].type === 'int'"
                          v-model="trainingForm[algorithmParameters[index + 1].name]"
                          :min="algorithmParameters[index + 1].min"
                          :max="algorithmParameters[index + 1].max"
                          :step="algorithmParameters[index + 1].step"
                          style="width: 100%"
                        />
                        <el-input-number 
                          v-else-if="algorithmParameters[index + 1].type === 'float'"
                          v-model="trainingForm[algorithmParameters[index + 1].name]"
                          :min="algorithmParameters[index + 1].min"
                          :max="algorithmParameters[index + 1].max"
                          :step="algorithmParameters[index + 1].step"
                          :precision="algorithmParameters[index + 1].precision"
                          style="width: 100%"
                        />
                      </el-form-item>
                    </el-col>
                  </el-row>
                </template>
              </el-form>
            </el-card>
            
            <!-- 通用标志 -->
            <el-card class="training-flags">
              <template #header>
                <div class="card-header">
                  <span>训练选项</span>
            </div>
              </template>
              
              <el-form :model="trainingForm" label-width="120px" :disabled="isTraining">
                <el-row :gutter="20">
                  <el-col :span="8">
                    <el-form-item label="启用渲染">
                      <el-switch v-model="trainingForm.render" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item label="保存模型">
                      <el-switch v-model="trainingForm.save_model" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item label="使用GPU">
                      <el-switch v-model="trainingForm.use_gpu" />
                    </el-form-item>
                  </el-col>
                </el-row>
                
                <el-row :gutter="20">
                  <el-col :span="8">
                    <el-form-item label="详细输出">
                      <el-switch v-model="trainingForm.verbose" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item label="模型保存频率">
                      <el-input-number v-model="trainingForm.save_frequency" :min="10" :max="1000" :step="10" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item label="日志输出频率">
                      <el-input-number v-model="trainingForm.log_frequency" :min="1" :max="100" :step="1" />
                    </el-form-item>
                  </el-col>
                </el-row>
              </el-form>
            </el-card>
            
            <!-- 训练日志 -->
            <el-card class="training-logs">
              <template #header>
                <div class="card-header">
                  <span>训练日志</span>
                  <el-button type="text" @click="clearLogs" size="small">清空</el-button>
                </div>
              </template>
              
              <div class="log-container">
                <div v-for="(log, index) in trainingLogs" :key="index" class="log-entry">
                  {{ log }}
                </div>
                <div v-if="trainingLogs.length === 0" class="empty-logs">
                  暂无日志信息
                </div>
                </div>
              </el-card>
          </div>
          
          <!-- 参数配置 -->
          <div v-if="activeMenu === 'config'" class="config-section">
            <div class="config-scrollable">
              <div class="section-header">
              <h3>统一配置管理器</h3>
              <div class="section-actions">
                <el-button 
                  type="primary" 
                  @click="loadAllConfigs"
                  :disabled="!selectedAlgorithm || !selectedGame"
                >
                  加载{{ selectedAlgorithm }}+{{ selectedGame }}配置
                </el-button>
                <el-button 
                  type="success" 
                  @click="saveCurrentFile"
                  :disabled="!currentFileContent"
                >
                  保存备份
                </el-button>
                <el-button 
                  type="warning" 
                  @click="resetToOriginal"
                  :disabled="!currentFileInfo || !isResetSupported"
                >
                  恢复到默认状态
                </el-button>
                <el-button 
                  type="info" 
                  @click="showBackupManager = true"
                  :disabled="!currentFileInfo"
                >
                  管理备份
                </el-button>
              </div>
            </div>
            
            <!-- 算法和游戏选择 -->
            <el-card class="config-selector">
              <el-form :model="configForm" label-width="120px">
                <el-row :gutter="20">
                  <el-col :span="12">
                    <el-form-item label="算法类型">
                      <el-select v-model="selectedAlgorithm" placeholder="选择算法" @change="onConfigAlgorithmChange" style="width: 100%">
                        <el-option 
                          v-for="(config, key) in algorithmConfigs" 
                          :key="key" 
                          :label="config.name" 
                          :value="key"
                        />
                      </el-select>
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="游戏类型">
                      <el-select v-model="selectedGame" placeholder="选择游戏" @change="onConfigGameChange" style="width: 100%">
                        <el-option 
                          v-for="(config, key) in gameConfigs" 
                          :key="key" 
                          :label="config.name" 
                          :value="key"
                        />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>
              </el-form>
            </el-card>
            
            <!-- 文件列表 -->
            <el-card v-if="configFiles.length > 0" class="file-list">
              <template #header>
                <div class="card-header">
                  <span>代码文件列表</span>
                </div>
              </template>
              <el-tabs v-model="activeFileTab" @tab-click="onFileTabClick" @tab-change="onFileTabClick">
                <el-tab-pane 
                  v-for="file in configFiles" 
                  :key="file.key"
                  :label="getFileDisplayName(file)"
                  :name="file.key"
                >
                  <div class="file-info">
                    <el-tag :type="file.category === 'algorithm' ? 'primary' : 'success'">
                      {{ file.category === 'algorithm' ? '算法' : '游戏' }}
                    </el-tag>
                    <span class="file-path">{{ file.path }}</span>
                  </div>
                </el-tab-pane>
              </el-tabs>
            </el-card>
            
            <!-- 文件编辑器 -->
            <el-card v-if="currentFileContent" class="config-editor">
              <template #header>
                <div class="card-header">
                  <span>{{ getFileDisplayName(currentFileInfo) }} - 代码编辑器</span>
                  <div class="editor-actions">
                    <el-button size="small" @click="formatCode">格式化</el-button>
                  </div>
                </div>
              </template>
              <div class="code-editor-container">
                <div class="syntax-highlight-editor">
                  <pre><code :class="languageClass" v-html="highlightedCode"></code></pre>
                  <textarea
                    v-model="currentFileContent"
                    class="code-textarea-overlay"
                    @input="onCodeChange"
                    @keydown="onCodeKeydown"
                    @scroll="onTextareaScroll"
                    spellcheck="false"
                    autocomplete="off"
                    autocorrect="off"
                    autocapitalize="off"
                    placeholder="代码内容将在这里显示..."
                  ></textarea>
                </div>
              </div>
            </el-card>
            
            <!-- 文件信息 -->
            <el-card v-if="currentFileInfo" class="config-info">
              <template #header>
                <div class="card-header">
                  <span>文件信息</span>
                </div>
              </template>
              <el-descriptions :column="2" border>
                <el-descriptions-item label="文件类型">{{ getFileDisplayName(currentFileInfo) }}</el-descriptions-item>
                <el-descriptions-item label="类别">{{ currentFileInfo.category === 'algorithm' ? '算法' : '游戏' }}</el-descriptions-item>
                <el-descriptions-item label="文件路径">{{ currentFileInfo.path }}</el-descriptions-item>
                <el-descriptions-item label="最后修改">{{ formatFileTime(currentFileInfo.lastModified) }}</el-descriptions-item>
              </el-descriptions>
            </el-card>
            </div>
          </div>
          
          <!-- 训练可视化 -->
          <div v-if="activeMenu === 'visualization'" class="visualization-section visualization-scrollable">
            <div class="section-header">
              <h3>训练可视化</h3>
              <div class="section-actions">
                <el-select v-model="selectedEnvironment" placeholder="选择训练记录" @change="loadMetricsData" style="width: 300px; margin-right: 10px;">
                  <el-option
                    v-for="file in availableMetricsFiles"
                    :key="file.name"
                    :label="file.displayName"
                    :value="file.name">
                    <span style="float: left">{{ file.displayName }}</span>
                    <span style="float: right; color: #8492a6; font-size: 13px">{{ file.type }}</span>
                  </el-option>
                </el-select>
                <el-button @click="exportData" :disabled="!hasTrainingData">导出数据</el-button>
              </div>
            </div>
            
            <!-- 实时图表控制面板 -->
            <div class="realtime-controls">
              <div class="metrics-selector">
                <h4>选择要显示的指标：</h4>
                <div class="metrics-checkboxes">
                  <el-checkbox-group v-model="selectedMetrics" @change="updateRealtimeChart">
                    <el-checkbox 
                      v-for="metric in availableMetricsForCurrentAlgorithm" 
                      :key="metric.key" 
                      :label="metric.key"
                    >
                      {{ metric.label }}
                    </el-checkbox>
                  </el-checkbox-group>
                  <div v-if="availableMetricsForCurrentAlgorithm.length === 0" style="color: red;">
                    没有可用的指标 (算法: {{ selectedAlgorithm }})
                  </div>
                </div>
              </div>
              
              <div class="chart-controls">
                <el-button-group>
                  <el-button :type="chartType === 'line' ? 'primary' : ''" @click="chartType = 'line'; updateRealtimeChart()">折线图</el-button>
                  <el-button :type="chartType === 'bar' ? 'primary' : ''" @click="chartType = 'bar'; updateRealtimeChart()">柱状图</el-button>
                  <el-button :type="chartType === 'scatter' ? 'primary' : ''" @click="chartType = 'scatter'; updateRealtimeChart()">散点图</el-button>
                </el-button-group>
                
                <span class="refresh-interval">
                  自动刷新间隔: 
                  <el-select v-model="refreshInterval" style="width: 80px;">
                    <el-option label="1秒" :value="1000"></el-option>
                    <el-option label="2秒" :value="2000"></el-option>
                    <el-option label="5秒" :value="5000"></el-option>
                    <el-option label="10秒" :value="10000"></el-option>
                  </el-select>
                </span>
              </div>
            </div>
            
            <!-- 实时图表显示区域 -->
            <div class="realtime-chart-container">
              <el-card class="main-chart-card">
                <template #header>
                  <div class="chart-header">
                    <span>训练指标实时监控</span>
                    <div class="chart-stats">
                      <span v-if="trainingMetrics.length > 0">
                        当前Episode: {{ trainingMetrics.length - 1 }} | 
                        最新奖励: {{ trainingMetrics[trainingMetrics.length - 1]?.total_reward?.toFixed(2) || 0 }}
                      </span>
                    </div>
                  </div>
                </template>
                <div ref="realtimeChartRef" class="realtime-chart"></div>
              </el-card>
              
              <!-- 训练统计 -->
              <el-card class="stats-card">
                <template #header>
                  <div class="card-header">
                    <span>训练统计</span>
                  </div>
                </template>
                <div class="stats-grid">
                  <div class="stat-item">
                    <div class="stat-value">{{ currentEpisode }}</div>
                    <div class="stat-label">当前Episode</div>
                  </div>
                  <div class="stat-item">
                    <div class="stat-value">{{ avgReward.toFixed(2) }}</div>
                    <div class="stat-label">平均奖励</div>
                  </div>
                  <div class="stat-item">
                    <div class="stat-value">{{ bestReward.toFixed(2) }}</div>
                    <div class="stat-label">最佳奖励</div>
                  </div>
                  <div class="stat-item">
                    <div class="stat-value">{{ totalSteps }}</div>
                    <div class="stat-label">总步数</div>
                  </div>
                </div>
              </el-card>
            </div>
            
            <!-- 无数据提示 -->
            <div v-if="!hasTrainingData" class="no-data-tip">
              <el-empty description="暂无训练数据">
                <template #image>
                  <el-icon size="100" color="#c0c4cc"><TrendCharts /></el-icon>
                </template>
                <el-button type="primary" @click="loadMetricsData">加载数据</el-button>
              </el-empty>
            </div>
          </div>
          
        </el-main>
      </el-container>
    </el-container>
    
    <!-- 备份管理对话框 -->
    <el-dialog
      v-model="showBackupManager"
      title="备份管理"
      width="80%"
      :close-on-click-modal="false"
    >
      <div class="backup-manager">
        <div class="backup-header">
          <h4>{{ getFileDisplayName(currentFileInfo) }} - 备份历史</h4>
          <div class="backup-actions">
            <el-button type="primary" @click="refreshBackups">刷新</el-button>
            <el-button type="danger" @click="deleteAllBackups" :disabled="backupList.length === 0">
              删除所有备份
            </el-button>
  </div>
        </div>
        
        <el-table :data="backupList" style="width: 100%" max-height="400">
          <el-table-column prop="filename" label="备份文件名" width="300" />
          <el-table-column prop="operation" label="操作类型" width="120">
            <template #default="scope">
              <el-tag :type="scope.row.operation === 'save' ? 'success' : scope.row.operation === 'reset' ? 'warning' : 'info'">
                {{ scope.row.operation === 'save' ? '保存' : scope.row.operation === 'reset' ? '重置' : '未知' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="created" label="创建时间" width="200">
            <template #default="scope">
              {{ formatFileTime(scope.row.created) }}
            </template>
          </el-table-column>
          <el-table-column prop="size" label="文件大小" width="120">
            <template #default="scope">
              {{ Math.round(scope.row.size / 1024 * 100) / 100 }} KB
            </template>
          </el-table-column>
          <el-table-column label="操作" width="200">
            <template #default="scope">
              <el-button size="small" @click="restoreBackup(scope.row)">恢复</el-button>
              <el-button size="small" type="danger" @click="deleteBackup(scope.row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-dialog>
    
    <!-- 添加图表对话框 -->
    <el-dialog v-model="showAddChartDialog" title="添加图表" width="500px">
      <div class="add-chart-form">
        <el-form :model="newChart" label-width="100px">
          <el-form-item label="图表标题">
            <el-input v-model="newChart.title" placeholder="请输入图表标题"></el-input>
          </el-form-item>
          <el-form-item label="选择指标">
            <el-checkbox-group v-model="newChart.metrics">
              <el-checkbox 
                v-for="metric in availableMetricsForChart" 
                :key="metric.value" 
                :label="metric.value">
                {{ metric.label }}
              </el-checkbox>
            </el-checkbox-group>
          </el-form-item>
          <el-form-item label="图表类型">
            <el-radio-group v-model="newChart.type">
              <el-radio label="line">折线图</el-radio>
              <el-radio label="bar">柱状图</el-radio>
            </el-radio-group>
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="showAddChartDialog = false">取消</el-button>
        <el-button type="primary" @click="addChart" :disabled="!newChart.title || newChart.metrics.length === 0">添加</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, computed, onBeforeUnmount, onMounted, nextTick, watch } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'

export default {
  name: 'DeveloperView',
  setup() {
    const store = useStore()
    const router = useRouter()
    const activeMenu = ref('training')
    const isTraining = ref(false)
    const trainingLogs = ref([])
    const trainingMetrics = ref([])
    const currentProcessId = ref(null)
    const enableRender = ref(false)
    
    // 算法和游戏配置
    const algorithmConfigs = ref({})
    const gameConfigs = ref({})
    const selectedAlgorithm = ref('DQN')
    const selectedGame = ref('mario')
    
    // 动态参数
    const availableEnvironments = ref([])
    const availableActionSpaces = ref({})
    const algorithmParameters = ref([])
    
    // 参数文件管理
    // 统一配置管理相关
    const configFiles = ref([])
    const currentFileContent = ref('')
    const currentFileInfo = ref(null)
    const activeFileTab = ref('')
    const showBackupManager = ref(false)
    const backupList = ref([])
    const syntaxHighlight = ref(true)
    const configForm = ref({})
    const isEditing = ref(false)
    const originalContent = ref('')
    
    // 兼容性变量 (保留旧API支持)
    const configFileType = ref('constants')
    const configFileContent = ref('')
    const configFilePath = ref('')
    const configFileInfo = ref(null)
    
    // 通用训练配置表单 - 初始化为空，将从备份文件加载
    const trainingForm = ref({
      algorithm: 'DQN',
      game: 'mario',
      environment: '',
      action_space: '',
      episodes: 0,
      max_steps_per_episode: 10000,
      save_frequency: 100,
      log_frequency: 10,
      render: false,
      save_model: true,
      use_gpu: true,
      verbose: true,
      // DQN特定参数 - 将从备份文件加载
      learning_rate: 0,
      gamma: 0,
      epsilon_start: 0,
      epsilon_final: 0,
      epsilon_decay: 0,
      batch_size: 0,
      memory_capacity: 0,
      target_update_frequency: 0,
      initial_learning: 0,
      beta_start: 0,
      beta_frames: 0
    })
    
    // 图表相关
    const availableMetricsFiles = ref([])
    const selectedEnvironment = ref('')
    const selectedMetrics = ref(['total_reward', 'best_reward'])
    const activeCharts = ref([])
    const showAddChartDialog = ref(false)
    const newChart = ref({
      title: '',
      metrics: [],
      type: 'line'
    })
    
    // 实时图表相关
    const realtimeChartInstance = ref(null)
    const realtimeChartRef = ref(null)
    const chartType = ref('line')
    const refreshInterval = ref(2000)
    const refreshTimer = ref(null)
    
    // 从store获取状态
    const username = computed(() => store.getters.username)
    const hasTrainingData = computed(() => trainingMetrics.value.length > 0)
    const currentEpisode = computed(() => {
      const lastMetric = trainingMetrics.value[trainingMetrics.value.length - 1]
      return lastMetric ? lastMetric.episode : 0
    })
    const avgReward = computed(() => {
      if (trainingMetrics.value.length === 0) return 0
      const recentMetrics = trainingMetrics.value.slice(-100)
      return recentMetrics.reduce((sum, m) => sum + (m.total_reward || 0), 0) / recentMetrics.length
    })
    const bestReward = computed(() => {
      if (trainingMetrics.value.length === 0) return 0
      return Math.max(...trainingMetrics.value.map(m => m.total_reward || 0))
    })
    const totalSteps = computed(() => {
      const lastMetric = trainingMetrics.value[trainingMetrics.value.length - 1]
      return lastMetric ? lastMetric.total_steps : 0
    })
    
    // 可用的指标选项
    const availableMetricsForChart = computed(() => [
      { value: 'total_reward', label: '总奖励' },
      { value: 'best_reward', label: '最佳奖励' },
      { value: 'average_reward', label: '平均奖励' },
      { value: 'epsilon', label: 'Epsilon值' },
      { value: 'average_loss', label: '平均损失' }
    ])
    
    // 根据当前选择的算法动态显示相关指标
    const availableMetricsForCurrentAlgorithm = computed(() => {
      // 优先从选择的训练记录文件路径判断算法类型
      let algorithm = 'dqn' // 默认值
      
      if (selectedEnvironment.value) {
        // 从文件路径中提取算法类型
        // 格式: SuperMarioBros-1-1-v0/PPO/20250914_221007_lr1e4_gamma99
        const pathParts = selectedEnvironment.value.split('/')
        if (pathParts.length >= 2) {
          const algorithmFromPath = pathParts[1].toLowerCase()
          if (algorithmFromPath === 'ppo' || algorithmFromPath === 'dqn') {
            algorithm = algorithmFromPath
          }
        }
      } else {
        // 如果没有选择训练记录，使用训练控制面板的算法选择
        algorithm = selectedAlgorithm.value?.toLowerCase() || 'dqn'
      }
      
      console.log('availableMetricsForCurrentAlgorithm: 当前算法:', algorithm)
      console.log('availableMetricsForCurrentAlgorithm: 选择的训练记录:', selectedEnvironment.value)
      
      // 基础指标（所有算法都有）
      const baseMetrics = [
        { key: 'total_reward', label: '总奖励' },
        { key: 'best_reward', label: '最佳奖励' },
        { key: 'average_reward', label: '平均奖励' },
        { key: 'average_loss', label: '平均损失' }
      ]
      
      // DQN特有指标
      const dqnMetrics = [
        { key: 'epsilon', label: 'Epsilon值' },
        { key: 'episode_length', label: '回合长度' },
        { key: 'total_steps', label: '总步数' },
        { key: 'learning_rate', label: '学习率' },
        { key: 'gamma', label: '折扣因子' }
      ]
      
      // PPO特有指标
      const ppoMetrics = [
        { key: 'episode_length', label: '回合长度' },
        { key: 'total_steps', label: '总步数' },
        { key: 'learning_rate', label: '学习率' },
        { key: 'gamma', label: '折扣因子' },
        { key: 'max_stage', label: '最大关卡' },
        { key: 'flag_get', label: '通关标志' },
        { key: 'avg_total_reward', label: '平均总奖励' },
        { key: 'policy_loss', label: '策略损失' },
        { key: 'value_loss', label: '价值损失' },
        { key: 'entropy', label: '熵值' }
      ]
      
      // 根据算法返回相应的指标
      let result
      if (algorithm === 'dqn') {
        result = [...baseMetrics, ...dqnMetrics]
      } else if (algorithm === 'ppo') {
        result = [...baseMetrics, ...ppoMetrics]
      } else {
        // 默认返回DQN指标
        result = [...baseMetrics, ...dqnMetrics]
      }
      
      console.log('availableMetricsForCurrentAlgorithm: 返回指标数量:', result.length)
      console.log('availableMetricsForCurrentAlgorithm: 指标列表:', result.map(m => m.key))
      return result
    })
    
    // 检查当前文件是否支持重置
    const isResetSupported = computed(() => {
      if (!currentFileInfo.value) return false
      const supportedResetTypes = ['constants', 'wrappers', 'model', 'replay_buffer', 'helpers', 'trainer', 'script']
      return supportedResetTypes.includes(currentFileInfo.value.type)
    })
    
    // 语言类名
    const languageClass = computed(() => {
      if (!currentFileInfo.value) return 'language-python'
      const fileType = currentFileInfo.value.type
      switch (fileType) {
        case 'constants':
        case 'script':
        case 'trainer':
        case 'model':
        case 'replay_buffer':
          return 'language-python'
        case 'wrappers':
          return 'language-python'
        default:
          return 'language-python'
      }
    })
    
    // 语法高亮的代码
    const highlightedCode = computed(() => {
      if (!currentFileContent.value) return ''
      
      try {
        // 使用 Prism.js 进行语法高亮
        const highlighted = Prism.highlight(
          currentFileContent.value,
          Prism.languages.python,
          'python'
        )
        return highlighted
      } catch (error) {
        console.error('语法高亮失败:', error)
        // 如果高亮失败，返回原始内容并转义HTML
        return currentFileContent.value
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;')
      }
    })
    
    // 监听备份管理对话框的打开，自动刷新备份列表
    watch(showBackupManager, (newVal) => {
      if (newVal && currentFileInfo.value) {
        refreshBackups()
      }
    })
    
    // 从备份文件初始化训练控制表单
    const initializeTrainingFormFromBackup = async () => {
      try {
        console.log('开始初始化训练控制表单...')
        
        // 首先尝试从最新的备份文件加载
        const res = await fetch(`/api/config-manager/${selectedAlgorithm.value.toLowerCase()}/mario`)
        console.log('API调用结果:', res.status, res.ok)
        
        if (res.ok) {
          const data = await res.json()
          console.log('API返回数据:', data)
          console.log('data.files类型:', typeof data.files, data.files)
          console.log('data.files是否为数组:', Array.isArray(data.files))
          
          // 检查数据结构
          if (data.files && typeof data.files === 'object') {
            // files是一个对象，不是数组，需要找到constants文件
            const algorithmConstantsFile = Object.values(data.files).find(file => file.type === 'constants' && file.category === 'algorithm')
            const gameConstantsFile = Object.values(data.files).find(file => file.type === 'constants' && file.category === 'game')
            
            if (algorithmConstantsFile && gameConstantsFile) {
              console.log('找到constants文件:', { algorithmConstantsFile, gameConstantsFile })
              // 合并两个文件的内容
              const combinedContent = algorithmConstantsFile.content + '\n' + gameConstantsFile.content
              await syncTrainingFormFromConstants(combinedContent)
              console.log('从最新备份文件初始化训练控制表单')
              return
            } else {
              console.log('未找到完整的constants文件:', { algorithmConstantsFile: !!algorithmConstantsFile, gameConstantsFile: !!gameConstantsFile })
            }
          } else {
            console.log('files不是对象:', typeof data.files)
          }
        } else {
          console.log('API调用失败:', res.status)
        }
        
        // 如果没有最新备份，从原始备份加载
        console.log('尝试从原始备份加载...')
        
        // 同时加载algorithm和game constants
        const [algorithmRes, gameRes] = await Promise.all([
          fetch('/api/config-manager/backup-original', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              algorithm: selectedAlgorithm.value.toLowerCase(),
              game: 'mario',
              fileType: 'constants',
              category: 'algorithm'
            })
          }),
          fetch('/api/config-manager/backup-original', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              algorithm: selectedAlgorithm.value.toLowerCase(),
              game: 'mario',
              fileType: 'constants',
              category: 'game'
            })
          })
        ])
        
        console.log('原始备份API调用结果:', { algorithm: algorithmRes.status, game: gameRes.status })
        
        if (algorithmRes.ok && gameRes.ok) {
          const [algorithmData, gameData] = await Promise.all([
            algorithmRes.json(),
            gameRes.json()
          ])
          
          console.log('原始备份API返回数据:', { algorithmData, gameData })
          
          if (algorithmData.success && algorithmData.content && gameData.success && gameData.content) {
            // 合并两个文件的内容
            const combinedContent = algorithmData.content + '\n' + gameData.content
            await syncTrainingFormFromConstants(combinedContent)
            console.log('从原始备份文件初始化训练控制表单')
            return
          }
        }
        
        console.warn('无法从备份文件加载，使用默认值')
      } catch (error) {
        console.error('初始化训练控制表单失败:', error)
      }
    }
    
    // 处理ResizeObserver错误
    onMounted(() => {
      // 抑制ResizeObserver错误
      const originalError = window.onerror
      window.onerror = (message, source, lineno, colno, error) => {
        if (message && message.includes('ResizeObserver loop completed with undelivered notifications')) {
          return true // 阻止错误显示
        }
        if (originalError) {
          return originalError(message, source, lineno, colno, error)
        }
        return false
      }
      
      // 抑制unhandledrejection错误
      const originalUnhandledRejection = window.onunhandledrejection
      window.onunhandledrejection = (event) => {
        if (event.reason && event.reason.message && 
            event.reason.message.includes('ResizeObserver loop completed with undelivered notifications')) {
          event.preventDefault()
          return true
        }
        if (originalUnhandledRejection) {
          return originalUnhandledRejection(event)
        }
        return false
      }
      
      // 清理函数
      return () => {
        window.onerror = originalError
        window.onunhandledrejection = originalUnhandledRejection
      }
    })
    
    // 监听训练控制表单的变化，自动同步到constants.py
    // 注意：这个监听器可能导致无限循环，暂时禁用
    // watch(trainingForm, (newVal, oldVal) => {
    //   // 避免初始化时的同步
    //   if (oldVal && currentFileInfo.value && 
    //       currentFileInfo.value.type === 'constants' && 
    //       currentFileInfo.value.category === 'algorithm') {
    //     syncConstantsFromTrainingForm()
    //   }
    // }, { deep: true })
    
    // 菜单选择处理
    const handleMenuSelect = (index) => {
      activeMenu.value = index
    }
    
    // 加载算法和游戏配置
    const loadConfigs = async () => {
      try {
        const res = await fetch('/api/algorithm-configs')
        const result = await res.json()
        if (result.success) {
          algorithmConfigs.value = result.algorithms || {}
          gameConfigs.value = result.games || {}
          
          console.log('loadConfigs: 加载的算法配置:', Object.keys(algorithmConfigs.value))
          
          // 设置默认值（只有在没有选择时才设置）
          if (Object.keys(algorithmConfigs.value).length > 0 && !selectedAlgorithm.value) {
            selectedAlgorithm.value = Object.keys(algorithmConfigs.value)[0]
            console.log('loadConfigs: 设置默认算法:', selectedAlgorithm.value)
          }
          if (Object.keys(gameConfigs.value).length > 0 && !selectedGame.value) {
            selectedGame.value = Object.keys(gameConfigs.value)[0]
          }
          
          console.log('loadConfigs: 最终算法选择:', selectedAlgorithm.value)
          
          // 初始化环境和动作空间
          updateGameConfig()
          updateAlgorithmConfig()
        }
      } catch (error) {
        console.error('加载配置失败:', error)
        ElMessage.error('加载算法配置失败')
      }
    }
    
    // 更新游戏配置
    const updateGameConfig = () => {
      const gameConfig = gameConfigs.value[selectedGame.value]
      if (gameConfig) {
        availableEnvironments.value = gameConfig.environments || []
        availableActionSpaces.value = gameConfig.action_spaces || {}
        
        // 设置默认环境
        if (availableEnvironments.value.length > 0) {
          trainingForm.value.environment = availableEnvironments.value[0]
        }
        // 设置默认动作空间
        if (Object.keys(availableActionSpaces.value).length > 0) {
          trainingForm.value.action_space = Object.keys(availableActionSpaces.value)[0]
        }
      }
    }
    
    // 更新算法配置
    const updateAlgorithmConfig = () => {
      const algorithmConfig = algorithmConfigs.value[selectedAlgorithm.value]
      if (algorithmConfig) {
        // 获取算法参数
        const params = algorithmConfig.parameters || {}
        algorithmParameters.value = Object.entries(params).map(([name, config]) => ({
          name,
          ...config
        }))
        
        // 设置算法参数的默认值
        algorithmParameters.value.forEach(param => {
          if (trainingForm.value[param.name] === undefined) {
            trainingForm.value[param.name] = param.default
          }
        })
      }
    }
    
    // 算法改变事件
    const onAlgorithmChange = () => {
      trainingForm.value.algorithm = selectedAlgorithm.value
      updateAlgorithmConfig()
    }
    
    // 游戏改变事件
    const onGameChange = () => {
      trainingForm.value.game = selectedGame.value
      updateGameConfig()
    }
    
    // 重置训练配置为默认值
    const resetTrainingConfig = async () => {
      try {
        ElMessageBox.confirm(
          '确定要重置算法和环境的常量配置为原始备份状态吗？这将覆盖当前的常量配置。',
          '重置常量配置确认',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }
        ).then(async () => {
          console.log('开始重置常量配置文件...')
          
          // 只重置常量配置文件
          const filesToReset = [
            { fileType: 'constants', category: 'algorithm' },
            { fileType: 'constants', category: 'game' }
          ]
          
          let successCount = 0
          let failCount = 0
          const failedFiles = []
          
          // 逐个重置文件
          for (const file of filesToReset) {
            try {
              const res = await fetch('/api/config-manager/reset-to-original', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                  algorithm: selectedAlgorithm.value.toLowerCase(),
                  game: 'mario',
                  fileType: file.fileType,
                  category: file.category
                })
              })
              
              const result = await res.json()
              if (result.success) {
                successCount++
                console.log(`成功重置 ${file.category}/${file.fileType}.py`)
        } else {
                failCount++
                failedFiles.push(`${file.category}/${file.fileType}.py: ${result.message}`)
                console.error(`重置失败 ${file.category}/${file.fileType}.py:`, result.message)
              }
            } catch (error) {
              failCount++
              failedFiles.push(`${file.category}/${file.fileType}.py: ${error.message}`)
              console.error(`重置失败 ${file.category}/${file.fileType}.py:`, error)
            }
          }
          
          // 显示结果
          if (successCount > 0) {
            let message = `成功重置 ${successCount} 个常量配置文件`
            if (failCount > 0) {
              message += `，${failCount} 个文件重置失败`
            }
            ElMessage.success(message)
            
            if (failedFiles.length > 0) {
              console.warn('重置失败的文件:', failedFiles)
            }
          } else {
            ElMessage.error('所有常量配置文件重置失败')
          }
          
          // 重新加载配置
          await loadConfigs()
          
          // 重新加载参数配置界面
          await loadAllConfigs()
          
          // 从备份文件初始化训练控制表单
          await initializeTrainingFormFromBackup()
          
          console.log('常量配置文件重置完成')
        }).catch(() => {
          // 用户取消
        })
      } catch (error) {
        console.error('重置配置失败:', error)
        ElMessage.error('重置配置失败')
      }
    }
    
    // 统一配置管理方法
    const loadAllConfigs = async () => {
      console.log('loadAllConfigs被调用')
      console.log('selectedAlgorithm.value:', selectedAlgorithm.value)
      console.log('selectedGame.value:', selectedGame.value)
      
      if (!selectedAlgorithm.value || !selectedGame.value) {
        ElMessage.warning('请先选择算法和游戏')
        return
      }
      
      try {
        const apiUrl = `/api/config-manager/${selectedAlgorithm.value.toLowerCase()}/${selectedGame.value.toLowerCase()}`
        console.log('API URL:', apiUrl)
        
        const res = await fetch(apiUrl)
        console.log('API响应状态:', res.status, res.ok)
        
        const result = await res.json()
        console.log('API返回结果:', result)
        
        if (result.success) {
          configFiles.value = Object.entries(result.files).map(([key, file]) => ({
            key,
            ...file,
            lastModified: new Date()
          }))
          
          console.log('加载的文件:', configFiles.value)
          console.log('文件数量:', configFiles.value.length)
          
          if (configFiles.value.length > 0) {
            activeFileTab.value = configFiles.value[0].key
            console.log('设置activeFileTab为:', activeFileTab.value)
            
            // 等待DOM更新完成，避免ResizeObserver错误
            await nextTick()
            setTimeout(async () => {
              await loadFileContent(configFiles.value[0])
            }, 100)
          }
          
          ElMessage.success(`配置加载成功，共 ${configFiles.value.length} 个文件`)
        } else {
          console.error('API返回失败:', result.message)
          ElMessage.error(result.message || '加载配置失败')
        }
      } catch (error) {
        console.error('加载配置失败:', error)
        ElMessage.error('加载配置失败')
      }
    }
    
    const loadFileContent = async (file) => {
      console.log('loadFileContent被调用，文件:', file)
      console.log('selectedAlgorithm.value:', selectedAlgorithm.value)
      console.log('selectedGame.value:', selectedGame.value)
      
      try {
        // 重新从服务器获取文件内容
        const res = await fetch(`/api/config-manager/${selectedAlgorithm.value.toLowerCase()}/${selectedGame.value.toLowerCase()}`)
        const result = await res.json()
        
        if (result.success) {
          const updatedFile = result.files[file.key]
          if (updatedFile) {
            // 更新configFiles数组中的内容
            const fileIndex = configFiles.value.findIndex(f => f.key === file.key)
            if (fileIndex !== -1) {
              configFiles.value[fileIndex] = {
                ...updatedFile,
                key: file.key,
                lastModified: new Date()
              }
            }
            
            // 使用 nextTick 确保DOM更新完成
            await nextTick()
            setTimeout(() => {
              currentFileContent.value = updatedFile.content
              currentFileInfo.value = {
                ...updatedFile,
                key: file.key
              }
              console.log('文件内容已从服务器重新加载:', updatedFile.type)
              console.log('文件内容长度:', updatedFile.content ? updatedFile.content.length : 0)
              console.log('文件内容前100字符:', updatedFile.content ? updatedFile.content.substring(0, 100) : '无内容')
            }, 50)
          }
        } else {
          // 如果API失败，使用缓存的内容
          currentFileContent.value = file.content
          currentFileInfo.value = file
        }
      } catch (error) {
        console.error('重新加载文件内容失败:', error)
        // 如果出错，使用缓存的内容
        currentFileContent.value = file.content
        currentFileInfo.value = file
      }
      
      console.log('当前文件信息已更新:', currentFileInfo.value)
    }
    
    const onFileTabClick = async (tab) => {
      console.log('点击的标签页:', tab)
      console.log('标签页名称:', tab.paneName)
      console.log('当前文件列表:', configFiles.value)
      
      const file = configFiles.value.find(f => f.key === tab.paneName)
      console.log('找到的文件:', file)
      
      if (file) {
        // 等待DOM更新完成，避免ResizeObserver错误
        await nextTick()
        setTimeout(async () => {
          await loadFileContent(file)
          console.log('已加载文件内容:', file.type, file.category)
        }, 50)
      } else {
        console.error('未找到对应的文件')
      }
    }
    
    const saveCurrentFile = async () => {
      if (!currentFileContent.value || !currentFileInfo.value) {
        ElMessage.warning('没有可保存的内容')
        return
      }
      
      try {
        const res = await fetch('/api/config-manager/save', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            algorithm: selectedAlgorithm.value.toLowerCase(),
            game: selectedGame.value.toLowerCase(),
            fileType: currentFileInfo.value.type,
            category: currentFileInfo.value.category,
            content: currentFileContent.value
          })
        })
        
        const result = await res.json()
        
        if (result.success) {
          ElMessage.success('备份保存成功')
          
          // 重置编辑状态
          isEditing.value = false
          originalContent.value = currentFileContent.value
          
          // 重新加载所有配置
          await loadAllConfigs()
          
          // 刷新备份列表
          await refreshBackups()
          
          // 如果保存的是constants.py文件，同步更新训练控制参数
          if (currentFileInfo.value.type === 'constants') {
            await syncTrainingFormFromConstants()
          }
        } else {
          ElMessage.error(result.message || '保存备份失败')
        }
      } catch (error) {
        console.error('保存备份失败:', error)
        ElMessage.error('保存备份失败')
      }
    }
    
    const resetToOriginal = async () => {
      if (!currentFileInfo.value) {
        ElMessage.warning('没有可重置的文件')
        return
      }
      
      // 检查是否支持重置
      const supportedResetTypes = ['constants', 'wrappers', 'model', 'replay_buffer', 'helpers', 'trainer', 'script']
      if (!supportedResetTypes.includes(currentFileInfo.value.type)) {
        ElMessage.warning(`文件类型 ${currentFileInfo.value.type} 不支持重置为原始状态`)
        return
      }
      
      try {
        await ElMessageBox.confirm(
          '确定要重置当前文件为原始状态吗？这将覆盖当前内容。',
          '重置确认',
          {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
            type: 'warning'
          }
        )
        
        const res = await fetch('/api/config-manager/reset-to-original', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            algorithm: selectedAlgorithm.value.toLowerCase(),
            game: selectedGame.value.toLowerCase(),
            fileType: currentFileInfo.value.type,
            category: currentFileInfo.value.category
          })
        })
        
        const result = await res.json()
        
        if (result.success) {
          ElMessage.success('文件已重置为原始状态')
          // 重新加载配置
          await loadAllConfigs()
          
          // 如果重置的是constants.py文件，同步更新训练控制参数
          if (currentFileInfo.value.type === 'constants') {
            await syncTrainingFormFromConstants()
          }
        } else {
          ElMessage.error(result.message || '重置为原始状态失败')
        }
      } catch (error) {
        if (error !== 'cancel') {
          console.error('重置为原始状态失败:', error)
          ElMessage.error('重置为原始状态失败')
        }
      }
    }
    
    const refreshBackups = async () => {
      if (!currentFileInfo.value) return
      
      try {
        const res = await fetch(`/api/config-manager/backups/${selectedAlgorithm.value.toLowerCase()}/${selectedGame.value.toLowerCase()}/${currentFileInfo.value.type}`)
        const result = await res.json()
        
        if (result.success) {
          backupList.value = result.backups
        } else {
          ElMessage.error(result.message || '获取备份列表失败')
        }
      } catch (error) {
        console.error('获取备份列表失败:', error)
        ElMessage.error('获取备份列表失败')
      }
    }
    
    const restoreBackup = async (backup) => {
      try {
        await ElMessageBox.confirm(
          `确定要恢复备份 "${backup.filename}" 吗？这将覆盖当前文件内容。`,
          '恢复备份确认',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )
        
        const res = await fetch('/api/config-manager/restore', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            algorithm: selectedAlgorithm.value.toLowerCase(),
            game: selectedGame.value.toLowerCase(),
            fileType: currentFileInfo.value.type,
            category: currentFileInfo.value.category,
            backupPath: backup.path
          })
        })
        
        const result = await res.json()
        
        if (result.success) {
          ElMessage.success('备份恢复成功')
          // 重新加载配置
          await loadAllConfigs()
          await refreshBackups()
          
          // 如果恢复的是constants.py文件，同步更新训练控制参数
          if (currentFileInfo.value.type === 'constants') {
            await syncTrainingFormFromConstants()
          }
        } else {
          ElMessage.error(result.message || '恢复备份失败')
        }
      } catch (error) {
        if (error !== 'cancel') {
          console.error('恢复备份失败:', error)
          ElMessage.error('恢复备份失败')
        }
      }
    }
    
    const deleteBackup = async (backup) => {
      try {
        await ElMessageBox.confirm(
          `确定要删除备份 "${backup.filename}" 吗？此操作不可恢复。`,
          '删除备份确认',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )
        
        const res = await fetch('/api/config-manager/delete-backup', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            backupPath: backup.path
          })
        })
        
        const result = await res.json()
        
        if (result.success) {
          ElMessage.success('备份删除成功')
          await refreshBackups()
        } else {
          ElMessage.error(result.message || '删除备份失败')
        }
      } catch (error) {
        if (error !== 'cancel') {
          console.error('删除备份失败:', error)
          ElMessage.error('删除备份失败')
        }
      }
    }
    
    // 删除所有版本备份（保留原始备份）
    const deleteAllBackups = async () => {
      try {
        if (!currentFileInfo.value) {
          ElMessage.warning('没有可删除的备份')
        return
      }
      
        // 过滤出版本备份（排除原始备份）
        const versionBackups = backupList.value.filter(backup => backup.operation === 'save')
        
        if (versionBackups.length === 0) {
          ElMessage.info('没有版本备份需要删除')
          return
        }
        
        await ElMessageBox.confirm(
          `确定要删除所有 ${versionBackups.length} 个版本备份吗？此操作不可恢复，原始备份将被保留。`,
          '删除所有备份确认',
          {
            confirmButtonText: '确定删除',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )
        
        // 逐个删除版本备份
        let deletedCount = 0
        let failedCount = 0
        
        for (const backup of versionBackups) {
          try {
            const res = await fetch('/api/config-manager/delete-backup', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                backupPath: backup.path
              })
            })
            
            const result = await res.json()
            
            if (result.success) {
              deletedCount++
            } else {
              failedCount++
              console.error(`删除备份失败: ${backup.filename}`, result.message)
            }
          } catch (error) {
            failedCount++
            console.error(`删除备份失败: ${backup.filename}`, error)
          }
        }
        
        if (deletedCount > 0) {
          ElMessage.success(`已删除 ${deletedCount} 个版本备份`)
          await refreshBackups()
        }
        
        if (failedCount > 0) {
          ElMessage.warning(`${failedCount} 个备份删除失败`)
        }
        
      } catch (error) {
        if (error !== 'cancel') {
          console.error('删除所有备份失败:', error)
          ElMessage.error('删除所有备份失败')
        }
      }
    }
    
    const getFileDisplayName = (file) => {
      if (!file) return ''
      const typeMap = {
        constants: '常量配置',
        trainer: '训练器',
        model: '模型',
        replay_buffer: '经验回放',
        helpers: '辅助函数',
        script: '训练脚本',
        wrappers: '环境包装器'
      }
      const categoryPrefix = file.category === 'algorithm' ? '算法' : '游戏'
      const typeName = typeMap[file.type] || file.type
      return `${categoryPrefix} - ${typeName}`
    }
    
    const formatFileTime = (time) => {
      if (!time) return ''
      return new Date(time).toLocaleString()
    }
    
    const formatCode = () => {
      // 简单的代码格式化
      ElMessage.info('代码格式化功能待实现')
    }
    
    // 代码编辑相关函数
    const onCodeChange = () => {
      isEditing.value = true
    }
    
    const onCodeKeydown = (event) => {
      // 处理Tab键缩进
      if (event.key === 'Tab') {
        event.preventDefault()
        const textarea = event.target
        const start = textarea.selectionStart
        const end = textarea.selectionEnd
        const value = textarea.value
        const newValue = value.substring(0, start) + '    ' + value.substring(end)
        
        currentFileContent.value = newValue
        
        // 设置光标位置
        nextTick(() => {
          textarea.selectionStart = textarea.selectionEnd = start + 4
        })
      }
      
      // 处理Ctrl+S保存
      if (event.ctrlKey && event.key === 's') {
        event.preventDefault()
        saveCurrentFile()
      }
    }
    
    // 滚动同步
    const onTextareaScroll = (event) => {
      const textarea = event.target
      const pre = textarea.parentElement.querySelector('pre')
      if (pre) {
        pre.scrollTop = textarea.scrollTop
        pre.scrollLeft = textarea.scrollLeft
      }
    }
    
    const startEditing = () => {
      originalContent.value = currentFileContent.value
      isEditing.value = true
    }
    
    const cancelEditing = () => {
      currentFileContent.value = originalContent.value
      isEditing.value = false
    }
    
    
    const getLanguageClass = () => {
      return languageClass.value
    }
    
    
    // 监听文件内容变化，重新高亮
    watch([currentFileContent, syntaxHighlight], () => {
      if (syntaxHighlight.value) {
        nextTick(() => {
          // 强制重新渲染语法高亮
          const codeElement = document.querySelector('.syntax-highlight-editor code')
          if (codeElement) {
            codeElement.innerHTML = highlightedCode.value
          }
        })
      }
    })
    
    const onConfigAlgorithmChange = () => {
      // 清空当前文件内容
      currentFileContent.value = ''
      currentFileInfo.value = null
      configFiles.value = []
    }
    
    const onConfigGameChange = () => {
      // 清空当前文件内容
      currentFileContent.value = ''
      currentFileInfo.value = null
      configFiles.value = []
    }
    
    // 从constants.py同步参数到训练控制表单
    const syncTrainingFormFromConstants = async (content = null) => {
      try {
        let constantsContent = content
        
        // 如果没有提供内容，从当前文件加载
        if (!constantsContent) {
          // 重新加载算法配置以获取最新的constants.py内容
          await loadConfigs()
          
          // 解析algorithm constants.py文件内容，提取参数值
          const algorithmConstantsFile = configFiles.value.find(file => 
            file.type === 'constants' && file.category === 'algorithm'
          )
          
          // 解析game constants.py文件内容，提取动作空间和环境参数
          const gameConstantsFile = configFiles.value.find(file => 
            file.type === 'constants' && file.category === 'game'
          )
          
          if (!algorithmConstantsFile) {
            ElMessage.warning('未找到算法constants配置文件')
            return
          }
          
          if (!gameConstantsFile) {
            ElMessage.warning('未找到游戏constants配置文件')
            return
          }
          
          // 合并两个文件的内容
          constantsContent = algorithmConstantsFile.content + '\n' + gameConstantsFile.content
        }
        
        // 解析Python常量文件，提取参数值
        const parseConstants = (content) => {
            const params = {}
            const lines = content.split('\n')
            
            for (const line of lines) {
              const trimmedLine = line.trim()
              if (trimmedLine && !trimmedLine.startsWith('#') && !trimmedLine.startsWith('"""')) {
                const match = trimmedLine.match(/^(\w+)\s*=\s*(.+)$/)
                if (match) {
                  const [, key, value] = match
                  try {
                    // 尝试解析不同类型的值
                    if (value.includes('e-') || value.includes('E-')) {
                      // 科学计数法
                      params[key] = parseFloat(value)
                    } else if (value.includes('.')) {
                      // 浮点数
                      params[key] = parseFloat(value)
                    } else if (value === 'True' || value === 'False') {
                      // 布尔值
                      params[key] = value === 'True'
                    } else if (!isNaN(parseInt(value))) {
                      // 整数
                      params[key] = parseInt(value)
                    } else if (value.startsWith("'") && value.endsWith("'") || value.startsWith('"') && value.endsWith('"')) {
                      // 字符串
                      params[key] = value.slice(1, -1)
                    } else {
                      // 其他情况，尝试直接解析
                      params[key] = value
                    }
                  } catch (e) {
                    // 解析失败，保持原值
                    console.warn(`无法解析参数 ${key}: ${value}`)
                  }
                }
              }
            }
            return params
          }
          
        const parsedParams = parseConstants(constantsContent)
        console.log('解析出的参数:', parsedParams)
        console.log('ACTION_SPACE值:', parsedParams['ACTION_SPACE'])
        console.log('ENVIRONMENT值:', parsedParams['ENVIRONMENT'])
        
        // 更新trainingForm中的对应参数
        // 根据当前选择的算法使用不同的参数映射
        let paramMapping = {}
        
        if (selectedAlgorithm.value.toLowerCase() === 'dqn') {
          paramMapping = {
            'NUM_EPISODES': 'episodes',
            'LEARNING_RATE': 'learning_rate',
            'GAMMA': 'gamma',
            'EPSILON_START': 'epsilon_start',
            'EPSILON_FINAL': 'epsilon_final',
            'EPSILON_DECAY': 'epsilon_decay',
            'BATCH_SIZE': 'batch_size',
            'MEMORY_CAPACITY': 'memory_capacity',
            'TARGET_UPDATE_FREQUENCY': 'target_update_frequency',
            'INITIAL_LEARNING': 'initial_learning',
            'BETA_START': 'beta_start',
            'BETA_FRAMES': 'beta_frames',
            'RENDER': 'render',
            'SAVE_MODEL': 'save_model',
            'USE_GPU': 'use_gpu',
            'VERBOSE': 'verbose',
            // 从game constants.py读取的参数
            'DEFAULT_ENVIRONMENT': 'environment',
            'DEFAULT_ACTION_SPACE': 'action_space',
            'MAX_STEPS_PER_EPISODE': 'max_steps_per_episode'
          }
        } else if (selectedAlgorithm.value.toLowerCase() === 'ppo') {
          paramMapping = {
            'NUM_EPISODES': 'episodes',
            'LEARNING_RATE': 'learning_rate',
            'GAMMA': 'gamma',
            'LAMBDA_GAE': 'lambda_gae',
            'CLIP_RATIO': 'clip_eps',  // constants.py使用CLIP_RATIO，训练表单使用clip_eps
            'VALUE_LOSS_COEF': 'vf_coef',  // constants.py使用VALUE_LOSS_COEF，训练表单使用vf_coef
            'ENTROPY_COEF': 'ent_coef',  // constants.py使用ENTROPY_COEF，训练表单使用ent_coef
            'MAX_GRAD_NORM': 'max_grad_norm',
            'PPO_EPOCHS': 'epochs',  // constants.py使用PPO_EPOCHS，训练表单使用epochs
            'BATCH_SIZE': 'minibatch_size',  // constants.py使用BATCH_SIZE，训练表单使用minibatch_size
            'NUM_ENV': 'num_env',
            'ROLLOUT_STEPS': 'rollout_steps',
            'SAVE_FREQUENCY': 'save_frequency',
            'LOG_FREQUENCY': 'log_frequency',
            'RENDER': 'render',
            'SAVE_MODEL': 'save_model',
            'USE_GPU': 'use_gpu',
            'VERBOSE': 'verbose',
            // 从game constants.py读取的参数
            'DEFAULT_ENVIRONMENT': 'environment',
            'DEFAULT_ACTION_SPACE': 'action_space',
            'MAX_STEPS_PER_EPISODE': 'max_steps_per_episode'
          }
        }
          
          let updatedCount = 0
          let skippedCount = 0
          let skippedParams = []
          Object.entries(paramMapping).forEach(([constantsKey, formKey]) => {
            if (parsedParams[constantsKey] !== undefined) {
              const oldValue = trainingForm.value[formKey]
              const newValue = parsedParams[constantsKey]
              
              // 特殊处理action_space：优先使用DEFAULT_ACTION_SPACE，如果不存在则使用ACTION_SPACE
              if (formKey === 'action_space') {
                const defaultActionSpace = parsedParams['DEFAULT_ACTION_SPACE']
                const actionSpace = parsedParams['ACTION_SPACE']
                const finalValue = defaultActionSpace !== undefined ? defaultActionSpace : actionSpace
                
                if (finalValue !== undefined) {
                  console.log(`同步动作空间参数: ${formKey} ${oldValue} -> ${finalValue}`)
                  console.log('动作空间同步详情:', { 
                    DEFAULT_ACTION_SPACE: defaultActionSpace, 
                    ACTION_SPACE: actionSpace, 
                    finalValue 
                  })
                  trainingForm.value[formKey] = finalValue
                  updatedCount++
                }
                return
              }
              
              // 检查值是否在UI组件的有效范围内
              const paramConfig = algorithmParameters.value.find(p => p.name === formKey)
              if (paramConfig) {
                if (newValue < paramConfig.min || newValue > paramConfig.max) {
                  const skipInfo = `${formKey}=${newValue} (范围: ${paramConfig.min}-${paramConfig.max})`
                  console.warn(`参数 ${skipInfo} 超出UI组件范围，跳过同步`)
                  skippedParams.push(skipInfo)
                  skippedCount++
                  return
                }
              }
              
              console.log(`同步参数: ${formKey} ${oldValue} -> ${newValue}`)
              trainingForm.value[formKey] = newValue
              updatedCount++
            }
          })
          
          if (updatedCount > 0) {
            let message = `已同步 ${updatedCount} 个参数到训练控制表单`
            if (skippedCount > 0) {
              message += `，跳过 ${skippedCount} 个超出范围的参数: ${skippedParams.join(', ')}`
            }
            ElMessage.success(message)
            console.log('同步的参数:', Object.entries(paramMapping)
              .filter(([constantsKey]) => parsedParams[constantsKey] !== undefined)
              .map(([constantsKey, formKey]) => `${constantsKey} -> ${formKey}: ${parsedParams[constantsKey]}`)
            )
          } else if (skippedCount > 0) {
            ElMessage.warning(`跳过 ${skippedCount} 个超出范围的参数: ${skippedParams.join(', ')}`)
          } else {
            ElMessage.warning('未找到可同步的参数')
          }
      } catch (error) {
        console.error('同步参数失败:', error)
        ElMessage.error('同步参数失败')
      }
    }
    
    // 从训练控制表单同步参数到constants.py
    const syncConstantsFromTrainingForm = async () => {
      try {
        // 确保已加载配置
        if (!selectedAlgorithm.value || !selectedGame.value) {
          ElMessage.warning('请先选择算法和游戏')
          return
        }
        
        // 加载当前的constants.py文件内容
        await loadAllConfigs()
        
        const constantsFile = configFiles.value.find(file => 
          file.type === 'constants' && file.category === 'algorithm'
        )
        
        if (!constantsFile) {
          ElMessage.warning('未找到constants.py文件')
          return
        }
        
        let content = constantsFile.content
        if (!content) {
          ElMessage.warning('constants.py文件内容为空')
          return
        }
        
        // 参数映射：从训练表单字段到constants.py中的变量名
        // 根据当前选择的算法使用不同的参数映射
        let algorithmParamMapping = {}
        
        if (selectedAlgorithm.value.toLowerCase() === 'dqn') {
          algorithmParamMapping = {
            'episodes': 'NUM_EPISODES',
            'learning_rate': 'LEARNING_RATE',
            'gamma': 'GAMMA',
            'epsilon_start': 'EPSILON_START',
            'epsilon_final': 'EPSILON_FINAL',
            'epsilon_decay': 'EPSILON_DECAY',
            'batch_size': 'BATCH_SIZE',
            'memory_capacity': 'MEMORY_CAPACITY',
            'target_update_frequency': 'TARGET_UPDATE_FREQUENCY',
            'initial_learning': 'INITIAL_LEARNING',
            'beta_start': 'BETA_START',
            'beta_frames': 'BETA_FRAMES',
            'render': 'RENDER',
            'save_model': 'SAVE_MODEL',
            'use_gpu': 'USE_GPU',
            'verbose': 'VERBOSE'
          }
        } else if (selectedAlgorithm.value.toLowerCase() === 'ppo') {
          algorithmParamMapping = {
            'episodes': 'NUM_EPISODES',
            'learning_rate': 'LEARNING_RATE',
            'gamma': 'GAMMA',
            'lambda_gae': 'LAMBDA_GAE',
            'clip_eps': 'CLIP_RATIO',  // 训练表单使用clip_eps，constants.py使用CLIP_RATIO
            'vf_coef': 'VALUE_LOSS_COEF',  // 训练表单使用vf_coef，constants.py使用VALUE_LOSS_COEF
            'ent_coef': 'ENTROPY_COEF',  // 训练表单使用ent_coef，constants.py使用ENTROPY_COEF
            'max_grad_norm': 'MAX_GRAD_NORM',
            'epochs': 'PPO_EPOCHS',  // 训练表单使用epochs，constants.py使用PPO_EPOCHS
            'minibatch_size': 'BATCH_SIZE',  // 训练表单使用minibatch_size，constants.py使用BATCH_SIZE
            'num_env': 'NUM_ENV',
            'rollout_steps': 'ROLLOUT_STEPS',
            'save_frequency': 'SAVE_FREQUENCY',
            'log_frequency': 'LOG_FREQUENCY',
            'render': 'RENDER',
            'save_model': 'SAVE_MODEL',
            'use_gpu': 'USE_GPU',
            'verbose': 'VERBOSE'
          }
        }
        
        const gameParamMapping = {
          'environment': 'DEFAULT_ENVIRONMENT',
          'action_space': 'DEFAULT_ACTION_SPACE',
          'max_steps_per_episode': 'MAX_STEPS_PER_EPISODE'
        }
        
        let updatedContent = content
        let updatedCount = 0
        
        // 更新算法参数
        Object.entries(algorithmParamMapping).forEach(([formKey, constantsKey]) => {
          const value = trainingForm.value[formKey]
          if (value !== undefined) {
            const regex = new RegExp(`(${constantsKey}\\s*=\\s*)[^\\n]+`, 'g')
            let newValue
            if (typeof value === 'string') {
              newValue = `"${value}"`
            } else if (typeof value === 'boolean') {
              newValue = value ? 'True' : 'False'
            } else {
              newValue = value
            }
            const replacement = `$1${newValue}`
            
            if (regex.test(updatedContent)) {
              updatedContent = updatedContent.replace(regex, replacement)
              updatedCount++
            }
          }
        })
        
        // 保存算法参数到algorithm constants.py
        if (updatedCount > 0) {
          const res = await fetch('/api/config-manager/save', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              algorithm: selectedAlgorithm.value.toLowerCase(),
              game: selectedGame.value.toLowerCase(),
              fileType: 'constants',
              category: 'algorithm',
              content: updatedContent
            })
          })
          
          const result = await res.json()
          if (!result.success) {
            ElMessage.error('保存算法参数失败: ' + result.message)
            return
          }
        }
        
        // 更新游戏参数
        const gameConstantsFile = configFiles.value.find(file => 
          file.type === 'constants' && file.category === 'game'
        )
        
        if (gameConstantsFile) {
          let gameContent = gameConstantsFile.content
          let gameUpdatedCount = 0
          
          Object.entries(gameParamMapping).forEach(([formKey, constantsKey]) => {
            const value = trainingForm.value[formKey]
            if (value !== undefined) {
              // 特殊处理action_space，需要同时更新DEFAULT_ACTION_SPACE
              if (formKey === 'action_space') {
                const defaultActionSpaceRegex = new RegExp(`(DEFAULT_ACTION_SPACE\\s*=\\s*)[^\\n]+`, 'g')
                const defaultActionSpaceReplacement = `$1"${value}"`
                if (defaultActionSpaceRegex.test(gameContent)) {
                  gameContent = gameContent.replace(defaultActionSpaceRegex, defaultActionSpaceReplacement)
                  gameUpdatedCount++
                }
              } else {
                const regex = new RegExp(`(${constantsKey}\\s*=\\s*)[^\\n]+`, 'g')
                let newValue
                if (typeof value === 'string') {
                  newValue = `"${value}"`
                } else if (typeof value === 'boolean') {
                  newValue = value ? 'True' : 'False'
                } else {
                  newValue = value
                }
                const replacement = `$1${newValue}`
                
                if (regex.test(gameContent)) {
                  gameContent = gameContent.replace(regex, replacement)
                  gameUpdatedCount++
                }
              }
            }
          })
          
          // 保存游戏参数到game constants.py
          if (gameUpdatedCount > 0) {
            const gameRes = await fetch('/api/config-manager/save', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                algorithm: selectedAlgorithm.value.toLowerCase(),
                game: selectedGame.value.toLowerCase(),
                fileType: 'constants',
                category: 'game',
                content: gameContent
              })
            })
            
            const gameResult = await gameRes.json()
            if (!gameResult.success) {
              ElMessage.error('保存游戏参数失败: ' + gameResult.message)
              return
            }
          }
        }
        
        ElMessage.success('参数同步完成')
        
        // 重新加载constants.py文件内容
        console.log('重新加载constants.py文件内容...')
        
        // 确保configFiles已加载
        if (configFiles.value.length === 0) {
          await loadAllConfigs()
        }
        
        // 找到constants文件并重新加载
        const reloadConstantsFile = configFiles.value.find(file => 
          file.type === 'constants' && file.category === 'algorithm'
        )
        
        if (reloadConstantsFile) {
          await loadFileContent(reloadConstantsFile)
          console.log('constants.py文件内容已重新加载')
        } else {
          console.warn('未找到constants文件，无法重新加载')
        }
        
      } catch (error) {
        console.error('同步参数到constants.py失败:', error)
      }
    }
    
    // 参数文件管理函数 (保留兼容性)
    const loadConfigFile = async () => {
      if (!selectedAlgorithm.value) {
        ElMessage.warning('请先选择算法')
        return
      }
      
      try {
        const res = await fetch(`/api/algorithm-config-file/${selectedAlgorithm.value.toLowerCase()}`)
        const result = await res.json()
        
        if (result.success) {
          configFileContent.value = result.content
          configFilePath.value = result.path
          configFileInfo.value = {
            algorithm: result.algorithm,
            path: result.path
          }
          ElMessage.success('参数文件加载成功')
        } else {
          ElMessage.error(result.message || '加载参数文件失败')
        }
      } catch (error) {
        console.error('加载参数文件失败:', error)
        ElMessage.error('加载参数文件失败')
      }
    }
    
    const loadScriptFile = async () => {
      if (!selectedAlgorithm.value) {
        ElMessage.warning('请先选择算法')
        return
      }
      
      try {
        const res = await fetch(`/api/algorithm-script/${selectedAlgorithm.value.toLowerCase()}`)
        const result = await res.json()
        
        if (result.success) {
          configFileContent.value = result.content
          configFilePath.value = result.path
          configFileInfo.value = {
            algorithm: result.algorithm,
            path: result.path
          }
          configFileType.value = 'script'
          ElMessage.success('训练脚本加载成功')
        } else {
          ElMessage.error(result.message || '加载训练脚本失败')
        }
      } catch (error) {
        console.error('加载训练脚本失败:', error)
        ElMessage.error('加载训练脚本失败')
      }
    }
    
    const saveConfigFile = async () => {
      if (!configFileContent.value || !selectedAlgorithm.value) {
        ElMessage.warning('没有可保存的内容')
        return
      }
      
      try {
        const res = await fetch('/api/save-algorithm-config-file', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            algorithm: selectedAlgorithm.value.toLowerCase(),
            content: configFileContent.value
          })
        })
        
        const result = await res.json()
        
        if (result.success) {
          ElMessage.success('参数文件保存成功')
          if (result.backupPath) {
            ElMessage.info(`原文件已备份到: ${result.backupPath}`)
          }
        } else {
          ElMessage.error(result.message || '保存参数文件失败')
        }
      } catch (error) {
        console.error('保存参数文件失败:', error)
        ElMessage.error('保存参数文件失败')
      }
    }
    
    const resetConfigFile = () => {
      ElMessageBox.confirm(
        '确定要重置参数文件为默认值吗？这将覆盖当前内容。',
        '重置确认',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }
      ).then(() => {
        loadConfigFile()
        ElMessage.success('已重置为默认参数')
      }).catch(() => { void 0 })
    }
    
    // 训练控制
    const startTraining = async () => {
      ElMessageBox.confirm(
        `确定要开始${selectedAlgorithm.value}训练吗？\n算法: ${selectedAlgorithm.value}\n游戏: ${selectedGame.value}\n环境: ${trainingForm.value.environment}\n动作空间: ${trainingForm.value.action_space}\n训练轮数: ${trainingForm.value.episodes}`,
        `开始${selectedAlgorithm.value}训练`,
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'info'
        }
      ).then(async () => {
        try {
          // 清空之前的日志和指标
          trainingLogs.value = []
          trainingMetrics.value = []
          isTraining.value = true
          
          const payload = {
            algorithm: selectedAlgorithm.value,
            game: selectedGame.value,
            environment: trainingForm.value.environment,
            action_space: trainingForm.value.action_space,
            episodes: trainingForm.value.episodes,
            max_steps_per_episode: trainingForm.value.max_steps_per_episode,
            save_frequency: trainingForm.value.save_frequency,
            log_frequency: trainingForm.value.log_frequency,
            render: trainingForm.value.render,
            save_model: trainingForm.value.save_model,
            use_gpu: trainingForm.value.use_gpu,
            verbose: trainingForm.value.verbose
          }
          
          // 添加算法特定参数
          algorithmParameters.value.forEach(param => {
            payload[param.name] = trainingForm.value[param.name]
          })
          
          const res = await fetch('/api/start-training', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
          })
          
          const result = await res.json()
          if (!result.success) {
            throw new Error(result.message || '启动失败')
          }
          
          currentProcessId.value = result.processId
          trainingLogs.value.push(`[${new Date().toLocaleTimeString()}] ${selectedAlgorithm.value}训练已启动`)
          
          // 开始轮询日志
          startLogPolling()
          
          // 自动同步训练参数到配置文件
          await syncConstantsFromTrainingForm()
          
          // 立即加载可用的指标文件并启动自动刷新
          await loadAvailableMetricsFiles()
          startAutoRefresh()
          
          ElMessage.success(`${selectedAlgorithm.value}训练已开始`)
          if (activeMenu.value !== 'training') {
            activeMenu.value = 'training'
          }
        } catch (e) {
          isTraining.value = false
          ElMessage.error(`启动${selectedAlgorithm.value}训练失败: ` + (e.message || '未知错误'))
        }
      }).catch(() => { void 0 })
    }
    
    const stopTraining = () => {
      ElMessageBox.confirm(
        `确定要停止当前${selectedAlgorithm.value}训练吗？`,
        '停止训练',
        {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }
      ).then(async () => {
        try {
          if (currentProcessId.value) {
            await fetch(`/api/stop/${currentProcessId.value}`, { method: 'POST' })
          }
        } catch (_) { void 0 }
        isTraining.value = false
        trainingLogs.value.push(`[${new Date().toLocaleTimeString()}] ${selectedAlgorithm.value}训练已手动停止`)
        ElMessage.info(`${selectedAlgorithm.value}训练已停止`)
      }).catch(() => { void 0 })
    }
    
    const clearLogs = () => {
      trainingLogs.value = []
      ElMessage.info('日志已清空')
    }
    
    // 日志轮询
    let logPollingInterval = null
    
    const startLogPolling = () => {
      if (logPollingInterval) {
        clearInterval(logPollingInterval)
      }
      
      logPollingInterval = setInterval(async () => {
        if (!isTraining.value || !currentProcessId.value) {
          clearInterval(logPollingInterval)
          return
        }
        
        try {
          const res = await fetch(`/api/logs/${currentProcessId.value}`)
          const result = await res.json()
          if (result.success) {
            // 更新日志
            trainingLogs.value = result.logs || []
            
            // 更新指标
            if (result.metrics && result.metrics.length > 0) {
              trainingMetrics.value = result.metrics
              updateCharts()
            }
            
            // 检查是否完成
            if (result.completed) {
              isTraining.value = false
              clearInterval(logPollingInterval)
              trainingLogs.value.push(`[${new Date().toLocaleTimeString()}] DQN训练完成`)
              ElMessage.success('DQN训练完成！')
            }
          }
        } catch (error) {
          console.error('获取训练日志失败:', error)
        }
      }, 2000)
    }
    
    // 加载可用的指标文件
    const loadAvailableMetricsFiles = async () => {
      try {
        const res = await fetch('/api/training-metrics-files')
        const result = await res.json()
        if (result.success) {
          // 处理新的文件结构
          const processedFiles = result.files.map(file => {
            if (file.type === 'experiment') {
              // 新格式：显示实验路径
              return {
                name: file.name,
                path: file.path,
                size: file.size,
                modified: file.modified,
                type: file.type,
                displayName: file.name // 使用实验路径作为显示名称
              }
            } else {
              // 旧格式：保持兼容
              return {
                name: file.name,
                path: file.path,
                size: file.size,
                modified: file.modified,
                type: file.type || 'legacy',
                displayName: file.name
              }
            }
          })
          
          availableMetricsFiles.value = processedFiles
          
          // 自动选择最新的文件
          if (processedFiles.length > 0) {
            const latestFile = processedFiles[0] // 已经按修改时间排序
            
            // 如果没有选择文件，或者正在训练且选择了旧文件，则自动切换到最新文件
            if (!selectedEnvironment.value || (isTraining.value && selectedEnvironment.value !== latestFile.name)) {
              selectedEnvironment.value = latestFile.name
              await loadMetricsData()
              
              // 只在第一次选择时显示消息，避免训练期间频繁提示
              if (!isTraining.value) {
                ElMessage.success(`已自动加载最新的训练记录: ${latestFile.displayName}`)
              }
            }
          }
        }
      } catch (error) {
        console.error('加载指标文件列表失败:', error)
      }
    }
    
    // 加载指标数据
    const loadMetricsData = async () => {
      if (!selectedEnvironment.value) {
        console.log('loadMetricsData: 没有选择环境')
        return
      }
      
      // 找到对应的文件信息
      const selectedFile = availableMetricsFiles.value.find(file => file.name === selectedEnvironment.value)
      if (!selectedFile) {
        console.error('loadMetricsData: 找不到选择的文件')
        return
      }
      
      const environmentName = selectedEnvironment.value.split('/')[0]
      const filePath = selectedFile.path
      console.log('loadMetricsData: 开始加载数据，环境:', environmentName, '文件路径:', filePath)
      
      try {
        const res = await fetch(`/api/training-metrics/${environmentName}?filePath=${encodeURIComponent(filePath)}`)
        const result = await res.json()
        console.log('loadMetricsData: API响应:', result)
        
        if (result.success) {
          trainingMetrics.value = result.metrics
          console.log('loadMetricsData: 数据加载成功，指标数量:', result.metrics.length)
          console.log('loadMetricsData: 前3个指标:', result.metrics.slice(0, 3))
          
          // 确保图表实例存在后再更新
          if (!realtimeChartInstance.value) {
            console.log('loadMetricsData: 图表实例不存在，尝试初始化')
            initRealtimeChart()
          } else {
            updateRealtimeChart()
          }
          
          // 始终启动自动刷新
          startAutoRefresh()
        } else {
          console.error('loadMetricsData: API返回失败:', result.message)
        }
      } catch (error) {
        console.error('加载指标数据失败:', error)
      }
    }
    
    // 实时图表相关方法
    const initRealtimeChart = () => {
      console.log('initRealtimeChart 被调用')
      console.log('realtimeChartRef.value:', !!realtimeChartRef.value)
      console.log('realtimeChartInstance.value:', !!realtimeChartInstance.value)
      
      if (realtimeChartRef.value && !realtimeChartInstance.value) {
        const container = realtimeChartRef.value
        
        // 强制设置容器尺寸
        container.style.width = '100%'
        container.style.height = '400px'
        container.style.minHeight = '400px'
        container.style.display = 'block'
        
        // 等待下一个事件循环，确保样式生效
        setTimeout(() => {
          try {
            // 禁用 ECharts 的 ResizeObserver
            realtimeChartInstance.value = echarts.init(container, null, {
              renderer: 'canvas',
              useDirtyRect: false,
              width: 600,
              height: 300
            })
            
            // 禁用图表的自动 resize
            if (realtimeChartInstance.value) {
              realtimeChartInstance.value.getZr().off('resize')
            }
            
            console.log('图表初始化成功')
            
            // 延迟更新图表
            setTimeout(() => {
              updateRealtimeChart()
            }, 100)
          } catch (error) {
            console.error('图表初始化失败:', error)
          }
        }, 100)
        
      } else if (realtimeChartInstance.value) {
        console.log('图表实例已存在，直接更新')
        updateRealtimeChart()
      } else {
        console.log('图表容器或实例不存在')
      }
    }
    
    const updateRealtimeChart = () => {
      console.log('updateRealtimeChart 被调用')
      console.log('realtimeChartInstance.value:', !!realtimeChartInstance.value)
      console.log('trainingMetrics.value.length:', trainingMetrics.value.length)
      
      if (!realtimeChartInstance.value || trainingMetrics.value.length === 0) {
        console.log('图表更新被跳过 - 缺少实例或数据')
        return
      }
      
      // 按episode排序数据
      const sortedMetrics = [...trainingMetrics.value].sort((a, b) => a.episode - b.episode)
      const episodes = sortedMetrics.map(m => m.episode)
      const series = []
      console.log('episodes:', episodes.length)
      console.log('episode range:', Math.min(...episodes), 'to', Math.max(...episodes))
      
      // 为每个选中的指标创建系列
      selectedMetrics.value.forEach(metricKey => {
        const data = sortedMetrics.map(m => m[metricKey] || 0)
        const metricLabel = getMetricLabel(metricKey)
        console.log(`指标 ${metricKey} (${metricLabel}):`, data.slice(0, 5), '...')
        
        series.push({
          name: metricLabel,
          type: chartType.value,
          data: data,
          smooth: chartType.value === 'line',
          symbol: chartType.value === 'scatter' ? 'circle' : 'none',
          symbolSize: chartType.value === 'scatter' ? 6 : 0,
          lineStyle: {
            width: 2
          },
          itemStyle: {
            borderWidth: 1
          }
        })
      })
      
      console.log('创建的系列数量:', series.length)
      console.log('系列数据:', series.map(s => ({ name: s.name, dataLength: s.data.length })))
      
      const option = {
        title: {
          text: '训练指标实时监控',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold'
          }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'none'
          },
          formatter: function(params) {
            let result = `Episode ${params[0].axisValue}<br/>`
            params.forEach(param => {
              result += `${param.seriesName}: ${param.value.toFixed(2)}<br/>`
            })
            return result
          }
        },
        legend: {
          data: series.map(s => s.name),
          top: 40,
          type: 'scroll'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: '15%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: episodes,
          name: 'Episode',
          nameLocation: 'middle',
          nameGap: 30
        },
        yAxis: {
          type: 'value',
          name: 'Value',
          nameLocation: 'middle',
          nameGap: 50
        },
        series: series,
        animation: true,
        animationDuration: 300
      }
      
      console.log('设置图表选项:', option)
      
      // 清除之前的渲染状态，防止虚线残留
      realtimeChartInstance.value.clear()
      
      // 设置新的图表选项
      realtimeChartInstance.value.setOption(option, true)
      console.log('图表选项设置完成')
    }
    
    const getMetricLabel = (metricKey) => {
      const labels = {
        'total_reward': '总奖励',
        'best_reward': '最佳奖励', 
        'average_reward': '平均奖励',
        'epsilon': 'Epsilon值',
        'average_loss': '平均损失',
        'episode_length': '回合长度',
        'total_steps': '总步数',
        'learning_rate': '学习率',
        'gamma': '折扣因子',
        'max_stage': '最大关卡',
        'flag_get': '通关标志',
        'avg_total_reward': '平均总奖励',
        'policy_loss': '策略损失',
        'value_loss': '价值损失',
        'entropy': '熵值'
      }
      return labels[metricKey] || metricKey
    }
    
    const startAutoRefresh = () => {
      if (refreshTimer.value) {
        clearInterval(refreshTimer.value)
      }
      
      // 在训练期间或选择了环境时启动自动刷新
      if (isTraining.value || selectedEnvironment.value) {
        refreshTimer.value = setInterval(async () => {
          try {
            // 如果正在训练，先检查是否有新的CSV文件
            if (isTraining.value) {
              await loadAvailableMetricsFiles()
            }
            
            // 如果有选择的环境，加载数据
            if (selectedEnvironment.value) {
              const selectedFile = availableMetricsFiles.value.find(file => file.name === selectedEnvironment.value)
              if (selectedFile) {
                const environmentName = selectedEnvironment.value.split('/')[0]
                const filePath = selectedFile.path
                const res = await fetch(`/api/training-metrics/${environmentName}?filePath=${encodeURIComponent(filePath)}`)
                const result = await res.json()
                if (result.success) {
                  trainingMetrics.value = result.metrics
                  updateRealtimeChart()
                }
              }
            }
          } catch (error) {
            console.error('自动刷新数据失败:', error)
          }
        }, refreshInterval.value)
      }
    }
    
    const stopAutoRefresh = () => {
      if (refreshTimer.value) {
        clearInterval(refreshTimer.value)
        refreshTimer.value = null
      }
    }
    
    // 监听刷新间隔变化
    watch(refreshInterval, () => {
      startAutoRefresh()
    })
    
    // 监听训练状态变化
    watch(isTraining, (newVal) => {
      if (newVal) {
        startAutoRefresh()
      } else {
        stopAutoRefresh()
      }
    })
    
    // 监听选中指标变化
    watch(selectedMetrics, () => {
      updateRealtimeChart()
    })
    
    // 监听算法变化，自动更新选中的指标
    watch(selectedAlgorithm, () => {
      // 当算法改变时，重置选中的指标为默认的基础指标
      selectedMetrics.value = ['total_reward', 'best_reward']
    })
    
    // 监听选择的训练记录变化，自动更新指标选择
    watch(selectedEnvironment, (newEnvironment) => {
      console.log('selectedEnvironment changed:', newEnvironment)
      // 当训练记录改变时，重置选中的指标为默认的基础指标
      selectedMetrics.value = ['total_reward', 'best_reward']
    })
    
    // 监听图表类型变化
    watch(chartType, () => {
      updateRealtimeChart()
    })
    
    // 监听页面切换，强制重新初始化图表
    watch(activeMenu, (newMenu) => {
      if (newMenu === 'visualization') {
        console.log('切换到可视化页面，强制重新初始化图表')
        // 清理现有实例
        if (realtimeChartInstance.value) {
          try {
            realtimeChartInstance.value.dispose()
            realtimeChartInstance.value = null
            console.log('已清理现有图表实例')
          } catch (error) {
            console.error('清理图表实例失败:', error)
          }
        }
        
        // 使用多层延迟重新初始化，避免 ResizeObserver 错误
        setTimeout(() => {
          requestAnimationFrame(() => {
            setTimeout(() => {
              console.log('开始强制初始化图表')
              initRealtimeChart()
              
              // 如果有数据，延迟更新图表
              if (trainingMetrics.value.length > 0) {
                setTimeout(() => {
                  console.log('有数据，更新图表')
                  updateRealtimeChart()
                }, 200)
              }
            }, 100)
          })
        }, 500)
      }
    })
    const addChart = () => {
      if (!newChart.value.title || newChart.value.metrics.length === 0) return
      
      const chartId = `chart_${Date.now()}`
      const chart = {
        id: chartId,
        title: newChart.value.title,
        metrics: [...newChart.value.metrics],
        type: newChart.value.type
      }
      
      activeCharts.value.push(chart)
      
      // 重置表单
      newChart.value = {
        title: '',
        metrics: [],
        type: 'line'
      }
      showAddChartDialog.value = false
      
      // 实时图表系统会自动更新，无需手动初始化
    }
    
    // 移除图表
    // 旧的图表方法已移除，现在使用实时图表系统
    
    // 图表相关
    const initCharts = async () => {
      console.log('initCharts 开始执行')
      await loadAvailableMetricsFiles()
      console.log('loadAvailableMetricsFiles 完成')
      
      // 等待DOM完全渲染
      await nextTick()
      
      // 直接初始化图表，让 initRealtimeChart 自己处理容器检查
      console.log('开始初始化图表')
      initRealtimeChart()
      
      // 如果有可用的指标文件，自动加载数据
      if (availableMetricsFiles.value.length > 0) {
        console.log('有可用文件，开始加载数据')
        setTimeout(() => {
          loadMetricsData()
        }, 500)
      } else {
        console.log('没有可用文件')
      }
    }
    
    const updateCharts = () => {
      if (trainingMetrics.value.length === 0) return
      
      // 实时图表系统会自动更新
      updateRealtimeChart()
    }
    
    const refreshCharts = () => {
      loadMetricsData()
    }
    
    const exportData = () => {
      if (trainingMetrics.value.length === 0) {
        ElMessage.warning('暂无数据可导出')
        return
      }
      
      const dataStr = JSON.stringify(trainingMetrics.value, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)
      const link = document.createElement('a')
      link.href = url
      link.download = `training_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`
      link.click()
      URL.revokeObjectURL(url)
      
      ElMessage.success('数据导出成功')
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
    
    // 组件生命周期
    onMounted(async () => {
      await loadConfigs()
      // 在加载配置后，从备份文件初始化训练控制表单
      await initializeTrainingFormFromBackup()
      await initCharts()
    })
    
    onBeforeUnmount(() => {
      // 清理日志轮询
      if (logPollingInterval) {
        clearInterval(logPollingInterval)
        logPollingInterval = null
      }
      
      // 清理实时图表
      if (realtimeChartInstance.value) {
        try {
          realtimeChartInstance.value.dispose()
          realtimeChartInstance.value = null
        } catch (error) {
          console.warn('清理图表实例时出错:', error)
        }
      }
      
      // 清理自动刷新定时器
      stopAutoRefresh()
      
      // 清理所有可能的定时器
      if (refreshTimer.value) {
        clearInterval(refreshTimer.value)
        refreshTimer.value = null
      }
    })

    return {
      activeMenu,
      username,
      isTraining,
      trainingLogs,
      trainingMetrics,
      trainingForm,
      enableRender,
      hasTrainingData,
      currentEpisode,
      avgReward,
      bestReward,
      totalSteps,
      algorithmConfigs,
      gameConfigs,
      selectedAlgorithm,
      selectedGame,
      availableEnvironments,
      availableActionSpaces,
      algorithmParameters,
      // 统一配置管理
      configFiles,
      currentFileContent,
      currentFileInfo,
      activeFileTab,
      showBackupManager,
      backupList,
      syntaxHighlight,
      isResetSupported,
      isEditing,
      originalContent,
      loadAllConfigs,
      loadFileContent,
      onFileTabClick,
      saveCurrentFile,
      resetToOriginal,
      refreshBackups,
      restoreBackup,
      deleteBackup,
      deleteAllBackups,
      getFileDisplayName,
      formatFileTime,
      formatCode,
      languageClass,
      getLanguageClass,
      highlightedCode,
      onCodeChange,
      onCodeKeydown,
      onTextareaScroll,
      startEditing,
      cancelEditing,
      onConfigAlgorithmChange,
      onConfigGameChange,
      syncTrainingFormFromConstants,
      syncConstantsFromTrainingForm,
      initializeTrainingFormFromBackup,
      // 参数文件管理 (保留兼容性)
      configFileType,
      configFileContent,
      configFilePath,
      configFileInfo,
      configForm,
      handleMenuSelect,
      onAlgorithmChange,
      onGameChange,
      resetTrainingConfig,
      loadConfigFile,
      loadScriptFile,
      saveConfigFile,
      resetConfigFile,
      startTraining,
      stopTraining,
      clearLogs,
      refreshCharts,
      exportData,
      logout,
      // 新的可视化功能
      availableMetricsFiles,
      selectedEnvironment,
      selectedMetrics,
      activeCharts,
      showAddChartDialog,
      newChart,
      availableMetricsForChart,
      availableMetricsForCurrentAlgorithm,
      loadAvailableMetricsFiles,
      loadMetricsData,
      addChart,
      // 旧的图表方法已移除
      // 实时图表相关
      realtimeChartRef,
      chartType,
      refreshInterval,
      updateRealtimeChart,
      initRealtimeChart,
      startAutoRefresh,
      stopAutoRefresh,
      getMetricLabel
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

.training-config {
  margin-bottom: 20px;
}

.algorithm-config {
  margin-top: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.algorithm-config h4 {
  margin: 0 0 15px 0;
  color: #409EFF;
}

.training-logs {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.log-container {
  height: 400px;
  overflow-y: auto;
  padding: 10px;
  background-color: #1e1e1e;
  color: #f0f0f0;
  font-family: monospace;
  border-radius: 4px;
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

.no-data {
  text-align: center;
  padding: 40px;
}

.charts-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.chart-card {
  grid-column: span 1;
}

.stats-card {
  grid-column: span 2;
}

.chart-container {
  height: 300px;
  width: 100%;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  padding: 20px;
}

.stat-item {
  text-align: center;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #409EFF;
  margin-bottom: 8px;
}

.stat-label {
  color: #606266;
  font-size: 14px;
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

.preview-video {
  width: 100%;
  height: 100%;
}

.video-placeholder {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100%;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.video-icon {
  font-size: 64px;
  color: #409EFF;
  margin-bottom: 20px;
}

.video-note {
  font-size: 14px;
  color: #909399;
  margin-top: 10px;
}

.config-section {
  padding: 20px;
}

.algorithm-selector {
  margin-bottom: 20px;
}

.config-editor {
  margin-bottom: 20px;
}

.config-textarea {
  font-family: 'Courier New', monospace;
  font-size: 14px;
}

.config-info {
  margin-top: 20px;
}

.algorithm-params {
  margin-top: 20px;
}

.algorithm-params .el-form-item {
  margin-bottom: 18px;
}

.algorithm-params .el-input-number {
  width: 100%;
}

.training-flags {
  margin-top: 20px;
}

.training-scrollable {
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  padding-right: 10px;
}

.config-scrollable {
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 10px;
  /* 避免ResizeObserver问题 */
  contain: layout style;
  will-change: scroll-position;
}

.config-selector {
  margin-bottom: 20px;
}

.file-list {
  margin-bottom: 20px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.file-path {
  color: #666;
  font-size: 12px;
}

.code-editor-container {
  position: relative;
}

.syntax-highlight-editor {
  position: relative;
  background: #2d3748;
  border: 1px solid #4a5568;
  border-radius: 6px;
  font-family: 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.6;
  height: 400px;
  overflow: hidden;
}

.syntax-highlight-editor pre {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 16px;
  background: transparent;
  color: #e2e8f0;
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  white-space: pre;
  word-wrap: normal;
  overflow: auto;
  z-index: 1;
  box-sizing: border-box;
}

.syntax-highlight-editor code {
  background: transparent !important;
  color: inherit;
  font-family: inherit;
}

.code-textarea-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  padding: 16px;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  color: transparent;
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  resize: none;
  caret-color: #e2e8f0;
  z-index: 2;
  box-sizing: border-box;
  overflow: auto;
  white-space: pre;
  word-wrap: normal;
}

.syntax-highlight-editor {
  background: #2d3748;
  border: 1px solid #4a5568;
  border-radius: 6px;
  padding: 0;
  font-family: 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.6;
  max-height: 500px;
  overflow-y: auto;
  position: relative;
}

.syntax-highlight-editor pre {
  margin: 0;
  padding: 16px;
  white-space: pre-wrap;
  word-wrap: break-word;
  background: transparent;
}

.syntax-highlight-editor code {
  background: transparent !important;
  color: #e2e8f0;
  font-family: inherit;
}

/* Prism.js 主题样式覆盖 - 适用于可编辑和只读编辑器 */
.syntax-highlight-editor .token.comment,
.syntax-highlight-editor .token.prolog,
.syntax-highlight-editor .token.doctype,
.syntax-highlight-editor .token.cdata,
.code-highlight-overlay .token.comment,
.code-highlight-overlay .token.prolog,
.code-highlight-overlay .token.doctype,
.code-highlight-overlay .token.cdata {
  color: #68d391;
}

.syntax-highlight-editor .token.punctuation,
.code-highlight-overlay .token.punctuation {
  color: #e2e8f0;
}

.syntax-highlight-editor .token.property,
.syntax-highlight-editor .token.tag,
.syntax-highlight-editor .token.boolean,
.syntax-highlight-editor .token.number,
.syntax-highlight-editor .token.constant,
.syntax-highlight-editor .token.symbol,
.syntax-highlight-editor .token.deleted,
.code-highlight-overlay .token.property,
.code-highlight-overlay .token.tag,
.code-highlight-overlay .token.boolean,
.code-highlight-overlay .token.number,
.code-highlight-overlay .token.constant,
.code-highlight-overlay .token.symbol,
.code-highlight-overlay .token.deleted {
  color: #f687b3;
}

.syntax-highlight-editor .token.selector,
.syntax-highlight-editor .token.attr-name,
.syntax-highlight-editor .token.string,
.syntax-highlight-editor .token.char,
.syntax-highlight-editor .token.builtin,
.syntax-highlight-editor .token.inserted,
.code-highlight-overlay .token.selector,
.code-highlight-overlay .token.attr-name,
.code-highlight-overlay .token.string,
.code-highlight-overlay .token.char,
.code-highlight-overlay .token.builtin,
.code-highlight-overlay .token.inserted {
  color: #fbb6ce;
}

.syntax-highlight-editor .token.operator,
.syntax-highlight-editor .token.entity,
.syntax-highlight-editor .token.url,
.syntax-highlight-editor .language-css .token.string,
.syntax-highlight-editor .style .token.string,
.code-highlight-overlay .token.operator,
.code-highlight-overlay .token.entity,
.code-highlight-overlay .token.url,
.code-highlight-overlay .language-css .token.string,
.code-highlight-overlay .style .token.string {
  color: #f6ad55;
}

.syntax-highlight-editor .token.atrule,
.syntax-highlight-editor .token.attr-value,
.syntax-highlight-editor .token.keyword,
.code-highlight-overlay .token.atrule,
.code-highlight-overlay .token.attr-value,
.code-highlight-overlay .token.keyword {
  color: #63b3ed;
}

.syntax-highlight-editor .token.function,
.syntax-highlight-editor .token.class-name,
.code-highlight-overlay .token.function,
.code-highlight-overlay .token.class-name {
  color: #68d391;
}

.syntax-highlight-editor .token.regex,
.syntax-highlight-editor .token.important,
.syntax-highlight-editor .token.variable {
  color: #f6ad55;
}

.editor-actions {
  display: flex;
  gap: 10px;
}

.backup-manager {
  max-height: 500px;
}

.backup-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.backup-actions {
  display: flex;
  gap: 10px;
}

.backup-header h4 {
  margin: 0;
  color: #303133;
}

.training-scrollable::-webkit-scrollbar,
.config-scrollable::-webkit-scrollbar {
  width: 8px;
}

.training-scrollable::-webkit-scrollbar-track,
.config-scrollable::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.training-scrollable::-webkit-scrollbar-thumb,
.config-scrollable::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.training-scrollable::-webkit-scrollbar-thumb:hover,
.config-scrollable::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 可视化相关样式 */
.metrics-selector {
  margin-bottom: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
}

.metrics-selector h4 {
  margin: 0 0 15px 0;
  color: #303133;
}

.metrics-checkboxes {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
}

.add-chart-card {
  border: 2px dashed #dcdfe6;
  background: #fafafa;
}

.add-chart-content {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}

.add-chart-form {
  padding: 20px 0;
}

.charts-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.chart-card {
  min-height: 350px;
}

.chart-container {
  height: 300px;
  width: 100%;
  max-width: 500px;
}

/* 实时图表样式 */
.realtime-controls {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.metrics-selector h4 {
  margin: 0 0 15px 0;
  color: #303133;
  font-size: 16px;
}

.metrics-checkboxes {
  margin-bottom: 20px;
}

.chart-controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
}

.refresh-interval {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #606266;
  font-size: 14px;
}

.realtime-chart-container {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
  margin-top: 20px;
}

.main-chart-card {
  min-height: 350px;
}

.realtime-chart {
  height: 300px;
  width: 100%;
  max-width: 600px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-stats {
  font-size: 14px;
  color: #606266;
}

.realtime-chart {
  height: 450px;
  width: 100%;
}

.stats-card {
  min-height: 500px;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.stat-item {
  text-align: center;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #409EFF;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #606266;
}

.no-data-tip {
  text-align: center;
  padding: 60px 20px;
}

/* 训练可视化滚动条样式 */
.visualization-scrollable {
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 10px;
  /* 避免ResizeObserver问题 */
  contain: layout style;
  will-change: scroll-position;
}

.visualization-scrollable::-webkit-scrollbar {
  width: 8px;
}

.visualization-scrollable::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.visualization-scrollable::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.visualization-scrollable::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>