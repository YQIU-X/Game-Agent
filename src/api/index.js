import api from './client'

export const loginApi = (payload) => api.post('/api/login', payload)

export const listConfigsApi = () => api.get('/api/configs')
export const readConfigApi = (filename) => api.get(`/api/configs/${encodeURIComponent(filename)}`)
export const saveConfigApi = (filename, data) => api.post(`/api/configs/${encodeURIComponent(filename)}`, data)

export const startTrainApi = (payload) => api.post('/api/train', payload)
export const fetchLogsApi = (processId) => api.get(`/api/logs/${encodeURIComponent(processId)}`)
export const stopProcessApi = (processId) => api.post(`/api/stop/${encodeURIComponent(processId)}`)

export const listGamesApi = () => api.get('/api/games')
export const listLevelsApi = (game) => api.get('/api/levels', { params: { game } })
export const listAgentsApi = (game) => api.get('/api/agents', { params: { game } })
export const startGameApi = (payload) => api.post('/api/start-game', payload)

// 新增：模型同步与列举、启动DQN
export const syncModelsApi = () => api.post('/api/sync-models')
export const listModelsApi = () => api.get('/api/models')
export const startDqnApi = (payload) => api.post('/api/start-dqn', payload)
export const startStreamApi = (payload) => api.post('/api/start-stream', payload)
export const fetchFrameApi = (processId) => api.get(`/api/frame/${encodeURIComponent(processId)}`)

// 新增：玩家控制相关API
export const startPlayerApi = (payload) => api.post('/api/start-player', payload)
export const playerActionApi = (processId, action) => api.post(`/api/player-action/${encodeURIComponent(processId)}`, { action })
export const fetchPlayerFrameApi = (processId) => api.get(`/api/player-frame/${encodeURIComponent(processId)}`)



