import axios from 'axios'

const apiBaseURL = process.env.VUE_APP_API_BASE || ''

const apiClient = axios.create({
  baseURL: apiBaseURL,
  headers: {
    'Content-Type': 'application/json; charset=UTF-8'
  },
  withCredentials: false,
  timeout: 30000
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error)
)

export default apiClient



