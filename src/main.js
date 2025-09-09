import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// 导入 Prism.js 语法高亮
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-markdown'

// 全局处理 ResizeObserver 错误 - 彻底忽略所有相关错误
const originalConsoleError = console.error
console.error = (...args) => {
  if (args[0] && typeof args[0] === 'string' && args[0].includes('ResizeObserver')) {
    return // 忽略所有 ResizeObserver 相关错误
  }
  originalConsoleError.apply(console, args)
}

// 全局错误处理 - 阻止所有 ResizeObserver 错误传播
window.addEventListener('error', (e) => {
  if (e.message && e.message.includes('ResizeObserver')) {
    e.stopImmediatePropagation()
    e.preventDefault()
    return false
  }
})

// 全局未处理的 Promise 拒绝处理
window.addEventListener('unhandledrejection', (e) => {
  if (e.reason && e.reason.message && e.reason.message.includes('ResizeObserver')) {
    e.preventDefault()
    return false
  }
})

// 重写 ResizeObserver 构造函数，添加错误处理
const OriginalResizeObserver = window.ResizeObserver
window.ResizeObserver = class extends OriginalResizeObserver {
  constructor(callback) {
    super((entries, observer) => {
      try {
        callback(entries, observer)
      } catch (error) {
        if (error.message && error.message.includes('ResizeObserver')) {
          // 忽略 ResizeObserver 错误
          return
        }
        throw error
      }
    })
  }
}

const app = createApp(App)

// 注册所有Element Plus图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 将 Prism 添加到全局属性
app.config.globalProperties.$prism = Prism

app.use(router)
   .use(store)
   .use(ElementPlus)
   .mount('#app')
