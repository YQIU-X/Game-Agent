<template>
  <div class="login-container">
    <el-card class="login-card">
      <div class="login-header">
        <img src="../assets/logo.png" alt="Logo" class="logo">
        <h2>游戏智能体开发平台</h2>
      </div>
      
      <el-form :model="loginForm" :rules="rules" ref="loginFormRef">
        <el-form-item prop="username">
          <el-input 
            v-model="loginForm.username" 
            placeholder="用户名" 
            prefix-icon="User"
          ></el-input>
        </el-form-item>
        
        <el-form-item prop="password">
          <el-input 
            v-model="loginForm.password" 
            :type="showPassword ? 'text' : 'password'" 
            placeholder="密码" 
            prefix-icon="Lock"
          >
            <template #suffix>
              <el-icon 
                class="password-icon" 
                @click="showPassword = !showPassword"
              >
                <component :is="showPassword ? 'View' : 'Hide'"></component>
              </el-icon>
            </template>
          </el-input>
        </el-form-item>
        
        <el-form-item prop="role">
          <el-radio-group v-model="loginForm.role">
            <el-radio label="developer">开发者</el-radio>
            <el-radio label="player">玩家</el-radio>
          </el-radio-group>
        </el-form-item>
        
        <el-form-item>
          <div class="login-options">
            <el-checkbox v-model="loginForm.rememberMe">记住密码</el-checkbox>
            <el-checkbox v-model="loginForm.autoLogin">自动登录</el-checkbox>
          </div>
        </el-form-item>
        
        <el-form-item>
          <el-button type="primary" class="login-button" @click="handleLogin" :loading="loading">
            登录
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script>
import { ref, reactive, onMounted, computed } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

export default {
  name: 'LoginView',
  setup() {
    const store = useStore()
    const router = useRouter()
    const loginFormRef = ref(null)
    const loading = ref(false)
    const showPassword = ref(false)
    
    // 表单数据
    const loginForm = reactive({
      username: '',
      password: '',
      role: 'player', // 默认为玩家
      rememberMe: false,
      autoLogin: false
    })
    
    // 表单验证规则
    const rules = {
      username: [
        { required: true, message: '请输入用户名', trigger: 'blur' },
        { min: 3, max: 20, message: '长度在 3 到 20 个字符', trigger: 'blur' }
      ],
      password: [
        { required: true, message: '请输入密码', trigger: 'blur' },
        { min: 6, max: 20, message: '长度在 6 到 20 个字符', trigger: 'blur' }
      ],
      role: [
        { required: true, message: '请选择角色', trigger: 'change' }
      ]
    }
    
    // 获取记住的凭据
    const rememberedCredentials = computed(() => store.getters.rememberedCredentials)
    
    // 组件挂载时检查是否有记住的凭据
    onMounted(() => {
      const { username, password, role } = rememberedCredentials.value
      
      if (username && password) {
        loginForm.username = username
        loginForm.password = password
        loginForm.role = role
        loginForm.rememberMe = true
        
        // 如果之前设置了自动登录，则自动登录
        if (localStorage.getItem('autoLogin') === 'true') {
          loginForm.autoLogin = true
          handleLogin()
        }
      }
    })
    
    // 登录处理
    const handleLogin = async () => {
      if (!loginFormRef.value) return
      
      await loginFormRef.value.validate(async (valid) => {
        if (valid) {
          loading.value = true
          
          try {
            // 调用登录action
            await store.dispatch('login', {
              username: loginForm.username,
              password: loginForm.password,
              role: loginForm.role,
              rememberMe: loginForm.rememberMe
            })
            
            // 保存自动登录设置
            localStorage.setItem('autoLogin', loginForm.autoLogin)
            
            // 登录成功后跳转到对应角色的页面
            const redirectPath = loginForm.role === 'developer' ? '/developer' : '/player'
            router.push(redirectPath)
            
            ElMessage.success('登录成功')
          } catch (error) {
            ElMessage.error('登录失败: ' + error.message)
          } finally {
            loading.value = false
          }
        }
      })
    }
    
    return {
      loginFormRef,
      loginForm,
      rules,
      loading,
      showPassword,
      handleLogin
    }
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f5f7fa;
}

.login-card {
  width: 400px;
  padding: 20px;
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
}

.logo {
  width: 80px;
  height: 80px;
  margin-bottom: 10px;
}

.login-options {
  display: flex;
  justify-content: space-between;
}

.login-button {
  width: 100%;
}

.password-icon {
  cursor: pointer;
}
</style>