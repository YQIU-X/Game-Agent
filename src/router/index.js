import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Login',
    component: () => import('../views/Login.vue')
  },
  {
    path: '/developer',
    name: 'Developer',
    component: () => import('../views/Developer.vue'),
    meta: { requiresAuth: true, role: 'developer' }
  },
  {
    path: '/player',
    name: 'Player',
    component: () => import('../views/Player.vue'),
    meta: { requiresAuth: true, role: 'player' }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

// 路由守卫
router.beforeEach((to, from, next) => {
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true'
  const userRole = localStorage.getItem('userRole')

  if (to.meta.requiresAuth && !isAuthenticated) {
    next({ name: 'Login' })
  } else if (to.meta.requiresAuth && to.meta.role !== userRole) {
    // 如果用户角色不匹配，重定向到对应角色的页面
    next({ name: userRole === 'developer' ? 'Developer' : 'Player' })
  } else {
    next()
  }
})

export default router