// src/router.js
import { createRouter, createWebHistory } from 'vue-router'
import Home from './components/Home.vue'
import Settings from './components/Settings.vue'
import Help from './components/Help.vue'

const routes = [
  { path: '/', component: Home },
  { path: '/settings', component: Settings, meta: { requiresAuth: true } },
  { path: '/help', component: Help },
  // 通配符路由，处理不存在的路径
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, _from, next) => {
  const token = typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null
  if (to.meta.requiresAuth && !token) {
    next({ path: '/', query: { redirect: to.fullPath } })
    return
  }
  next()
})

export default router
