// src/router.js
import { createRouter, createWebHistory } from 'vue-router'
import Home from './components/Home.vue'
import Settings from './components/Settings.vue'
import Help from './components/Help.vue'

const routes = [
  { path: '/', component: Home },
  { path: '/settings', component: Settings },
  { path: '/help', component: Help }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
