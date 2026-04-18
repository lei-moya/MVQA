<template>
  <el-container class="main-container">
    <el-header height="60px">
      <Header/>
    </el-header>
    <el-main class="main-content">
      <div class="router-view-fill">
        <router-view />
      </div>
    </el-main>
    <Login @login-success="handleLoginSuccess" />
  </el-container>
</template>

<script setup>
import Header from './components/Header.vue';
import Login from './components/Login.vue';

const handleLoginSuccess = (user) => {
  // 刷新页面以更新Header中的用户信息
  window.location.reload();
};
</script>

<style>
:root {
  --mvqa-text: #1e293b;
  --mvqa-muted: #64748b;
  --el-border-radius-base: 9px;
}

/* 重置默认样式 */
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html,
body {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

body {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB',
  'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;
  color: var(--mvqa-text);
  background: linear-gradient(165deg, #e9eef5 0%, #f3f5f9 42%, #edf1f7 100%);
}

#app {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

/* 减小所有卡片头部的padding */
.el-card__header {
  padding: 10px 15px !important;
}

.el-card__body {
  padding: 15px !important;
  overflow: hidden;
}

.el-card {
  border: 1px solid rgba(15, 23, 42, 0.06);
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.6) inset,
    0 4px 20px rgba(15, 23, 42, 0.06) !important;
}

.el-form-item__content {
  margin-right: 10px;
}

.main-container {
  height: 100vh;
  overflow: hidden;
}

.el-header {
  padding: 0 20px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 4px 20px rgba(15, 23, 42, 0.07);
  z-index: 10;
}

.el-main {
  padding: 20px;
  background: transparent;
  overflow: hidden;
  height: calc(100vh - 60px); /* 减去header的高度 */
}

.router-view-fill {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.horizontal-layout {
  display: flex;
  gap: 20px;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

.layout-item {
  overflow: hidden;
  height: 100%;
}

.full-height-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.full-height-card .el-card__body {
  flex: 1;
  overflow: hidden;
}

.mb-4 {
  margin-bottom: 16px;
}

.mt-4 {
  margin-top: 16px;
}
</style>