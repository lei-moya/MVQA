<template>
  <el-container class="main-container">
    <el-header height="60px">
      <Header/>
    </el-header>
    <el-main class="main-content">
      <router-view />
    </el-main>
    <Login @login-success="handleLoginSuccess" />
  </el-container>
</template>

<script setup>
import Header from './components/Header.vue';
import Login from './components/Login.vue';

const handleLoginSuccess = (user) => {
  console.log('登录成功:', user);
  // 刷新页面以更新Header中的用户信息
  window.location.reload();
};
</script>

<style>
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
  color: #303133;
  background-color: #f5f7fa;
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

.el-form-item__content {
  margin-right: 10px;
}

.main-container {
  height: 100vh;
  overflow: hidden;
}

.el-header {
  padding: 0 20px;
  background-color: #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  z-index: 10;
}

.el-main {
  padding: 20px;
  background-color: #f5f7fa;
  overflow: hidden;
  height: calc(100vh - 60px); /* 减去header的高度 */
}

.horizontal-layout {
  display: flex;
  gap: 20px;
  height: 100%;
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