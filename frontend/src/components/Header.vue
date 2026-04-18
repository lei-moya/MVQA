<template>
  <header>
    <div class="left-section">
      <h1><i class="fa-solid fa-chart-line"></i> 视频质量分析平台 </h1>
      <div class="badge">Vue 3 + FastAPI + ECharts</div>
    </div>
    <div class="center-section">
      <el-menu :default-active="activeIndex" class="el-menu-demo" mode="horizontal" @select="handleSelect">
        <el-menu-item index="1" @click="$router.push('/')">
          <i class="fa-solid fa-home"></i>
          <span>首页</span>
        </el-menu-item>
        <el-menu-item index="3" @click="$router.push('/settings')">
          <i class="fa-solid fa-gear"></i>
          <span>设置</span>
        </el-menu-item>
        <el-menu-item index="4" @click="$router.push('/help')">
          <i class="fa-solid fa-question-circle"></i>
          <span>帮助</span>
        </el-menu-item>
      </el-menu>
    </div>
    <div class="right-section">
      <div class="user-info" v-if="user">
        <el-dropdown trigger="click" placement="bottom">
          <div style="display: flex; align-items: center; gap: 12px; cursor: pointer;">
            <span class="username">{{ user.username }}</span>
            <div class="avatar">
              <i class="fa-solid fa-user-circle"></i>
            </div>
          </div>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item @click="showProfileDialog = true">
                <i class="fa-solid fa-user"></i>
                <span>个人资料</span>
              </el-dropdown-item>
              <el-dropdown-item @click="showAboutDialog = true">
                <i class="fa-solid fa-info-circle"></i>
                <span>关于我们</span>
              </el-dropdown-item>
              <el-dropdown-item divided @click="handleLogout">
                <i class="fa-solid fa-sign-out-alt"></i>
                <span>登出</span>
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
      <div class="user-info" v-else>
        <div style="display: flex; align-items: center; gap: 12px;">
          <span class="username">游客</span>
          <div class="avatar">
            <i class="fa-solid fa-user"></i>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 个人资料浮框 -->
    <el-dialog
      v-model="showProfileDialog"
      title="个人资料"
      width="400px"
      center
      close-on-click-modal
      :show-close="false"
      :custom-class="'profile-dialog'"
    >
      <div class="profile-content">
        <div class="profile-info">
          <div class="info-item">
            <span class="info-label">用户名:</span>
            <span class="info-value">{{ user?.username }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">B站ID:</span>
            <span class="info-value">{{ user?.bilibili_mid }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">B站邮箱:</span>
            <span class="info-value">{{ user?.email }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">等级:</span>
            <span class="info-value">{{ user?.level || '未知' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">创建时间:</span>
            <span class="info-value">{{ formatDate(user?.created_at) }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">更新时间:</span>
            <span class="info-value">{{ formatDate(user?.updated_at) }}</span>
          </div>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showProfileDialog = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
    
    <!-- 关于我们浮框 -->
    <el-dialog
      v-model="showAboutDialog"
      title="关于我们"
      width="400px"
      center
      close-on-click-modal
      :show-close="false"
      :custom-class="'about-dialog'"
    >
      <div class="about-content">
        <div class="about-logo">
          <i class="fa-solid fa-chart-line"></i>
        </div>
        <div class="about-info">
          <div class="about-item">
            <span class="about-label">项目名称:</span>
            <span class="about-value">视频质量分析平台</span>
          </div>
          <div class="about-item">
            <span class="about-label">版本:</span>
            <span class="about-value">1.0.0</span>
          </div>
          <div class="about-item">
            <span class="about-label">作者:</span>
            <span class="about-value">项目团队 Lei Moya</span>
          </div>
          <div class="about-item">
            <span class="about-label">技术栈:</span>
            <span class="about-value">Vue 3 + FastAPI + ECharts</span>
          </div>
          <div class="about-item">
            <span class="about-label">描述:</span>
            <span class="about-value">一个用于分析视频质量的综合平台，支持视频上传、分析和评分。</span>
          </div>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showAboutDialog = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </header>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { ElMessage } from 'element-plus';
import { logout, getCurrentUser } from '../api/index.js';

const route = useRoute();
const user = ref(null);
const showProfileDialog = ref(false);
const showAboutDialog = ref(false);

// 格式化日期函数
const formatDate = (dateStr) => {
  if (!dateStr) return '未知';
  const date = new Date(dateStr);
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

const activeIndex = computed(() => {
  if (route.path === '/') return '1';
  if (route.path === '/settings') return '3';
  if (route.path === '/help') return '4';
  return '1';
});

const handleSelect = (key, keyPath) => {
  // 处理菜单选择
};

const handleLogout = async () => {
  try {
    // 尝试调用后端登出API
    await logout();
  } catch (error) {
    console.error('登出API调用失败:', error);
    // 即使API调用失败，也继续执行登出操作
  } finally {
    // 无论API调用是否成功，都删除本地存储的token和user
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    user.value = null;
    ElMessage.success('登出成功');
    // 刷新页面以显示登录组件
    window.location.reload();
  }
};

onMounted(async () => {
  // 检查是否已登录
  const token = localStorage.getItem('token');
  if (token) {
    try {
      // 通过API获取最新的用户信息
      const response = await getCurrentUser();
      user.value = response.data;
      // 更新localStorage中的用户信息
      localStorage.setItem('user', JSON.stringify(response.data));
    } catch (error) {
      console.error('获取用户信息失败:', error);
      // 如果API调用失败，尝试从localStorage中读取
      const userStr = localStorage.getItem('user');
      if (userStr) {
        user.value = JSON.parse(userStr);
      }
    }
  }
});
</script>

<style scoped>
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  height: 100%;
}

.left-section {
  display: flex;
  align-items: flex-end;
  gap: 20px;
}

.left-section :deep(h1) {
  font-size: 1.2rem;
  font-weight: 600;
  color: #0f172a;
  letter-spacing: -0.02em;
}

.left-section :deep(h1 i) {
  color: #2563eb;
}

.badge {
  font-size: 0.72rem;
  font-weight: 500;
  color: #475569;
  background: rgba(37, 99, 235, 0.08);
  border: 1px solid rgba(37, 99, 235, 0.14);
  padding: 4px 10px;
  border-radius: 999px;
  letter-spacing: 0.02em;
}

.center-section {
  flex: 1;
  margin: 0 40px;
}

.right-section {
  display: flex;
  align-items: center;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.username {
  font-size: 14px;
  font-weight: 500;
  color: #334155;
}

.avatar {
  font-size: 2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.avatar img {
  font-size: 1rem; /* 重置图片的字体大小 */
}

.avatar:hover {
  color: #2563eb;
}

:deep(.el-menu) {
  border-bottom: none;
}

:deep(.el-menu-item) {
  font-size: 16px;
  font-weight: 500;
  padding: 0 18px;
  height: 60px;
  line-height: 60px;
  color: #334155;
}

:deep(.el-menu-item.is-active) {
  color: #2563eb !important;
  border-bottom-color: #2563eb !important;
}

:deep(.el-menu-item i) {
  margin-right: 8px;
}

:deep(.el-dropdown-menu) {
  min-width: 160px;
}

:deep(.el-dropdown-item) {
  display: flex;
  align-items: center;
  gap: 8px;
}

:deep(.el-dropdown-item i) {
  font-size: 14px;
}

/* 个人资料浮框样式 */
.profile-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0;
}

.profile-info {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 15px;
  padding: 0 20px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #f0f0f0;
  width: 100%;
}

.info-item .info-label {
  font-weight: 500;
  color: #666;
  min-width: 100px;
  text-align: left;
}

.info-item .info-value {
  font-weight: 500;
  color: #333;
  text-align: right;
  flex: 1;
  margin-left: 20px;
}

.info-item:last-child {
  border-bottom: none;
}

.info-label {
  font-weight: 500;
  color: #666;
}

.info-value {
  color: #333;
  font-size: 14px;
}

.dialog-footer {
  display: flex;
  justify-content: center;
}

/* 个人资料浮框样式，确保在弹幕之上 */
:deep(.profile-dialog) {
  z-index: 10000 !important;
}

:deep(.profile-dialog .el-dialog__wrapper) {
  z-index: 10000 !important;
}

/* 关于我们浮框样式 */
:deep(.about-dialog) {
  z-index: 10000 !important;
}

:deep(.about-dialog .el-dialog__wrapper) {
  z-index: 10000 !important;
}

.about-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0;
}

.about-logo {
  font-size: 4rem;
  color: #3b82f6;
  margin-bottom: 20px;
}

.about-info {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 15px;
  padding: 0 20px;
}

.about-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #f0f0f0;
  width: 100%;
}

.about-item .about-label {
  font-weight: 500;
  color: #666;
  min-width: 100px;
  text-align: left;
}

.about-item .about-value {
  font-weight: 500;
  color: #333;
  text-align: right;
  flex: 1;
  margin-left: 20px;
}

.about-item:last-child {
  border-bottom: none;
}
</style>
