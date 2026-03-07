<template>
  <div class="login-overlay" v-if="!isLoggedIn">
    <div class="login-container">
      <div class="login-header">
        <h2><i class="fa-solid fa-right-to-bracket"></i> B站登录</h2>
        <p>使用B站账号扫码登录</p>
      </div>
      <div class="login-content">
        <div class="qr-code-container" v-if="qrCode">
          <img :src="qrCode" alt="登录二维码" class="qr-code" />
          <p class="qr-tip">请使用B站App扫描二维码</p>
          <p class="qr-expire">二维码将在 {{ timeLeft }} 秒后自动刷新</p>
        </div>
        <div class="loading" v-else>
          <el-icon class="is-loading"><component :is="Loading" /></el-icon>
          <span>获取二维码中...</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';
import { Loading } from '@element-plus/icons-vue';
import { getLoginQRCode, checkQRStatus } from '../api/index.js';

const isLoggedIn = ref(false);
const qrCode = ref('');
const qrKey = ref('');
const loading = ref(false);
const pollInterval = ref(null);
const refreshInterval = ref(null);
const timeLeft = ref(60); // 二维码有效期60秒

// 触发登录成功事件
const emit = defineEmits(['login-success']);

const getQRCode = async () => {
  loading.value = true;
  try {
    const response = await getLoginQRCode();
    qrCode.value = response.data.qr_code;
    qrKey.value = response.data.qr_key;
    startPolling();
    startRefreshTimer();
  } catch (error) {
    ElMessage.error('获取二维码失败');
    console.error('获取二维码失败:', error);
  } finally {
    loading.value = false;
  }
};

const startPolling = () => {
  // 清除之前的轮询
  if (pollInterval.value) {
    clearInterval(pollInterval.value);
  }
  
  // 每2秒检查一次二维码状态
  pollInterval.value = setInterval(async () => {
    try {
      const response = await checkQRStatus(qrKey.value);
      if (response.data.status === 'success') {
        // 登录成功
        clearInterval(pollInterval.value);
        clearInterval(refreshInterval.value);
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        isLoggedIn.value = true;
        emit('login-success', response.data.user);
        ElMessage.success('登录成功');
      } else if (response.data.status === 'error') {
        // 登录错误
        clearInterval(pollInterval.value);
        clearInterval(refreshInterval.value);
        ElMessage.error(response.data.message || '登录失败，请重新尝试');
        // 重新获取二维码
        setTimeout(() => {
          getQRCode();
        }, 1000);
      } else if (response.data.code === 86038) {
        // 二维码过期，重新获取
        clearInterval(pollInterval.value);
        clearInterval(refreshInterval.value);
        getQRCode();
        ElMessage.warning('二维码已过期，正在重新获取');
      } else if (response.data.code === 86101) {
        // 请扫描二维码
        console.log('请扫描二维码...');
      } else if (response.data.code === 86090) {
        // 请在App中确认登录
        console.log('请在B站App中确认登录');
      } else {
        // 其他状态
        console.log('二维码状态:', response.data.code, response.data.message);
      }
    } catch (error) {
      console.error('检查二维码状态失败:', error);
      // 网络错误，重新获取二维码
      clearInterval(pollInterval.value);
      clearInterval(refreshInterval.value);
      ElMessage.error('网络错误，请重新尝试');
      setTimeout(() => {
        getQRCode();
      }, 1000);
    }
  }, 2000);
};

const startRefreshTimer = () => {
  // 清除之前的定时器
  if (refreshInterval.value) {
    clearInterval(refreshInterval.value);
  }
  
  // 重置倒计时
  timeLeft.value = 60;
  
  // 开始倒计时
  refreshInterval.value = setInterval(() => {
    timeLeft.value--;
    if (timeLeft.value <= 0) {
      // 倒计时结束，刷新二维码
      clearInterval(refreshInterval.value);
      getQRCode();
      ElMessage.info('二维码已自动刷新');
    }
  }, 1000);
};

onMounted(() => {
  // 检查是否已登录
  const token = localStorage.getItem('token');
  if (token) {
    isLoggedIn.value = true;
  } else {
    getQRCode();
  }
});

onUnmounted(() => {
  // 清除轮询和定时器
  if (pollInterval.value) {
    clearInterval(pollInterval.value);
  }
  if (refreshInterval.value) {
    clearInterval(refreshInterval.value);
  }
});
</script>

<style scoped>
.login-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(5px);
}

.login-container {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  padding: 30px;
  width: 400px;
  max-width: 90%;
  animation: fadeIn 0.3s ease;
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
}

.login-header h2 {
  margin-bottom: 10px;
  color: #00a1d6;
  font-size: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.login-header p {
  color: #666;
  font-size: 14px;
}

.login-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 30px;
}

.qr-code-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
}

.qr-code {
  width: 200px;
  height: 200px;
  border: 1px solid #e0e0e0;
  padding: 10px;
  border-radius: 8px;
}

.qr-tip {
  color: #666;
  font-size: 14px;
  text-align: center;
}

.qr-expire {
  color: #999;
  font-size: 12px;
  text-align: center;
  margin-top: 5px;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
  padding: 40px 0;
}

.loading span {
  color: #666;
  font-size: 14px;
}

.login-footer {
  display: flex;
  justify-content: center;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>