<template>
  <el-card class="data-block-card">
    <div class="data-content">
      <el-statistic
        :title="scoreTitle"
        :value="value"
        :precision="0"
        class="main-statistic"
      >
        <template #title>
          <div class="statistic-title">{{ scoreTitle }}</div>
        </template>
        <template #suffix>
          <i class="fa-solid fa-arrow-trend-up trend-icon"></i>
        </template>
      </el-statistic>
      <div class="status-info">
        <div class="danmu-status">
          弹幕状态：<span :class="['status-badge', getStatusClass(danmuStatus)]">{{ danmuStatus }}</span>
        </div>
        <div class="upload-time" v-if="uploadTime">
          上传时间：<span class="time-text">{{ formatUploadTime(uploadTime) }}</span>
        </div>
      </div>
    </div>
  </el-card>
</template>

<script setup>
const props = defineProps({
  value: Number,
  danmuStatus: {
    type: String,
    default: '良好'
  },
  scoreTitle: {
    type: String,
    default: '整体评分'
  },
  uploadTime: {
    type: String,
    default: ''
  }
});

const formatUploadTime = (time) => {
  if (!time) return '';
  // 处理不同格式的时间字符串
  let date;
  if (typeof time === 'string') {
    // 尝试直接解析
    date = new Date(time);
    // 如果解析失败，尝试处理不同格式
    if (isNaN(date.getTime())) {
      // 尝试处理 "2026-03-03 06:56:11.935" 格式
      const parts = time.split(/[- :.]/);
      if (parts.length >= 6) {
        date = new Date(
          parseInt(parts[0]),
          parseInt(parts[1]) - 1,
          parseInt(parts[2]),
          parseInt(parts[3]),
          parseInt(parts[4]),
          parseInt(parts[5])
        );
      }
    }
  } else {
    date = new Date(time);
  }
  
  if (isNaN(date.getTime())) {
    return '';
  }
  
  // 手动构建本地时间字符串
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  
  return `${year}/${month}/${day} ${hours}:${minutes}:${seconds}`;
};

const getStatusClass = (status) => {
  switch (status) {
    case '良好':
      return 'status-good';
    case '欠佳':
      return 'status-poor';
    case '无弹幕':
      return 'status-none';
    default:
      return 'status-poor';
  }
};
</script>

<style scoped>
.data-block-card {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
}

.data-block-card:hover {
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.09);
}

.data-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  /* padding: 20px; */
}

.main-statistic {
  width: 100%;
  text-align: center;
}

.statistic-title {
  font-size: 1.2rem;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 10px;
}

:deep(.el-statistic__content) {
  font-size: 2.5rem;
  font-weight: bold;
  color: #3b82f6;
  margin: 10px 0;
}

.trend-icon {
  color: #10b981;
  font-size: 1.2rem;
  margin-left: 8px;
}

.status-info {
  margin-top: 15px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.danmu-status {
  font-size: 0.9rem;
  color: #6b7280;
}

.upload-time {
  font-size: 0.8rem;
  color: #6b7280;
}

.time-text {
  font-weight: 500;
  color: #4b5563;
}

.status-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-weight: 500;
  font-size: 0.8rem;
  margin-left: 5px;
}

.status-good {
  background-color: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.status-poor {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

.status-none {
  background-color: rgba(107, 114, 128, 0.1);
  color: #6b7280;
}
</style>


