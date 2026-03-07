<template>
  <el-card shadow="hover" style="height: 100%; display: flex; flex-direction: column;">
    <template #header>
      <div class="card-header">
        <span><i class="fa-solid fa-folder-open"></i> 已上传视频</span>
        <el-badge :value="files.length" type="primary" />
      </div>
    </template>
    <div class="table-container">
      <el-empty v-if="files.length === 0" description="暂无上传视频" style="height: 100%; display: flex; align-items: center; justify-content: center;" />
      <el-table v-else :data="files" style="width: 100%;" :stripe="true" :header-cell-style="{ position: 'sticky', top: 0, zIndex: 0, backgroundColor: '#ffffff' }">
        <el-table-column prop="filename" label="视频名" min-width="150">
          <template #default="scope">
            <div class="file-info" :class="{ 'current-video': scope.row.id === currentVideoId }">
              <i v-if="scope.row.id === currentVideoId" class="fa-solid fa-play-circle current-video-icon"></i>
              <i v-else class="fa-regular fa-file-video"></i>
              <span class="file-name" :title="scope.row.filename">{{ getFileNameWithoutExtension(scope.row.filename) }}</span>
              <span v-if="scope.row.id === currentVideoId" class="current-video-badge">当前</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="涨幅" width="80">
          <template #default="scope">
            <span v-if="scope.row.score_change !== null" class="score-change" :class="getScoreChangeClass(scope.row.score_change)">
              <i :class="getScoreChangeIcon(scope.row.score_change)"></i>
              <span>{{ scope.row.score_change.toFixed(2) }}</span>
            </span>
            <span v-else class="score-change none">
              <i class="fa-solid fa-minus"></i>
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="80">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ getStatusText(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="60">
          <template #default="scope">
            <div style="display: flex; flex-direction: column; gap: 4px; align-items: stretch;">
              <el-button 
                v-if="scope.row.status === 'completed'" 
                type="primary" 
                size="small"
                @click="viewDetails(scope.row)"
                style="width: 100%; margin: 0;"
              >
                查看
              </el-button>
              <el-button 
                type="danger" 
                size="small"
                @click="deleteVideo(scope.row)"
                style="width: 100%; margin: 0;"
              >
                删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </el-card>
</template>

<script setup>
import { ElMessage, ElMessageBox } from 'element-plus';

const props = defineProps({
  files: Array,
  currentVideoId: {
    type: Number,
    default: null
  }
});

const emit = defineEmits(['view', 'delete']);

const getStatusType = (status) => {
  switch (status) {
    case 'pending':
      return 'info';
    case 'processing':
      return 'warning';
    case 'completed':
      return 'success';
    case 'failed':
      return 'danger';
    default:
      return 'info';
  }
};

const getStatusText = (status) => {
  switch (status) {
    case 'pending':
      return '待处理';
    case 'processing':
      return '处理中';
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
    default:
      return status;
  }
};

const viewDetails = (video) => {
  emit('view', video);
};

const getFileNameWithoutExtension = (filename) => {
  let nameWithoutExt = filename.replace(/\.[^/.]+$/, '');
  // 只显示前9个字符
  if (nameWithoutExt.length > 9) {
    nameWithoutExt = nameWithoutExt.substring(0, 9) + '...';
  }
  return nameWithoutExt;
};

const deleteVideo = async (video) => {
  try {
    // 显示确认对话框
    await ElMessageBox.confirm(
      `确定要删除视频 "${getFileNameWithoutExtension(video.filename)}" 吗？`,
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    );
    
    // 通知父组件处理删除
    emit('delete', video.id);
  } catch (error) {
    // 如果用户取消删除，不显示错误消息
    if (error !== 'cancel') {
      ElMessage.error('删除失败，请重试');
      console.error('Delete error:', error);
    }
  }
};

// 获取涨幅样式类
const getScoreChangeClass = (scoreChange) => {
  if (scoreChange > 0) {
    return 'increase';
  } else if (scoreChange < 0) {
    return 'decrease';
  } else {
    return 'no-change';
  }
};

// 获取涨幅图标
const getScoreChangeIcon = (scoreChange) => {
  if (scoreChange > 0) {
    return 'fa-solid fa-arrow-trend-up';
  } else if (scoreChange < 0) {
    return 'fa-solid fa-arrow-trend-down';
  } else {
    return 'fa-solid fa-minus';
  }
};
</script>

<style scoped>
/* 卡片头部 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: #1f2937;
  padding: 8px 0;
}

.card-header i {
  color: #3b82f6;
  margin-right: 8px;
  font-size: 16px;
}

/* 徽章样式 */
:deep(.el-badge__content) {
  font-size: 14px;
  min-width: 24px;
  height: 24px;
  line-height: 24px;
  background-color: #3b82f6;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
  transition: all 0.3s ease;
}

:deep(.el-badge__content:hover) {
  transform: scale(1.1);
  box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4);
}

/* 文件信息 */
.file-info {
  display: flex;
  align-items: center;
  gap: 8px;
  position: relative;
}

.file-info i {
  color: #3b82f6;
  font-size: 16px;
  flex-shrink: 0;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  color: #1f2937;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 当前视频样式 */
.file-info.current-video {
  background-color: rgba(59, 130, 246, 0.1);
  padding: 4px 8px;
  border-radius: 6px;
  border-left: 3px solid #3b82f6;
  transition: all 0.3s ease;
}

.file-info.current-video:hover {
  background-color: rgba(59, 130, 246, 0.15);
  transform: translateX(2px);
}

.current-video-icon {
  color: #3b82f6 !important;
  font-size: 18px !important;
  animation: pulse 2s infinite;
}

.current-video-badge {
  background-color: #3b82f6;
  color: white;
  font-size: 12px;
  font-weight: bold;
  padding: 2px 6px;
  border-radius: 10px;
  margin-left: auto;
  flex-shrink: 0;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
  transition: all 0.3s ease;
}

.current-video-badge:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4);
}

@keyframes pulse {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
  100% {
    transform: scale(1);
  }
}

/* 涨幅样式 */
.score-change {
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px;
  border-radius: 10px;
  font-weight: bold;
  transition: all 0.3s ease;
}

.score-change:hover {
  transform: scale(1.05);
}

.score-change.increase {
  color: #10b981;
  background-color: rgba(16, 185, 129, 0.1);
}

.score-change.decrease {
  color: #ef4444;
  background-color: rgba(239, 68, 68, 0.1);
}

.score-change.no-change {
  color: #6b7280;
  background-color: rgba(107, 114, 128, 0.1);
}

.score-change.none {
  color: #9ca3af;
  display: flex;
  align-items: center;
  gap: 4px;
}

/* 分数样式 */
.score {
  font-weight: bold;
  color: #3b82f6;
}

.no-score {
  color: #9ca3af;
}

/* 表格容器 */
.table-container {
  flex: 1;
  overflow: auto;
  height: 100%;
  border-radius: 0 0 8px 8px;
}

/* 自定义滚动条 */
.table-container::-webkit-scrollbar {
  width: 6px;
}

.table-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.table-container::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 3px;
  transition: background 0.3s ease;
}

.table-container::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}

/* 表格样式 */
:deep(.el-table) {
  border-radius: 0 0 8px 8px;
  overflow: hidden;
}

:deep(.el-table__row) {
  transition: all 0.3s ease;
}

:deep(.el-table__row:hover) {
  background-color: #f3f4f6 !important;
  transform: translateX(4px);
}

:deep(.el-table__header th) {
  background-color: #f9fafb !important;
  font-weight: 600;
  color: #1f2937;
  border-bottom: 2px solid #e5e7eb !important;
}

:deep(.el-table__cell) {
  border-bottom: 1px solid #f3f4f6 !important;
}

/* 按钮样式 */
:deep(.el-button) {
  transition: all 0.3s ease;
  border-radius: 4px;
}

:deep(.el-button:hover) {
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

:deep(.el-button--primary:hover) {
  box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
}

:deep(.el-button--danger:hover) {
  box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.4);
}

/* 标签样式 */
:deep(.el-tag) {
  border-radius: 4px;
  font-size: 12px;
  padding: 2px 8px;
  transition: all 0.3s ease;
}

:deep(.el-tag:hover) {
  transform: scale(1.05);
}

/* 空状态 */
:deep(.el-empty) {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f9fafb;
  border-radius: 0 0 8px 8px;
}

:deep(.el-empty__description) {
  color: #9ca3af;
  font-size: 14px;
}

:deep(.el-empty__image) {
  width: 80px;
  height: 80px;
}
</style>
