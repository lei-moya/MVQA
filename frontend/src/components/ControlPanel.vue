<template>
  <div class="control-panel">
    <el-card class="control-card upload-card">
      <template #header>
        <div class="card-header">
          <i class="fa-solid fa-sliders header-icon"></i>
          <span class="header-title">数据源控制</span>
        </div>
      </template>
      <FileUpload @upload="handleUpload" />
    </el-card>
    <div class="file-list-container">
      <FileList :files="files" :current-video-id="currentVideoId" @view="handleView" @delete="handleDelete" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import FileUpload from './FileUpload.vue';
import FileList from './FileList.vue';

const props = defineProps({
  files: Array,
  currentVideoId: {
    type: Number,
    default: null
  }
});

const emit = defineEmits(['upload', 'view', 'delete']);

const handleUpload = (filesData) => {
  emit('upload', filesData);
};

const handleView = (video) => {
  emit('view', video);
};

const handleDelete = (videoId) => {
  emit('delete', videoId);
};
</script>

<style scoped>
/* 控制面板容器 */
.control-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
}

/* 控制卡片 */
.control-card {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  transition: all 0.3s ease;
}

.control-card:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
}

/* 上传卡片 */
.upload-card {
  flex-shrink: 0;
}

/* 文件列表容器 */
.file-list-container {
  flex: 1;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  transition: all 0.3s ease;
}

.file-list-container:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
}

/* 卡片头部 */
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 600;
  color: #1f2937;
  padding: 8px 0;
}

.header-icon {
  color: #3b82f6;
  margin-right: 8px;
  font-size: 16px;
}

.header-title {
  font-size: 16px;
  font-weight: 600;
  flex: 1;
}

/* 文件数量徽章 */
.file-count-badge {
  font-size: 14px;
  min-width: 24px;
  height: 24px;
  line-height: 24px;
  background-color: #3b82f6;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
  transition: all 0.3s ease;
}

.file-count-badge:hover {
  transform: scale(1.1);
  box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4);
}

/* 确保FileList组件占满容器高度 */
.file-list-container {
  height: 100%;
}

.file-list-container :deep(.el-card) {
  height: 100%;
  border-radius: 8px;
  overflow: hidden;
}

.file-list-container :deep(.el-card__body) {
  padding: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
}
</style>
