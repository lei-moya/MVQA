<template>
  <div class="upload-container">
    <div class="url-upload-section">
      <h3 class="section-title">
        <i class="fa-solid fa-link"></i> 视频 URL 上传
      </h3>
      <el-input
        v-model="inputUrl"
        placeholder="输入 B 站视频网址..."
        class="url-input"
        :disabled="uploading"
      >
        <template #append>
          <el-button type="primary" @click="handleUrlSubmit" :disabled="uploading || !inputUrl" class="url-submit-btn">
            <i class="fa-solid fa-play"></i> 解析
          </el-button>
        </template>
      </el-input>
    </div>
    
    <div class="file-upload-section">
      <h3 class="section-title">
        <i class="fa-solid fa-file-upload"></i> 文件上传
        <div class="upload-tip">
          仅支持mp4和ass
        </div>
      </h3>
      <div class="upload-controls">
        <el-upload
          class="upload-demo"
          action="#"
          :auto-upload="false"
          :multiple="true"
          :limit="6"
          :on-change="handleFileChange"
          :on-exceed="handleExceed"
          :file-list="fileList"
          :before-upload="beforeUpload"
          accept=".mp4,.ass"
          :show-file-list="false"
        >
          <el-button type="primary" :disabled="uploading" class="select-file-btn">
            <i class="fa-solid fa-folder-open"></i> 选择文件
          </el-button>
        </el-upload>
        <el-button
          type="success"
          @click="handleFileUpload"
          :disabled="uploading || !validateFiles()"
          class="upload-btn"
        >
          <i class="fa-solid fa-arrow-up"></i> 开始上传
        </el-button>
        <el-button
          type="warning"
          @click="clearFiles"
          :disabled="uploading || fileList.length === 0"
          class="clear-btn"
        >
          <i class="fa-solid fa-trash-can"></i> 清空文件
        </el-button>
      </div>
      
      <div class="upload-content">
        <!-- 文件组列表 -->
        <div v-if="fileList.length === 0" class="empty-state">
          <i class="fa-solid fa-file-circle-exclamation"></i>
          <p>未选择文件</p>
        </div>
        <ul v-else class="group-list">
          <li v-for="(group, index) in fileGroups" :key="index" class="group-item">
            <div class="file-item">
              <div class="file-info">
                <i class="fa-solid fa-file-video file-icon"></i>
                <span class="file-name">{{ truncateFileName(group.baseName) }}</span>
              </div>
              <div class="file-status">
                <div class="status-item" title="视频文件">
                  <span
                    class="status-dot"
                    :class="{ 'green': group.hasVideo }"
                  ></span>
                  <span class="status-label">视频</span>
                </div>
                <div class="status-item" title="弹幕文件">
                  <span
                    class="status-dot"
                    :class="{ 'red': group.hasDanmu }"
                  ></span>
                  <span class="status-label">弹幕</span>
                </div>
              </div>
              <div class="file-actions">
                <button class="delete-btn" type="button" @click.prevent="removeGroupFiles(group.baseName)" title="删除文件组">
                  <i class="fa-solid fa-trash"></i>
                </button>
              </div>
            </div>
          </li>
        </ul>
      </div>
      
      <el-progress
        v-if="uploading"
        :percentage="uploadProgress"
        :stroke-width="8"
        :format="formatProgress"
        class="upload-progress"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { uploadVideo, uploadVideoByUrl } from '../api/index.js';
import { ElMessage } from 'element-plus';

const inputUrl = ref('');
const fileList = ref([]);
const uploading = ref(false);
const uploadProgress = ref(0);

const emit = defineEmits(['upload']);

const handleUrlSubmit = async () => {
  if (!inputUrl.value) {
    ElMessage.warning('请输入视频 URL');
    return;
  }
  
  // 验证 URL 格式
  const urlPattern = /^https?:\/\/(www\.)?bilibili\.com\/video\/BV[0-9A-Za-z]+/;
  if (!urlPattern.test(inputUrl.value)) {
    ElMessage.warning('请输入有效的 B 站视频 URL');
    return;
  }
  
  uploading.value = true;
  uploadProgress.value = 0;
  
  try {
    ElMessage.info('正在下载视频，请稍候...');
    // 模拟进度更新
    const progressInterval = setInterval(() => {
      if (uploadProgress.value < 90) {
        uploadProgress.value += 10;
      }
    }, 2000);
    
    const response = await uploadVideoByUrl(inputUrl.value);
    
    clearInterval(progressInterval);
    uploadProgress.value = 100;
    
    ElMessage.success('视频 URL 上传成功，正在分析...');
    emit('upload', { type: 'url', video: response.data });
    inputUrl.value = ''; // 清空输入框
    
    // 延迟重置进度条，让用户看到100%的状态
    setTimeout(() => {
      uploadProgress.value = 0;
    }, 1000);
  } catch (error) {
    ElMessage.error(`URL 上传失败: ${error.response?.data?.detail || '网络错误'}`);
    console.error('URL upload error:', error);
  } finally {
    uploading.value = false;
  }
};

const handleFileChange = (file, newFileList) => {
  // 过滤掉非视频和非弹幕文件，检查文件大小
  const validNewFiles = newFileList.filter(newFile => {
    if (!isVideoFile(newFile) && !isDanmuFile(newFile)) {
      ElMessage.warning(`文件 ${newFile.name} 不是支持的文件类型`);
      return false;
    }
    
    if (newFile.size > 2 * 1024 * 1024 * 1024) { // 2GB
      ElMessage.warning(`文件 ${newFile.name} 超过2GB限制`);
      return false;
    }
    
    return true;
  });
  
  // 处理同名同后缀文件覆盖
  const updatedFileList = [...fileList.value];
  
  validNewFiles.forEach(newFile => {
    // 检查是否存在同名同后缀文件
    const existingIndex = updatedFileList.findIndex(f => f.name === newFile.name);
    if (existingIndex !== -1) {
      // 覆盖现有文件
      updatedFileList[existingIndex] = newFile;
    } else {
      // 添加新文件
      updatedFileList.push(newFile);
    }
  });
  
  // 检查视频文件和弹幕文件数量
  const videoFiles = updatedFileList.filter(f => isVideoFile(f));
  const danmuFiles = updatedFileList.filter(f => isDanmuFile(f));
  
  if (videoFiles.length > 3) {
    ElMessage.warning('最多只能上传3个视频文件');
    // 只保留前3个视频文件
    const filteredFiles = updatedFileList.filter(f => !isVideoFile(f) || videoFiles.indexOf(f) < 3);
    fileList.value = filteredFiles;
    return;
  }
  
  if (danmuFiles.length > 3) {
    ElMessage.warning('最多只能上传3个弹幕文件');
    // 只保留前3个弹幕文件
    const filteredFiles = updatedFileList.filter(f => !isDanmuFile(f) || danmuFiles.indexOf(f) < 3);
    fileList.value = filteredFiles;
    return;
  }
  
  // 更新文件列表
  fileList.value = updatedFileList;
};

const handleExceed = (files, fileList) => {
  ElMessage.warning('最多只能上传3个视频文件和3个弹幕文件，总共6个文件');
};

const removeGroupFiles = (baseName) => {
  // 移除指定基础文件名的所有文件
  fileList.value = fileList.value.filter(item => {
    const itemBaseName = item.name.replace(/\.(mp4|ass)$/, '');
    return itemBaseName !== baseName;
  });
  ElMessage.success('文件组已删除');
};

const clearFiles = (event) => {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  fileList.value = [];
  ElMessage.success('文件已清空');
};

const formatProgress = (percentage) => {
  return `${percentage.toFixed(0)}%`;
};

// 截断文件名，只显示前6个字符
const truncateFileName = (name) => {
  if (name.length <= 7) return name;
  return name.substring(0, 7) + '...';
};

// 判断是否为视频文件
const isVideoFile = (file) => {
  return file.name.endsWith('.mp4');
};

// 判断是否为弹幕文件
const isDanmuFile = (file) => {
  return file.name.endsWith('.ass');
};

// 检查视频文件是否有匹配的弹幕文件
const hasMatchingDanmu = (videoFile) => {
  const videoName = videoFile.name.replace('.mp4', '');
  return fileList.value.some(file => file.name === `${videoName}.ass`);
};

// 检查弹幕文件是否有匹配的视频文件
const hasMatchingVideo = (danmuFile) => {
  const danmuName = danmuFile.name.replace('.ass', '');
  return fileList.value.some(file => file.name === `${danmuName}.mp4`);
};

// 检查文件类型
const beforeUpload = (file) => {
  // 检查文件类型
  if (!isVideoFile(file) && !isDanmuFile(file)) {
    ElMessage.warning('只支持 .mp4 和 .ass 文件');
    return false;
  }
  
  return true;
};

// 按文件名分组
const fileGroups = computed(() => {
  const groups = {};
  
  // 首先处理所有文件，按基础文件名分组
  fileList.value.forEach(file => {
    let baseName;
    if (file.name.endsWith('.mp4')) {
      baseName = file.name.replace('.mp4', '');
    } else if (file.name.endsWith('.ass')) {
      baseName = file.name.replace('.ass', '');
    } else {
      return; // 跳过非视频和非弹幕文件
    }
    
    if (!groups[baseName]) {
      groups[baseName] = {
        baseName,
        hasVideo: false,
        hasDanmu: false
      };
    }
    
    if (file.name.endsWith('.mp4')) {
      groups[baseName].hasVideo = true;
    } else if (file.name.endsWith('.ass')) {
      groups[baseName].hasDanmu = true;
    }
  });
  
  // 转换为数组
  return Object.values(groups);
});

// 视频文件数量
const videoFileCount = computed(() => {
  return fileList.value.filter(file => file.name.endsWith('.mp4')).length;
});

// 弹幕文件数量
const danmuFileCount = computed(() => {
  return fileList.value.filter(file => file.name.endsWith('.ass')).length;
});



// 验证文件配对
const validateFiles = () => {
  const videoFiles = fileList.value.filter(file => file.name.endsWith('.mp4'));
  return videoFiles.length > 0;
};

const handleFileUpload = async () => {
  if (fileList.value.length === 0) return;
  
  // 验证文件配对
  if (!validateFiles()) {
    ElMessage.error('至少需要一个视频文件');
    return;
  }
  
  // 检查是否有单独的弹幕文件（没有对应视频文件的弹幕文件）
  const danmuFiles = fileList.value.filter(file => file.name.endsWith('.ass'));
  const singleDanmuFiles = [];
  
  for (const danmuFile of danmuFiles) {
    const videoFileName = danmuFile.name.replace('.ass', '.mp4');
    const hasMatchingVideo = fileList.value.some(file => file.name === videoFileName);
    if (!hasMatchingVideo) {
      singleDanmuFiles.push(danmuFile.name);
    }
  }
  
  // 显示单独弹幕文件的提示信息
  if (singleDanmuFiles.length > 0) {
    ElMessage.warning(`以下弹幕文件没有对应的视频文件，将被忽略: ${singleDanmuFiles.join(', ')}`);
  }
  
  const videoFiles = fileList.value.filter(file => file.name.endsWith('.mp4'));
  
  uploading.value = true;
  uploadProgress.value = 0;
  
  try {
    // 收集成功上传的文件名称
    const uploadedFileNames = [];
    
    // 逐个上传视频文件
    for (let i = 0; i < videoFiles.length; i++) {
      const file = videoFiles[i].raw;
      const formData = new FormData();
      formData.append('video', file);
      
      // 查找对应的弹幕文件
      const danmuFileName = file.name.replace('.mp4', '.ass');
      const danmuFile = fileList.value.find(f => f.name === danmuFileName);
      if (danmuFile) {
        formData.append('danmu', danmuFile.raw);
        uploadedFileNames.push(danmuFileName);
      }
      
      const response = await uploadVideo(formData, (progressEvent) => {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        uploadProgress.value = Math.round((i * 100 + percent) / videoFiles.length);
      });
      
      uploadedFileNames.push(file.name);
      ElMessage.success(`视频 ${file.name} 上传成功，正在分析...`);
      emit('upload', { type: 'file', video: response.data });
    }
    
    // 只移除成功上传的文件，保留单独的弹幕文件
    fileList.value = fileList.value.filter(file => !uploadedFileNames.includes(file.name));
  } catch (error) {
    ElMessage.error('上传失败，请重试');
    console.error('Upload error:', error);
  } finally {
    uploading.value = false;
    uploadProgress.value = 0;
  }
};
</script>

<style scoped>
/* 上传容器 */
.upload-container {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* 区块标题 */
.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 12px 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding-bottom: 8px;
  border-bottom: 2px solid #e5e7eb;
}

.section-title i {
  color: #3b82f6;
  font-size: 18px;
}

/* URL上传部分 */
.url-upload-section {
  background-color: #f9fafb;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.url-upload-section:hover {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.url-input {
  width: 100%;
}

.url-submit-btn {
  transition: all 0.3s ease;
}

.url-submit-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
}

/* 消除输入框和按钮之间的间隙 */
:deep(.el-input-group__append) {
  padding: 0 !important;
  background-color: transparent !important;
  border-left: none !important;
}

:deep(.el-input-group__append .el-button) {
  margin: 0 !important;
  border-radius: 0 4px 4px 0 !important;
}

/* 文件上传部分 */
.file-upload-section {
  background-color: #f9fafb;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.file-upload-section:hover {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* 上传控制按钮 */
.upload-controls {
  display: flex;
  gap: 15px;
  margin-bottom: 12px;
  margin-left: 10px;
  flex-wrap: wrap;
}

.select-file-btn,
.upload-btn,
.clear-btn {
  transition: all 0.3s ease;
}

.select-file-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
}

.upload-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.4);
}

.clear-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.4);
}

/* 上传提示 */
.upload-tip {
  font-size: 14px;
  color: #6b7280;
  padding: 3px 5px;
  background-color: #f3f4f6;
  border-radius: 4px;
  border-inline: 3px solid #3b82f6;
}

/* 空状态 */
.empty-state {
  padding: 5px 0;
  text-align: center;
  color: #9ca3af;
  font-size: 16px;
  background-color: #ffffff;
  border: 2px dashed #e5e7eb;
  border-radius: 8px;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.empty-state:hover {
  border-color: #3b82f6;
  background-color: #f0f9ff;
}

.empty-state i {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.5;
  display: flex;
  align-items: center;
  justify-content: center;
}

.empty-state p {
  margin: 0;
  font-weight: 500;
}

/* 文件分组样式 */
.group-list {
  list-style: none;
  padding: 0;
  margin: 0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.group-item {
  border-bottom: 1px solid #e5e7eb;
  transition: all 0.3s ease;
}

.group-item:hover {
  background-color: #f3f4f6;
}

.group-item:last-child {
  border-bottom: none;
}

.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  background-color: #ffffff;
  transition: all 0.3s ease;
}

.file-item:hover {
  background-color: #f9fafb;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.file-icon {
  font-size: 20px;
  color: #3b82f6;
  flex-shrink: 0;
}

.file-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
  font-size: 14px;
  color: #1f2937;
}

.file-status {
  display: flex;
  gap: 20px;
  margin: 0 20px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #6b7280;
}

.file-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.delete-btn {
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  padding: 8px;
  border-radius: 4px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.delete-btn:hover {
  color: #ef4444;
  background-color: #fef2f2;
  transform: scale(1.05);
}

.delete-btn i {
  font-size: 16px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #d1d5db;
  transition: all 0.3s ease;
  position: relative;
}

.status-dot.green {
  background-color: #10b981;
  box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
  animation: pulse 2s infinite;
}

.status-dot.red {
  background-color: #ef4444;
  box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.3);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(59, 130, 246, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
  }
}

/* 上传进度条 */
.upload-progress {
  margin: 10px 0;
  border-radius: 4px;
  overflow: hidden;
}

:deep(.el-progress__bar) {
  border-radius: 4px;
}

:deep(.el-progress__inner) {
  border-radius: 4px;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  transition: width 0.5s ease;
}

/* 文件统计信息 */
.file-stats {
  padding: 12px 16px;
  background-color: #f3f4f6;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #4b5563;
  font-weight: 500;
}

.file-stats i {
  color: #3b82f6;
  font-size: 16px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .upload-controls {
    flex-direction: column;
  }
  
  .upload-controls button {
    width: 100%;
  }
  
  .file-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .file-status {
    margin: 0;
    width: 100%;
  }
  
  .file-actions {
    align-self: flex-end;
  }
}

/* 确保所有按钮没有默认的margin-left */
:deep(.el-button) {
  margin-left: 0 !important;
}
</style>
