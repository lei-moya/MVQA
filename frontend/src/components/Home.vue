<template>
  <div class="horizontal-layout">
    <div class="layout-item" style="flex: 2;">
      <el-card shadow="hover" class="full-height-card">
        <ChartMain
            :video-src="videoSrc"
            :danmu-src="danmuSrc"
            :line-data="chartData.lineData"
            :time="chartData.time"
            ref="chartMainRef"
            @mouse-enter="handleMouseEnterCanvas"
            @mouse-leave="handleMouseLeaveCanvas"
            @mouse-move="handleMouseMoveCanvas"
            @click="handleCanvasClick"
        />
      </el-card>
    </div>
    <div class="layout-item" style="flex: 0.8; display: flex; flex-direction: column; gap: 16px;">
      <el-card shadow="hover">
        <DataBlock 
          :value="displayData.coreIndex" 
          :danmu-status="displayData.danmuStatus" 
          :score-title="displayData.scoreTitle" 
          :upload-time="currentVideo?.created_at"
        />
      </el-card>
      <el-card shadow="hover" style="flex: 1;">
        <WidgetCard
            title="视频"
            icon="fa-bullseye"
            :data="getRadarData(chartData.radar1Data)"
            type="video"
            :legend-data="getVideoLegendData(chartData.radar1Data)"
        />
      </el-card>
      <el-card shadow="hover" style="flex: 1;">
        <WidgetCard
            title="音频"
            icon="fa-spider"
            :data="getRadarData(chartData.radar2Data)"
            type="audio"
            :legend-data="getAudioLegendData(chartData.radar2Data)"
        />
      </el-card>
    </div>
    <div class="layout-item" style="flex: 1; display: flex; flex-direction: column; height: 100%;">
      <ControlPanel 
        :files="videoFiles" 
        :current-video-id="currentVideo?.id" 
        @upload="handleUpload" 
        @view="handleView" 
        @delete="handleDelete"
      />
    </div>
  </div>
  <InputDialog
      v-model:show="showInputDialog"
      :files="pendingFiles"
      @close="closeInputDialog"
      @submit="submitVideoInfo"
  />
</template>

<script setup>
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue';
import { ElMessage } from 'element-plus';
import InputDialog from './InputDialog.vue';
import ChartMain from './ChartMain.vue';
import DataBlock from './DataBlock.vue';
import WidgetCard from './WidgetCard.vue';
import ControlPanel from './ControlPanel.vue';
import { uploadVideo, getVideoDetail, getVideoList, deleteVideo } from '../api/index.js';

// 状态管理
const videoFiles = ref([]);
const videoSrc = ref('');
const danmuSrc = ref('');
const currentVideo = ref(null);
const isMouseInCanvas = ref(false);
const currentClipIndex = ref(-1);
const chartMainRef = ref(null);
const showInputDialog = ref(false);
const pendingFiles = ref(null);

// 图表数据
const chartData = reactive({
  time: [],
  lineData: [],
  radar1Data: [{ value: [0, 0, 0, 0, 0, 0], name: '视频质量' }],
  radar2Data: [{ value: [0, 0, 0, 0, 0, 0], name: '音频质量' }],
});

// 显示数据
const displayData = reactive({
  coreIndex: 0,
  danmuStatus: '未知',
  scoreTitle: '整体评分',
});

// 定时器管理
let videoLoadTimer = null;
let pollTimer = null;

// 工具函数
const getVideoPlayer = () => {
  return chartMainRef.value?.videoPlayer;
};

const getFilenameFromPath = (filePath) => {
  return filePath?.split(/[\/]/).pop() || '';
};

const generateTimeLabels = (clipCount, totalDuration) => {
  const duration = totalDuration || 100;
  const clipDuration = duration / clipCount;
  
  return Array.from({ length: clipCount }, (_, index) => {
    const timeInSeconds = index * clipDuration;
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  });
};

const processRadarData = (scoreData, name) => {
  const videoRadarData = scoreData.slice(0, 5).map(val => val || 0);
  const audioRadarData = scoreData.slice(5, 10).map(val => val || 0);
  
  return {
    video: [{
      value: [...videoRadarData, 0],
      name: name ? `${name} 视频质量` : '视频质量'
    }],
    audio: [{
      value: [...audioRadarData, 0],
      name: name ? `${name} 音频质量` : '音频质量'
    }]
  };
};

// 数据处理函数
const getRadarData = (radarData) => {
  return radarData[0]?.value || [];
};

const getVideoLegendData = (radarData) => {
  const data = radarData[0]?.value || [];
  return [
    { name: '清晰度', value: (data[0] || 0).toFixed(2), color: '#10b981' },
    { name: '色彩', value: (data[1] || 0).toFixed(2), color: '#3b82f6' },
    { name: '饱和度', value: (data[2] || 0).toFixed(2), color: '#f59e0b' },
    { name: '稳定性', value: (data[3] || 0).toFixed(2), color: '#ef4444' },
    { name: '亮度', value: (data[4] || 0).toFixed(2), color: '#8b5cf6' }
  ];
};

const getAudioLegendData = (radarData) => {
  const data = radarData[0]?.value || [];
  return [
    { name: '音量', value: (data[0] || 0).toFixed(2), color: '#8b5cf6' },
    { name: '音质', value: (data[1] || 0).toFixed(2), color: '#ec4899' },
    { name: '噪音', value: (data[2] || 0).toFixed(2), color: '#8b5cf6' },
    { name: '清晰度', value: (data[3] || 0).toFixed(2), color: '#ec4899' },
    { name: '均衡', value: (data[4] || 0).toFixed(2), color: '#8b5cf6' }
  ];
};

const setVideoSource = (filePath) => {
  if (filePath && currentVideo.value) {
    // 获取认证令牌
    const token = localStorage.getItem('token');
    // 为视频流URL添加token参数
    videoSrc.value = `http://localhost:8000/api/videos/${currentVideo.value.id}/stream?token=${token}`;
    
    if (currentVideo.value.danmu_path) {
      const danmuFilename = getFilenameFromPath(currentVideo.value.danmu_path);
      danmuSrc.value = `http://localhost:8000/${danmuFilename}`;
    } else {
      danmuSrc.value = '';
    }
    
    videoLoadTimer = setTimeout(() => {
      const videoPlayer = getVideoPlayer();
      if (videoPlayer) {
        updateTimeLabels(videoPlayer.duration);
      }
    }, 1000);
  } else {
    videoSrc.value = '';
    danmuSrc.value = '';
    ElMessage.warning('视频文件路径不存在');
  }
};

const updateDisplayData = (video) => {
  currentVideo.value = video;
  
  const videoScore = video.video_score || [];
  const overallScore = videoScore.length > 10 ? videoScore[10] : 0;
  const danmuScore = videoScore.length > 11 ? videoScore[11] : null;
  
  displayData.coreIndex = overallScore;
  displayData.danmuStatus = danmuScore === null ? '无弹幕' : (danmuScore < 0 ? '欠佳' : '良好');
  
  const radarData = processRadarData(videoScore);
  chartData.radar1Data = radarData.video;
  chartData.radar2Data = radarData.audio;
  
  const clipScores = video.clip_scores || [];
  if (clipScores.length > 0) {
    chartData.lineData = clipScores.map(clip => clip.length > 10 ? clip[10] : 0);
    
    if (chartData.time.length === 0 || isMouseInCanvas.value) {
      chartData.time = generateTimeLabels(clipScores.length, 100);
    }
  } else {
    chartData.lineData = [overallScore];
    chartData.time = ['视频总分'];
  }
};

const updateTimeLabels = (totalDuration) => {
  if (!currentVideo.value) return;
  
  const clipScores = currentVideo.value.clip_scores || [];
  if (clipScores.length > 0) {
    chartData.time = generateTimeLabels(clipScores.length, totalDuration);
  }
};

// 事件处理函数
const handleVideoUpload = (video) => {
  if (!video || typeof video !== 'object' || !video.filename || !video.id) {
    ElMessage.error('无效的视频数据');
    return;
  }
  
  const existingIndex = videoFiles.value.findIndex(f => f.filename === video.filename);
  
  if (existingIndex !== -1) {
    videoFiles.value[existingIndex] = video;
  } else {
    videoFiles.value.unshift(video);
  }
  
  pollVideoStatus(video.id);
};

const handleUpload = async (uploadData) => {
  if (!uploadData || typeof uploadData !== 'object') {
    ElMessage.error('无效的上传数据');
    return;
  }
  
  if (uploadData.video) {
    handleVideoUpload(uploadData.video);
  } else {
    ElMessage.warning('未提供视频数据');
  }
};

const pollVideoStatus = async (videoId) => {
  // 检查是否已登录
  const token = localStorage.getItem('token');
  if (!token) {
    console.log('用户未登录，跳过视频状态检查');
    return;
  }
  
  const checkStatus = async () => {
    try {
      const response = await getVideoDetail(videoId);
      const video = response.data;
      
      const index = videoFiles.value.findIndex(f => f.id === videoId);
      if (index !== -1) {
        videoFiles.value[index] = video;
      }
      
      if (video.status === 'completed' || video.status === 'failed') {
        if (video.status === 'completed') {
          ElMessage.success('视频分析完成');
          updateDisplayData(video);
          setVideoSource(video.file_path);
          // 重新加载视频列表，确保所有视频都显示在列表中
          loadVideoList();
        } else {
          ElMessage.error('视频分析失败');
        }
        return;
      }
      
      pollTimer = setTimeout(checkStatus, 2000);
    } catch (error) {
      console.error('Error checking video status:', error);
    }
  };
  
  checkStatus();
};

const handleView = (video) => {
  if (!video || typeof video !== 'object') {
    ElMessage.error('无效的视频数据');
    return;
  }
  
  cleanupTimers();
  updateDisplayData(video);
  setVideoSource(video.file_path);
};

const cleanupTimers = () => {
  if (videoLoadTimer) {
    clearTimeout(videoLoadTimer);
    videoLoadTimer = null;
  }
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
};

const loadVideoList = async () => {
  // 检查是否已登录
  const token = localStorage.getItem('token');
  if (!token) {
    console.log('用户未登录，跳过视频列表加载');
    return;
  }
  
  try {
    const response = await getVideoList();
    videoFiles.value = response.data;
    console.log('视频列表加载成功:', response.data);
  } catch (error) {
    console.error('Error loading video list:', error);
    // 检查是否是401错误（未授权）
    if (error.response && error.response.status === 401) {
      // 清除本地存储的token和user
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      ElMessage.error('登录已过期，请重新登录');
      // 刷新页面以显示登录组件
      window.location.reload();
    } else {
      ElMessage.error(`加载视频列表失败: ${error.response?.data?.detail || '网络错误'}`);
    }
  }
};

const handleDelete = async (videoId) => {
  if (!videoId) {
    ElMessage.error('视频ID缺失');
    return;
  }
  
  try {
    const isCurrentVideo = currentVideo.value && currentVideo.value.id === videoId;
    
    const videoPlayer = getVideoPlayer();
    if (videoPlayer) {
      videoPlayer.pause();
      videoPlayer.src = '';
      videoPlayer.currentTime = 0;
      // 清空视频源引用，确保文件被释放
      videoSrc.value = '';
      danmuSrc.value = '';
    }
    
    // 等待一小段时间，确保浏览器释放文件资源
    await new Promise(resolve => setTimeout(resolve, 500));
    
    await deleteVideo(videoId);
    videoFiles.value = videoFiles.value.filter(file => file.id !== videoId);
    await loadVideoList();
    
    if (isCurrentVideo) {
      currentVideo.value = null;
      displayData.coreIndex = 0;
      displayData.danmuStatus = '未知';
      chartData.lineData = [];
      chartData.time = [];
      chartData.radar1Data = [{ value: [0, 0, 0, 0, 0, 0], name: '视频质量' }];
      chartData.radar2Data = [{ value: [0, 0, 0, 0, 0, 0], name: '音频质量' }];
    }
    
    ElMessage.success('视频删除成功');
  } catch (error) {
    console.error('Error handling delete:', error);
    ElMessage.error('视频删除失败，请稍后重试');
  }
};

const handleMouseEnterCanvas = () => {
  isMouseInCanvas.value = true;
  displayData.scoreTitle = '片段评分';
};

const handleMouseLeaveCanvas = () => {
  isMouseInCanvas.value = false;
  currentClipIndex.value = -1;
  displayData.scoreTitle = '整体评分';
  if (currentVideo.value) {
    updateDisplayData(currentVideo.value);
  }
};

const handleMouseMoveCanvas = (clipIndex) => {
  if (!currentVideo.value || currentClipIndex.value === clipIndex) return;
  
  currentClipIndex.value = clipIndex;
  const clipScores = currentVideo.value.clip_scores || [];
  
  if (clipIndex >= 0 && clipIndex < clipScores.length) {
    const clipData = clipScores[clipIndex];
    const clipScore = clipData.length > 10 ? clipData[10] : 0;
    const danmuScore = clipData.length > 11 ? clipData[11] : 0;
    
    displayData.coreIndex = clipScore;
    displayData.danmuStatus = danmuScore < 0 ? '欠佳' : '良好';
    
    const radarData = processRadarData(clipData, `片段 ${clipIndex + 1}`);
    chartData.radar1Data = radarData.video;
    chartData.radar2Data = radarData.audio;
  }
};

const handleCanvasClick = (clipIndex) => {
  if (clipIndex < 0) return;
  
  const videoPlayer = getVideoPlayer();
  if (!currentVideo.value || !videoPlayer) {
    ElMessage.warning('无法播放视频，请先选择一个视频');
    return;
  }
  
  const totalDuration = videoPlayer.duration || 100;
  const clipCount = currentVideo.value.clip_scores?.length || 1;
  
  if (clipIndex >= clipCount) return;
  
  const clipDuration = totalDuration / clipCount;
  const startTime = clipIndex * clipDuration;
  
  videoPlayer.currentTime = startTime;
  videoPlayer.play();
  updateTimeLabels(totalDuration);
};

// 生命周期钩子
onMounted(() => {
  loadVideoList();
});

onBeforeUnmount(() => {
  cleanupTimers();
});


</script>

<style scoped>
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
</style>