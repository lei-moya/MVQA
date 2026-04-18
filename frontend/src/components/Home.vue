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
        :total-count="listTotal"
        :loading-more="listLoading"
        :has-more="listHasMore"
        :filter-status="listFilterStatus"
        :filter-filename="listFilterFilename"
        @upload="handleUpload" 
        @view="handleView" 
        @delete="handleDelete"
        @reanalyze="handleReanalyze"
        @load-more="loadMoreVideos"
        @filters-change="onListFiltersChange"
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
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue';
import { ElMessage } from 'element-plus';
import InputDialog from './InputDialog.vue';
import ChartMain from './ChartMain.vue';
import DataBlock from './DataBlock.vue';
import WidgetCard from './WidgetCard.vue';
import ControlPanel from './ControlPanel.vue';
import {
  getVideoDetail,
  getVideoList,
  deleteVideo,
  reanalyzeVideo,
  videoStreamUrl,
  uploadsPublicUrl,
  VIDEO_LIST_PAGE_SIZE,
  getApiErrorDetail,
} from '../api/index.js';

// 状态管理
const videoFiles = ref([]);
const listTotal = ref(0);
const listLoading = ref(false);
const listHasMore = computed(
  () => listTotal.value > 0 && videoFiles.value.length < listTotal.value
);
const videoSrc = ref('');
const danmuSrc = ref('');
const currentVideo = ref(null);
const isMouseInCanvas = ref(false);
const currentClipIndex = ref(-1);
const chartMainRef = ref(null);
const showInputDialog = ref(false);
const pendingFiles = ref(null);
const listFilterStatus = ref('');
const listFilterFilename = ref('');
/** 列表固定按上传时间降序，与后端默认 sort 一致 */
const LIST_DEFAULT_SORT = 'created_desc';

// 图表数据
const chartData = reactive({
  time: [],
  lineData: [],
  radar1Data: [{ value: [0, 0, 0, 0, 0], name: '视频质量' }],
  radar2Data: [{ value: [0, 0, 0, 0, 0], name: '音频质量' }],
});

// 显示数据
const displayData = reactive({
  coreIndex: 0,
  danmuStatus: '未知',
  scoreTitle: '整体评分',
});

// 定时器：视频加载延迟；每个 videoId 独立轮询，避免多任务上传时互相覆盖
let videoLoadTimer = null;
/** @type {Map<number, ReturnType<typeof setTimeout>>} */
const activeVideoPolls = new Map();
/** 任务状态轮询连续失败次数（避免偶发网络错误反复弹窗） @type {Map<number, number>} */
const videoPollFailureCount = new Map();
const VIDEO_POLL_FAIL_THRESHOLD = 5;

// 工具函数
const getVideoPlayer = () => {
  return chartMainRef.value?.videoPlayer;
};

// 节流函数
const throttle = (func, delay) => {
  let timeoutId;
  return function(...args) {
    if (!timeoutId) {
      timeoutId = setTimeout(() => {
        func.apply(this, args);
        timeoutId = null;
      }, delay);
    }
  };
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
  // 确保scoreData是数组
  const safeScoreData = Array.isArray(scoreData) ? scoreData : [];
  // 处理嵌套数组的情况
  const finalScoreData = Array.isArray(safeScoreData[0]) ? safeScoreData[0] : safeScoreData;
  // 确保值是数字
  const videoRadarData = finalScoreData.slice(0, 5).map(val => Number(val) || 0);
  const audioRadarData = finalScoreData.slice(5, 10).map(val => Number(val) || 0);
  
  return {
    video: [{
      value: videoRadarData,
      name: name ? `${name} 视频质量` : '视频质量'
    }],
    audio: [{
      value: audioRadarData,
      name: name ? `${name} 音频质量` : '音频质量'
    }]
  };
};

// 数据处理函数
const getRadarData = (radarData) => {
  const data = radarData[0]?.value || [];
  // 确保返回的是正确的评分数据数组
  const finalData = Array.isArray(data[0]) ? data[0] : data;
  // 确保返回的数据长度为5，与WidgetCard中的指示器数量匹配
  return finalData.slice(0, 5);
};

const getVideoLegendData = (radarData) => {
  const data = radarData[0]?.value || [];
  // 确保data[0]是数组
  const scoreData = Array.isArray(data[0]) ? data[0] : data;
  return [
    { name: '清晰度', value: (Number(scoreData[0]) || 0).toFixed(2), color: '#10b981' },
    { name: '色彩', value: (Number(scoreData[1]) || 0).toFixed(2), color: '#3b82f6' },
    { name: '饱和度', value: (Number(scoreData[2]) || 0).toFixed(2), color: '#f59e0b' },
    { name: '稳定性', value: (Number(scoreData[3]) || 0).toFixed(2), color: '#ef4444' },
    { name: '亮度', value: (Number(scoreData[4]) || 0).toFixed(2), color: '#8b5cf6' }
  ];
};

const getAudioLegendData = (radarData) => {
  const data = radarData[0]?.value || [];
  // 确保data[0]是数组
  const scoreData = Array.isArray(data[0]) ? data[0] : data;
  return [
    { name: '音量', value: (Number(scoreData[0]) || 0).toFixed(2), color: '#8b5cf6' },
    { name: '音质', value: (Number(scoreData[1]) || 0).toFixed(2), color: '#ec4899' },
    { name: '噪音', value: (Number(scoreData[2]) || 0).toFixed(2), color: '#8b5cf6' },
    { name: '清晰度', value: (Number(scoreData[3]) || 0).toFixed(2), color: '#ec4899' },
    { name: '均衡', value: (Number(scoreData[4]) || 0).toFixed(2), color: '#8b5cf6' }
  ];
};

/** 根据 ``currentVideo`` 设置流地址与弹幕（不依赖服务端返回的绝对路径是否存在） */
const setVideoSource = () => {
  if (!currentVideo.value?.id) {
    videoSrc.value = '';
    danmuSrc.value = '';
    ElMessage.warning('无效的视频记录');
    return;
  }
  const token = localStorage.getItem('token');
  videoSrc.value = videoStreamUrl(currentVideo.value.id, token);

  if (currentVideo.value.danmu_path) {
    danmuSrc.value = uploadsPublicUrl(currentVideo.value.danmu_path);
  } else {
    danmuSrc.value = '';
  }

  videoLoadTimer = setTimeout(() => {
    const videoPlayer = getVideoPlayer();
    if (videoPlayer) {
      updateTimeLabels(videoPlayer.duration);
    }
  }, 1000);
};

const updateDisplayData = (video) => {

  
  currentVideo.value = video;
  
  // 检查视频状态
  if (video.status === 'failed') {
    // 分析失败时的处理
    displayData.coreIndex = 0;
    displayData.danmuStatus = '分析失败';
    
    // 显示失败信息
    chartData.radar1Data = [{ value: [0, 0, 0, 0, 0, 0], name: '视频质量' }];
    chartData.radar2Data = [{ value: [0, 0, 0, 0, 0, 0], name: '音频质量' }];
    chartData.lineData = [0];
    chartData.time = ['分析失败'];
    

    return;
  }
  
  // 处理视频评分数据
  const videoScore = video.video_score || [];
  // 确保videoScore是数组
  const safeVideoScore = Array.isArray(videoScore) ? videoScore : [];
  
  // 处理嵌套数组的情况
  const score = Array.isArray(safeVideoScore[0]) ? safeVideoScore[0] : safeVideoScore;
  
  // 确保值是数字
  const overallScore = score.length > 10 ? Number(score[10]) || 0 : 0;
  const danmuScore = score.length > 11 ? Number(score[11]) || null : null;
  
  displayData.coreIndex = overallScore;
  displayData.danmuStatus = danmuScore === null ? '无弹幕' : (danmuScore < 0 ? '欠佳' : '良好');
  
  const radarData = processRadarData(score);
  
  chartData.radar1Data = radarData.video;
  chartData.radar2Data = radarData.audio;
  
  // 处理片段评分数据
  const clipScores = video.clip_scores || [];
  
  // 确保clipScores是数组
  const safeClipScores = Array.isArray(clipScores) ? clipScores : [];
  
  // 检查clipScores的结构，确保它是二维数组
  let processedClipScores = [];
  if (safeClipScores.length > 0) {
    if (Array.isArray(safeClipScores[0])) {
      // 已经是二维数组，直接使用
      processedClipScores = safeClipScores;
    } else {
      // 如果是一维数组，可能是模型返回的单个片段数据
      // 这是一个片段的评分数据，将其包装为二维数组
      processedClipScores = [safeClipScores];
    }
  }
  
  if (processedClipScores.length > 0) {
    // 有片段数据时，显示片段级数据
    chartData.lineData = processedClipScores.map(clip => {
      // 确保clip是数组
      const safeClip = Array.isArray(clip) ? clip : [];
      // 处理嵌套数组的情况
      const finalClip = Array.isArray(safeClip[0]) ? safeClip[0] : safeClip;
      // 确保值是数字
      const clipScore = finalClip.length > 10 ? Number(finalClip[10]) || 0 : 0;
      return clipScore;
    });
    
    chartData.time = generateTimeLabels(processedClipScores.length, 100);
  } else {
    // 没有片段数据或未选中任何片段时，显示视频级数据
    chartData.lineData = [overallScore];
    chartData.time = ['视频总分'];
  }
};

const updateTimeLabels = (totalDuration) => {
  if (!currentVideo.value) return;
  
  const clipScores = currentVideo.value.clip_scores || [];
  // 确保clipScores是数组
  const safeClipScores = Array.isArray(clipScores) ? clipScores : [];
  
  // 检查clipScores的结构，确保它是二维数组
  let processedClipScores = [];
  if (safeClipScores.length > 0) {
    if (Array.isArray(safeClipScores[0])) {
      // 已经是二维数组，直接使用
      processedClipScores = safeClipScores;
    } else {
      // 如果是一维数组，可能是模型返回的单个片段数据
      // 这是一个片段的评分数据，将其包装为二维数组
      processedClipScores = [safeClipScores];
    }
  }
  
  if (processedClipScores.length > 0) {
    chartData.time = generateTimeLabels(processedClipScores.length, totalDuration);
  }
};

// 事件处理函数
const sortVideoFiles = () => {
  videoFiles.value.sort(
    (a, b) => new Date(b.created_at) - new Date(a.created_at)
  );
};

const handleVideoUpload = (video) => {
  if (!video || typeof video !== 'object' || !video.filename || !video.id) {
    ElMessage.error('无效的视频数据');
    return;
  }
  
  const existingIndex = videoFiles.value.findIndex(f => f.filename === video.filename);
  
  if (existingIndex !== -1) {
    videoFiles.value[existingIndex] = video;
    sortVideoFiles();
  } else {
    videoFiles.value.unshift(video);
    listTotal.value += 1;
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

const clearPollForVideo = (videoId) => {
  const t = activeVideoPolls.get(videoId);
  if (t) {
    clearTimeout(t);
    activeVideoPolls.delete(videoId);
  }
  videoPollFailureCount.delete(videoId);
};

const pollVideoStatus = async (videoId) => {
  const token = localStorage.getItem('token');
  if (!token) {
    return;
  }

  clearPollForVideo(videoId);

  const scheduleNext = () => {
    const tid = setTimeout(checkStatus, 2000);
    activeVideoPolls.set(videoId, tid);
  };

  const checkStatus = async () => {
    try {
      const response = await getVideoDetail(videoId);
      const video = response.data;
      videoPollFailureCount.delete(videoId);

      const index = videoFiles.value.findIndex((f) => f.id === videoId);
      if (index !== -1) {
        videoFiles.value[index] = video;
      } else if (!videoFiles.value.some((f) => f.id === video.id)) {
        videoFiles.value.unshift(video);
      }
      sortVideoFiles();

        if (video.status === 'completed' || video.status === 'failed') {
        clearPollForVideo(videoId);
        if (video.status === 'completed') {
          ElMessage.success('视频分析完成');
          updateDisplayData(video);
          setVideoSource();
        } else {
          const tip =
            Array.isArray(video.suggestions) && video.suggestions.length
              ? String(video.suggestions[0])
              : '视频分析失败';
          ElMessage.error(tip);
          updateDisplayData(video);
        }
        return;
      }

      scheduleNext();
    } catch (error) {
      const n = (videoPollFailureCount.get(videoId) || 0) + 1;
      videoPollFailureCount.set(videoId, n);
      console.warn('轮询任务状态失败:', error);
      if (n >= VIDEO_POLL_FAIL_THRESHOLD) {
        clearPollForVideo(videoId);
        ElMessage.error(
          getApiErrorDetail(error, '任务状态多次查询失败，已停止自动更新')
        );
      } else {
        scheduleNext();
      }
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
  setVideoSource();
};

const cleanupTimers = () => {
  if (videoLoadTimer) {
    clearTimeout(videoLoadTimer);
    videoLoadTimer = null;
  }
  for (const t of activeVideoPolls.values()) {
    clearTimeout(t);
  }
  activeVideoPolls.clear();
  videoPollFailureCount.clear();
};

const handleListApiError = (error) => {
  if (error.response && error.response.status === 401) {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    ElMessage.error('登录已过期，请重新登录');
    window.location.reload();
    return;
  }
  ElMessage.error(`加载视频列表失败: ${getApiErrorDetail(error)}`);
};

const onListFiltersChange = (q) => {
  listFilterStatus.value = q.status ?? '';
  listFilterFilename.value = q.filename ?? '';
  videoFiles.value = [];
  listTotal.value = 0;
  loadMoreVideos();
};

/** 首次/登录后：清空并从第一页拉取 */
const loadVideoListInitial = async () => {
  const token = localStorage.getItem('token');
  if (!token) return;

  videoFiles.value = [];
  listTotal.value = 0;
  await loadMoreVideos();
};

/** 滚动触底：追加下一页 */
const loadMoreVideos = async () => {
  const token = localStorage.getItem('token');
  if (!token) return;
  if (listLoading.value) return;
  if (listTotal.value > 0 && videoFiles.value.length >= listTotal.value) return;

  listLoading.value = true;
  try {
    const params = {
      skip: videoFiles.value.length,
      limit: VIDEO_LIST_PAGE_SIZE,
      sort: LIST_DEFAULT_SORT,
    };
    if (listFilterStatus.value) params.status = listFilterStatus.value;
    if (listFilterFilename.value) params.filename = listFilterFilename.value;
    const response = await getVideoList(params);
    const raw = response.data || {};
    const items = Array.isArray(raw.items) ? raw.items : [];
    const total = typeof raw.total === 'number' ? raw.total : 0;
    listTotal.value = total;
    const seen = new Set(videoFiles.value.map((v) => v.id));
    for (const v of items) {
      if (!seen.has(v.id)) {
        videoFiles.value.push(v);
        seen.add(v.id);
      }
    }
  } catch (error) {
    handleListApiError(error);
  } finally {
    listLoading.value = false;
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
    listTotal.value = Math.max(0, listTotal.value - 1);
    
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
    ElMessage.error(`视频删除失败: ${getApiErrorDetail(error)}`);
  }
};

const handleReanalyze = async (videoId) => {
  if (!videoId) return;
  try {
    await reanalyzeVideo(videoId);
    ElMessage.success('已重新提交分析');
    pollVideoStatus(videoId);
  } catch (error) {
    ElMessage.error(getApiErrorDetail(error, '重新分析失败'));
  }
};

const handleMouseEnterCanvas = () => {
  isMouseInCanvas.value = true;
  displayData.scoreTitle = '片段评分';
};

const handleMouseLeaveCanvas = throttle(() => {
  isMouseInCanvas.value = false;
  // 保持当前片段索引不变，不重置为 -1
  // 保持 canvas 显示片段数据
  
  // 但更新其他组件显示视频级数据
  if (currentVideo.value) {
    // 处理视频评分数据
    const videoScore = currentVideo.value.video_score || [];
    const safeVideoScore = Array.isArray(videoScore) ? videoScore : [];
    const score = Array.isArray(safeVideoScore[0]) ? safeVideoScore[0] : safeVideoScore;
    const overallScore = score.length > 10 ? Number(score[10]) || 0 : 0;
    const danmuScore = score.length > 11 ? Number(score[11]) || null : null;
    
    // 更新 DataBlock 组件显示
    displayData.coreIndex = overallScore;
    displayData.danmuStatus = danmuScore === null ? '无弹幕' : (danmuScore < 0 ? '欠佳' : '良好');
    displayData.scoreTitle = '整体评分';
    
    // 更新 WidgetCard 组件显示
    const radarData = processRadarData(score);
    chartData.radar1Data = radarData.video;
    chartData.radar2Data = radarData.audio;
    
    // 保持折线图显示片段级数据
    // 不更新 chartData.lineData 和 chartData.time
  }
}, 100);

const handleMouseMoveCanvas = throttle((clipIndex) => {
  if (!currentVideo.value || currentClipIndex.value === clipIndex) return;
  
  currentClipIndex.value = clipIndex;
  const clipScores = currentVideo.value.clip_scores || [];
  // 确保clipScores是数组
  const safeClipScores = Array.isArray(clipScores) ? clipScores : [];
  
  // 检查clipScores的结构，确保它是二维数组
  let processedClipScores = [];
  if (safeClipScores.length > 0) {
    if (Array.isArray(safeClipScores[0])) {
      // 已经是二维数组，直接使用
      processedClipScores = safeClipScores;
    } else {
      // 如果是一维数组，可能是模型返回的单个片段数据
      // 这是一个片段的评分数据，将其包装为二维数组
      processedClipScores = [safeClipScores];
    }
  }
  
  if (clipIndex >= 0 && clipIndex < processedClipScores.length) {
    const clipData = processedClipScores[clipIndex];
    // 确保clipData是数组
    const safeClipData = Array.isArray(clipData) ? clipData : [];
    // 处理嵌套数组的情况
    const finalClipData = Array.isArray(safeClipData[0]) ? safeClipData[0] : safeClipData;
    // 确保值是数字
    const clipScore = finalClipData.length > 10 ? Number(finalClipData[10]) || 0 : 0;
    const danmuScore = finalClipData.length > 11 ? Number(finalClipData[11]) || 0 : 0;
    
    displayData.coreIndex = clipScore;
    displayData.danmuStatus = danmuScore < 0 ? '欠佳' : '良好';
    
    const radarData = processRadarData(finalClipData, `片段 ${clipIndex + 1}`);
    chartData.radar1Data = radarData.video;
    chartData.radar2Data = radarData.audio;
  }
}, 50);

const handleCanvasClick = (clipIndex) => {
  if (clipIndex < 0) return;
  
  const videoPlayer = getVideoPlayer();
  if (!currentVideo.value || !videoPlayer) {
    ElMessage.warning('无法播放视频，请先选择一个视频');
    return;
  }
  
  const totalDuration = videoPlayer.duration || 100;
  const clipScores = currentVideo.value.clip_scores || [];
  // 确保clipScores是数组
  const safeClipScores = Array.isArray(clipScores) ? clipScores : [];
  
  // 检查clipScores的结构，确保它是二维数组
  let processedClipScores = [];
  if (safeClipScores.length > 0) {
    if (Array.isArray(safeClipScores[0])) {
      // 已经是二维数组，直接使用
      processedClipScores = safeClipScores;
    } else {
      // 如果是一维数组，可能是模型返回的单个片段数据
      // 这是一个片段的评分数据，将其包装为二维数组
      processedClipScores = [safeClipScores];
    }
  }
  
  const clipCount = processedClipScores.length || 1;
  
  if (clipIndex >= clipCount) return;
  
  const clipDuration = totalDuration / clipCount;
  const startTime = clipIndex * clipDuration;
  
  videoPlayer.currentTime = startTime;
  videoPlayer.play();
  updateTimeLabels(totalDuration);
};

// 输入对话框相关方法
const closeInputDialog = () => {
  showInputDialog.value = false;
  pendingFiles.value = null;
};

const submitVideoInfo = async (videoInfo) => {
  try {
    // 这里可以添加处理视频信息的逻辑
    showInputDialog.value = false;
    pendingFiles.value = null;
  } catch (error) {
  }
};

// 生命周期钩子
onMounted(() => {
  loadVideoListInitial();
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
</style>