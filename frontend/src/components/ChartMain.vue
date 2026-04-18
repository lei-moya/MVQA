<template>
  <el-card class="chart-main" shadow="hover">
    <template #header>
      <div class="card-header">
        <span><i class="fa-solid fa-wave-square"></i> 视频质量趋势 </span>
      </div>
    </template>
    <div class="video-wrapper">
      <div class="video-container">
        <video
          ref="videoPlayerRef"
          controls
          class="video-player"
          @timeupdate="handleTimeUpdate"
        ></video>
        <div class="danmu-container" v-if="showDanmu && activeDanmus.length > 0">
          <div 
            v-for="danmu in activeDanmus" 
            :key="danmu.id"
            class="danmu-item"
            :class="{ hidden: !danmu.visible }"
            :style="{
              left: danmu.left + 'px',
              top: danmu.top + 'px',
              animationDuration: danmu.duration + 's',
              color: danmu.color
            }"
          >
            {{ danmu.text }}
          </div>
        </div>
        <div v-if="!videoSrc" class="video-placeholder">
          <div class="placeholder-content">
            <i class="fa-solid fa-film"></i>
            <p>暂无视频源</p>
          </div>
        </div>
      </div>
      <div class="danmu-control">
        <input 
          type="checkbox" 
          id="danmu-toggle" 
          v-model="danmuEnabled"
          @change="toggleDanmu"
        >
        <label for="danmu-toggle">显示弹幕</label>
      </div>
    </div>
    <div class="chart-wrapper">
      <div ref="lineChartRef" class="chart-container"></div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  videoSrc: String,
  danmuSrc: String,
  lineData: Array,
  time: Array
});

const emit = defineEmits(['mouse-enter', 'mouse-leave', 'mouse-move', 'click']);

const videoPlayerRef = ref(null);
const lineChartRef = ref(null);
let lineChart = null;

// 弹幕相关数据
const danmuData = ref([]);
const activeDanmus = ref([]);
const showDanmu = ref(false);
const danmuEnabled = ref(true);

// 弹幕轨道管理
const danmuTracks = ref([]);
const trackHeight = 30;
const maxTracks = 10;

// 事件监听器引用
let eventListeners = {
  mouseenter: null,
  mouseleave: null,
  mousemove: null,
  click: null
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

// 初始化图表
let initRetryCount = 0;
const maxRetries = 10;

const initCharts = () => {
  if (!lineChartRef.value) return;

  const chartDom = lineChartRef.value;

  const rect = chartDom.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    initRetryCount++;
    if (initRetryCount < maxRetries) {
      setTimeout(initCharts, 200);
    }
    return;
  }

  initRetryCount = 0;

  Object.entries(eventListeners).forEach(([event, listener]) => {
    if (listener) {
      chartDom.removeEventListener(event, listener);
      eventListeners[event] = null;
    }
  });

  if (lineChart) {
    lineChart.dispose();
    lineChart = null;
  }

  try {
    lineChart = echarts.init(chartDom);
    updateChartData();
  } catch (error) {
    console.error('初始化图表失败:', error);
    if (initRetryCount < maxRetries) {
      initRetryCount++;
      setTimeout(initCharts, 200);
    }
    return;
  }

  eventListeners.mouseenter = () => emit('mouse-enter');
  eventListeners.mouseleave = () => emit('mouse-leave');
  eventListeners.mousemove = throttle((event) => {
    if (!lineChart) return;
    const r = chartDom.getBoundingClientRect();
    const x = event.clientX - r.left;
    const y = event.clientY - r.top;

    const pointInGrid = lineChart.convertFromPixel('grid', [x, y]);
    if (pointInGrid && pointInGrid[0] !== undefined) {
      const dataIndex = Math.round(pointInGrid[0]);
      if (dataIndex >= 0 && dataIndex < props.time.length) {
        emit('mouse-move', dataIndex);
      }
    }
  }, 50);
  eventListeners.click = (event) => {
    if (!lineChart) return;
    const r = chartDom.getBoundingClientRect();
    const x = event.clientX - r.left;
    const y = event.clientY - r.top;

    const pointInGrid = lineChart.convertFromPixel('grid', [x, y]);
    if (pointInGrid && pointInGrid[0] !== undefined) {
      const dataIndex = Math.round(pointInGrid[0]);
      if (dataIndex >= 0 && dataIndex < props.time.length) {
        emit('click', dataIndex);
      }
    }
  };

  Object.entries(eventListeners).forEach(([event, listener]) => {
    if (listener) {
      chartDom.addEventListener(event, listener);
    }
  });
};

// 更新图表数据
const updateChartData = () => {
  if (!lineChart) return;
  
  const lineOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      borderColor: '#3b82f6',
      borderWidth: 1,
      textStyle: {color: '#fff'},
      padding: 10,
      formatter: function(params) {
        const value = Number(params[0].value) || 0;
        return `时间: ${params[0].name}<br/>评分: ${value.toFixed(2)}`;
      }
    },
    animationDurationUpdate: 300,
    animationEasingUpdate: 'cubicOut',
    grid: {left: '3%', right: '4%', bottom: '3%', top: '10%', containLabel: true},
    xAxis: {
      type: 'category',
      data: props.time,
      boundaryGap: false,
      axisLine: {lineStyle: {color: '#e5e7eb'}},
      axisLabel: {color: '#6b7280', fontSize: 12},
      axisTick: {show: false}
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      axisLine: {show: false},
      axisLabel: {color: '#6b7280', fontSize: 12},
      axisTick: {show: false},
      splitLine: {lineStyle: {type: 'dashed', color: '#f3f4f6'}}
    },
    series: [{
      name: '评分',
      type: 'line',
      smooth: true,
      data: props.lineData,
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          {offset: 0, color: 'rgba(59, 130, 246, 0.3)'},
          {offset: 1, color: 'rgba(59, 130, 246, 0.05)'}
        ])
      },
      itemStyle: {color: '#3b82f6'},
      lineStyle: {width: 3, shadowColor: 'rgba(59, 130, 246, 0.3)', shadowBlur: 10},
      symbol: 'circle',
      symbolSize: 6,
      emphasis: {
        focus: 'series',
        itemStyle: {symbolSize: 8, shadowColor: 'rgba(59, 130, 246, 0.5)', shadowBlur: 15}
      }
    }]
  };
  lineChart.setOption(lineOption);
};

const handleResize = () => {
  lineChart && lineChart.resize();
};

const updateVideoSrc = () => {
  if (videoPlayerRef.value && props.videoSrc) {
    videoPlayerRef.value.src = props.videoSrc;
  }
};

const loadDanmu = async () => {
  if (!props.danmuSrc) {
    danmuData.value = [];
    showDanmu.value = false;
    return;
  }
  
  try {
    const response = await fetch(props.danmuSrc);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const text = await response.text();
    const danmus = parseDanmuFile(text);
    danmuData.value = danmus;
    showDanmu.value = danmuEnabled.value && danmus.length > 0;
  } catch (error) {
    console.error('加载弹幕失败:', error);
    showDanmu.value = false;
  }
};

const toggleDanmu = () => {
  showDanmu.value = danmuEnabled.value && danmuData.value.length > 0;
  if (!danmuEnabled.value) {
    activeDanmus.value = [];
  }
};

const parseDanmuFile = (text) => {
  const danmus = [];
  
  if (text.toLowerCase().includes('[script info]')) {
    const lines = text.split('\n');
    let inEvents = false;
    
    for (const line of lines) {
      if (line.trim() === '[Events]') {
        inEvents = true;
        continue;
      }
      
      if (line.trim().startsWith('Dialogue:')) {
        const parts = [];
        let currentPart = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
          const char = line[i];
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            parts.push(currentPart);
            currentPart = '';
          } else {
            currentPart += char;
          }
        }
        parts.push(currentPart);
        
        if (parts.length >= 10) {
          const startTime = parseAssTime(parts[1]);
          const textWithStyle = parts.slice(9).join(',');
          const text = textWithStyle.replace(/\{[^}]*\}/g, '').trim();
          
          if (text) {
            danmus.push({
              time: startTime,
              text: text,
              color: getRandomColor()
            });
          }
        }
      }
    }
  } else if (text.includes('<d p=')) {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(text, 'text/xml');
    const danmuNodes = xmlDoc.getElementsByTagName('d');
    
    for (let i = 0; i < danmuNodes.length; i++) {
      const node = danmuNodes[i];
      const p = node.getAttribute('p');
      const parts = p.split(',');
      if (parts.length >= 1) {
        const startTime = parseFloat(parts[0]);
        const text = node.textContent.trim();
        
        if (text) {
          danmus.push({
            time: startTime,
            text: text,
            color: getRandomColor()
          });
        }
      }
    }
  }
  
  return danmus;
};

const parseAssTime = (timeStr) => {
  const parts = timeStr.split(':');
  if (parts.length === 3) {
    const [hour, minute, second] = parts.map(p => parseFloat(p));
    return hour * 3600 + minute * 60 + second;
  }
  return 0;
};

const getRandomColor = () => {
  const colors = ['#ffffff', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
  return colors[Math.floor(Math.random() * colors.length)];
};

const initDanmuTracks = () => {
  danmuTracks.value = Array(maxTracks).fill(false);
};

const findAvailableTrack = () => {
  for (let i = 0; i < danmuTracks.value.length; i++) {
    if (!danmuTracks.value[i]) {
      danmuTracks.value[i] = true;
      return i;
    }
  }
  return Math.floor(Math.random() * maxTracks);
};

const releaseTrack = (trackIndex) => {
  if (trackIndex >= 0 && trackIndex < danmuTracks.value.length) {
    danmuTracks.value[trackIndex] = false;
  }
};

// 节流处理时间更新事件
const handleTimeUpdate = throttle(() => {
  if (!videoPlayerRef.value || !showDanmu.value || danmuData.value.length === 0 || videoPlayerRef.value.paused) return;
  
  const currentTime = videoPlayerRef.value.currentTime;
  const videoWidth = videoPlayerRef.value.offsetWidth || 800;
  
  // 过滤出当前时间应该显示的弹幕
  const currentDanmus = danmuData.value.filter(danmu => {
    return Math.abs(danmu.time - currentTime) < 0.5;
  });
  
  currentDanmus.forEach(danmu => {
    // 检查是否已存在相同内容和时间的弹幕
    const existing = activeDanmus.value.find(d => 
      d.text === danmu.text && Math.abs(d.time - danmu.time) < 0.1
    );
    if (!existing) {
      const trackIndex = findAvailableTrack();
      const top = trackIndex * trackHeight + 10;
      
      const danmuWithId = {
        ...danmu,
        left: videoWidth,
        top: top,
        duration: 15,
        trackIndex: trackIndex,
        id: Date.now() + Math.random(),
        visible: true
      };
      
      activeDanmus.value.push(danmuWithId);
      
      // 弹幕14秒后开始淡出
      setTimeout(() => {
        const d = activeDanmus.value.find(d => d.id === danmuWithId.id);
        if (d) {
          d.visible = false;
        }
      }, 14000);
      
      // 弹幕15秒后完全移除
      setTimeout(() => {
        const index = activeDanmus.value.findIndex(d => d.id === danmuWithId.id);
        if (index !== -1) {
          activeDanmus.value.splice(index, 1);
          releaseTrack(danmuWithId.trackIndex);
        }
      }, 15000);
    }
  });
}, 100);

onMounted(() => {
  initDanmuTracks();
  updateVideoSrc();
  loadDanmu();
  window.addEventListener('resize', handleResize);
  
  // 延迟初始化图表，确保DOM元素已经渲染完成
  setTimeout(() => {
    initCharts();
  }, 500);
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize);
  if (lineChart) {
    lineChart.dispose();
    lineChart = null;
  }
  // 移除事件监听器
  if (lineChartRef.value) {
    Object.entries(eventListeners).forEach(([event, listener]) => {
      if (listener) {
        lineChartRef.value.removeEventListener(event, listener);
      }
    });
  }
});

watch(() => props.videoSrc, () => {
  updateVideoSrc();
  // 视频源变化时清空弹幕
  activeDanmus.value = [];
  danmuData.value = [];
  showDanmu.value = false;
  initDanmuTracks();
});

watch(() => props.danmuSrc, () => {
  loadDanmu();
});

watch([() => props.lineData, () => props.time], () => {
  nextTick(() => {
    updateChartData();
  });
}, { deep: true });

// 暴露视频播放器引用
defineExpose({
  videoPlayer: videoPlayerRef
});
</script>

<style scoped>
.chart-main {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chart-main .el-card__body {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 20px;
  margin: 0;
}

.card-header {
  display: flex;
  align-items: center;
  font-weight: bold;
  color: #1f2937;
  font-size: 16px;
}

.card-header i {
  margin-right: 8px;
  color: #3b82f6;
}

.video-wrapper {
  margin-bottom: 2px;
  flex: 0 0 auto;
}

.video-container {
  position: relative;
  width: 100%;
  max-height: 400px;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.1), 0 8px 24px rgba(15, 23, 42, 0.12);
}

.video-player {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
  display: block;
  z-index: 1;
  position: relative;
}

.danmu-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: hidden;
  z-index: 5;
}

.danmu-item {
  position: absolute;
  white-space: nowrap;
  font-size: 20px;
  font-weight: bold;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
  animation: danmuMove linear forwards;
  padding: 2px 8px;
  border-radius: 2px;
  background-color: rgba(0, 0, 0, 0.3);
  transition: opacity 1s ease-out;
  opacity: 1;
}

.danmu-item.hidden {
  opacity: 0;
}

@keyframes danmuMove {
  from { transform: translateX(0); }
  to { transform: translateX(calc(-100% - 800px)); }
}

.video-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f9fafb 0%, #000000 100%);
  z-index: 0;
}

.placeholder-content {
  text-align: center;
  color: #6b7280;
}

.placeholder-content i {
  font-size: 3rem;
  margin-bottom: 16px;
  color: #e6e9ed;
}

.placeholder-content p {
  font-size: 1rem;
  margin: 0;
  color: #e6e9ed;
}

.chart-wrapper {
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
  overflow: hidden;
  flex: 1;
  min-height: 250px;
}

.chart-container {
  width: 100%;
  height: 305px;
  min-height: 305px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .video-container {
    aspect-ratio: 4 / 3;
  }
  
  .danmu-item {
    font-size: 16px;
  }
}

.danmu-control {
  margin-top: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 14px;
  color: #333;
}

.danmu-control input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.danmu-control label {
  cursor: pointer;
  user-select: none;
}
</style>
