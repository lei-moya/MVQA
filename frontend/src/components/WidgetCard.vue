<template>
  <el-card shadow="hover" class="widget-card">
    <template #header>
      <div class="card-header" :style="headerStyle">
        <span><i :class="icon"></i> {{ title }}</span>
      </div>
    </template>
    <div class="chart-wrapper">
      <div ref="chartRef" class="chart-container"></div>
      <div class="data-display" v-if="showLegend">
        <div class="data-item" v-for="(item, index) in legendData" :key="index">
          <span class="data-label" :style="{ color: item.color }">{{ item.name }}</span>
          <span class="data-value">{{ item.value }}</span>
        </div>
      </div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  title: String,
  icon: String,
  data: Array,
  type: String,
  headerStyle: Object,
  showLegend: {
    type: Boolean,
    default: true
  },
  legendData: {
    type: Array,
    default: () => []
  }
});

const chartRef = ref(null);
let chart = null;

const initRadar = () => {
  if (!chartRef.value) return;
  
  chart = echarts.init(chartRef.value);
  
  // 定义不同端点的颜色
  const videoColors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];
  const audioColors = ['#8b5cf6', '#ec4899', '#8b5cf6', '#ec4899', '#8b5cf6'];
  const colors = props.type === 'video' ? videoColors : audioColors;
  
  // 生成带颜色的指示器
  const indicators = props.type === 'video'
    ? ['清晰度', '色彩', '饱和度', '稳定性', '亮度']
    : ['音量', '音质', '噪音', '清晰度', '均衡'];
  
  const indicatorConfig = indicators.map((name, index) => ({
    name: name,
    max: 10,
    axisName: {
      color: colors[index]
    }
  }));
  
  const option = {
    animationDurationUpdate: 400,
    animationEasingUpdate: 'cubicOut',
    radar: {
      indicator: indicatorConfig,
      radius: '60%',
      splitArea: {show: false},
      axisName: {fontSize: 10}
    },
    series: [{
      type: 'radar',
      data: [{
        value: props.data,
        name: props.type === 'video' ? 'Model A' : 'Model B',
        areaStyle: {color: props.type === 'video' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(139, 92, 246, 0.3)'},
        lineStyle: {color: props.type === 'video' ? '#10b981' : '#8b5cf6'},
        itemStyle: {
          color: function(params) {
            return colors[params.dataIndex % colors.length];
          }
        }
      }]
    }]
  };
  chart.setOption(option);
};

const handleResize = () => {
  chart && chart.resize();
};

onMounted(() => {
  initRadar();
  window.addEventListener('resize', handleResize);
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize);
  chart && chart.dispose();
});

watch(() => props.data, () => {
  initRadar();
});
</script>

<style scoped>
.widget-card {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  transition: all 0.3s ease;
}

.widget-card:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  color: #1f2937;
  font-size: 14px;
}

.card-header i {
  margin-right: 8px;
  color: #3b82f6;
}

.chart-wrapper {
  display: flex;
  align-items: center;
  height: 180px;
  gap: 10px;
  padding: 10px;
}

.chart-container {
  flex: 1;
  height: 100%;
  min-height: 160px;
}

.data-display {
  width: 90px;
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  justify-content: center;
  background: #f9fafb;
  border-radius: 6px;
  box-shadow: inset 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.data-item {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  font-size: 12px;
  justify-content: space-between;
  padding: 4px 0;
}

.data-item:last-child {
  margin-bottom: 0;
}

.data-label {
  flex: 1;
  font-weight: 500;
}

.data-value {
  font-weight: bold;
  color: #3b82f6;
  font-size: 13px;
}

/* 自定义滚动条 */
.data-display::-webkit-scrollbar {
  width: 4px;
}

.data-display::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 2px;
}

.data-display::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 2px;
}

.data-display::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}
</style>
