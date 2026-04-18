<template>
  <el-card shadow="hover" style="height: 100%; display: flex; flex-direction: column;">
    <template #header>
      <div class="card-header">
        <span><i class="fa-solid fa-folder-open"></i> 已上传视频</span>
        <el-badge :value="badgeCount" type="primary" />
      </div>
    </template>
    <div class="list-body">
    <div class="list-toolbar">
      <el-select
        :model-value="filterStatus"
        placeholder="状态"
        clearable
        size="small"
        class="toolbar-select"
        @change="onStatusChange"
      >
        <el-option label="全部状态" value="" />
        <el-option label="待处理" value="pending" />
        <el-option label="待分析" value="downloaded" />
        <el-option label="处理中" value="processing" />
        <el-option label="已完成" value="completed" />
        <el-option label="失败" value="failed" />
      </el-select>
      <el-input
        v-model="filenameDraft"
        size="small"
        clearable
        placeholder="文件名包含…"
        class="toolbar-input"
        @keyup.enter="applyFilenameSearch"
      />
      <el-button type="primary" size="small" class="toolbar-search-btn" @click="applyFilenameSearch">搜索</el-button>
    </div>
    <div
      class="table-container"
      @scroll.passive="onScroll"
    >
      <el-empty
        v-if="files.length === 0 && !loadingMore"
        :description="emptyDescription"
        style="height: 100%; display: flex; align-items: center; justify-content: center;"
      />
      <template v-else-if="files.length === 0 && loadingMore">
        <div class="list-loading-placeholder">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>加载列表中…</span>
        </div>
      </template>
      <el-table v-else :data="files" style="width: 100%;" :stripe="true" :header-cell-style="{ position: 'sticky', top: 0, zIndex: 0, backgroundColor: '#ffffff' }">
        <el-table-column label="" width="52" align="center">
          <template #default="scope">
            <img
              v-if="scope.row.thumbnail_path"
              :src="thumbUrl(scope.row.thumbnail_path)"
              class="row-thumb"
              alt=""
            />
            <i v-else class="fa-regular fa-image row-thumb-placeholder" aria-hidden="true"></i>
          </template>
        </el-table-column>
        <el-table-column prop="filename" label="视频名" min-width="130">
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
              <span>{{ Math.round(scope.row.score_change) }}</span>
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
        <el-table-column label="操作" width="72">
          <template #default="scope">
            <div class="action-col">
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
                v-if="scope.row.status === 'completed' || scope.row.status === 'failed'"
                type="warning"
                size="small"
                @click="$emit('reanalyze', scope.row.id)"
                style="width: 100%; margin: 0;"
              >
                重分析
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
      <div v-if="files.length > 0" class="list-footer">
        <template v-if="loadingMore">
          <el-icon class="is-loading footer-icon"><Loading /></el-icon>
          <span>加载中…</span>
        </template>
        <span v-else-if="!hasMore" class="footer-done">
          已加载全部 {{ totalCount }} 条
        </span>
        <span v-else class="footer-hint">下滑加载更多</span>
      </div>
    </div>
    </div>
  </el-card>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { ElMessageBox } from 'element-plus';
import { Loading } from '@element-plus/icons-vue';
import { uploadsPublicUrl } from '../api/index.js';

const thumbUrl = (p) => (p ? uploadsPublicUrl(p) : '');

const props = defineProps({
  files: Array,
  currentVideoId: {
    type: Number,
    default: null,
  },
  /** 服务端总数（用于角标与底部文案） */
  totalCount: {
    type: Number,
    default: 0,
  },
  loadingMore: {
    type: Boolean,
    default: false,
  },
  hasMore: {
    type: Boolean,
    default: false,
  },
  filterStatus: {
    type: String,
    default: '',
  },
  filterFilename: {
    type: String,
    default: '',
  },
});

const emit = defineEmits(['view', 'delete', 'reanalyze', 'load-more', 'filters-change']);

const filenameDraft = ref(props.filterFilename);
watch(
  () => props.filterFilename,
  (v) => {
    filenameDraft.value = v;
  }
);

const emptyDescription = computed(() =>
  props.filterStatus || props.filterFilename ? '暂无符合条件的视频' : '暂无上传视频'
);

const emitFilters = (payload) => {
  emit('filters-change', payload);
};

const onStatusChange = (v) => {
  emitFilters({
    status: v ?? '',
    filename: filenameDraft.value.trim(),
  });
};

const applyFilenameSearch = () => {
  emitFilters({
    status: props.filterStatus,
    filename: filenameDraft.value.trim(),
  });
};

const badgeCount = computed(() =>
  props.totalCount > 0 ? props.totalCount : props.files?.length ?? 0
);

const lastLoadMoreAt = ref(0);
const LOAD_MORE_COOLDOWN_MS = 450;

const onScroll = (e) => {
  const el = e.target;
  const threshold = 80;
  if (el.scrollHeight - el.scrollTop - el.clientHeight > threshold) return;
  if (!props.hasMore || props.loadingMore) return;
  const now = Date.now();
  if (now - lastLoadMoreAt.value < LOAD_MORE_COOLDOWN_MS) return;
  lastLoadMoreAt.value = now;
  emit('load-more');
};

const getStatusType = (status) => {
  switch (status) {
    case 'pending':
      return 'info';
    case 'downloaded':
      return '';
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
    case 'downloaded':
      return '待分析';
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
    // 用户取消不提示；其它异常极少见，实际删除错误由首页接口结果提示
    if (error !== 'cancel') {
      console.warn('确认删除对话框异常:', error);
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

.row-thumb {
  width: 40px;
  height: 28px;
  object-fit: cover;
  border-radius: 4px;
  display: block;
}

.row-thumb-placeholder {
  color: #d1d5db;
  font-size: 1.25rem;
}

.action-col {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: stretch;
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
  filter: brightness(1.06);
  box-shadow: 0 4px 6px rgba(59, 130, 246, 0.35);
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
  box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4);
  filter: brightness(1.05);
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.72;
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
  filter: brightness(1.03);
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

.list-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.list-toolbar {
  flex-shrink: 0;
  padding: 8px 10px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  gap: 8px;
  background: #fafafa;
  min-width: 0;
}

.toolbar-select {
  width: 118px;
  flex-shrink: 0;
}

.toolbar-input {
  flex: 1;
  min-width: 0;
}

.list-toolbar :deep(.el-input) {
  min-width: 0;
}

.toolbar-search-btn {
  flex-shrink: 0;
}

/* 表格容器 */
.table-container {
  flex: 1;
  min-height: 0;
  overflow: auto;
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
  background-color: #eef2f7 !important;
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
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
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
  filter: brightness(1.04);
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

.list-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 12px;
  font-size: 12px;
  color: #6b7280;
  border-top: 1px solid #f3f4f6;
  flex-shrink: 0;
}

.footer-icon {
  font-size: 14px;
}

.footer-done {
  color: #9ca3af;
}

.footer-hint {
  color: #9ca3af;
}

.list-loading-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 160px;
  color: #6b7280;
  font-size: 14px;
}
</style>
