<template>
  <el-card shadow="hover" class="settings-card">
    <template #header>
      <div class="card-header">
        <i class="fa-solid fa-gear"></i>
        <span>系统设置</span>
      </div>
    </template>
    
    <div class="settings-inner">
      <el-tabs v-model="activeTab" class="settings-tabs">
        <el-tab-pane label="视频处理" name="video">
          <div class="tab-pane-body">
          <el-form :model="config.video_processing" label-width="120px">
            <el-form-item label="视频片段数">
              <el-input-number v-model="config.video_processing.num_clips" :min="1" :max="250" :step="1" />
              <span class="form-hint">视频将被分割成多少个片段进行分析</span>
            </el-form-item>
            
            <el-form-item label="每片段帧数">
              <el-input-number v-model="config.video_processing.frames_per_clip" :min="1" :max="20" :step="1" />
              <span class="form-hint">每个片段提取的帧数</span>
            </el-form-item>
            
            <el-form-item label="目标分辨率">
              <div class="resolution-input">
                <el-input-number v-model="config.video_processing.target_size[0]" :min="128" :max="1024" :step="32" />
                <span class="resolution-separator">×</span>
                <el-input-number v-model="config.video_processing.target_size[1]" :min="128" :max="1024" :step="32" />
              </div>
              <span class="form-hint">视频帧的目标分辨率</span>
            </el-form-item>
          </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="音频处理" name="audio">
          <div class="tab-pane-body">
          <el-form :model="config.audio_processing" label-width="120px">
            <el-form-item label="采样率">
              <el-select v-model="config.audio_processing.sample_rate" style="width: 150px;">
                <el-option label="16000 Hz" :value="16000" />
                <el-option label="22050 Hz" :value="22050" />
                <el-option label="44100 Hz" :value="44100" />
              </el-select>
              <span class="form-hint">音频采样率</span>
            </el-form-item>
          </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="敏感词" name="sensitive">
          <div class="tab-pane-body tab-pane-body--sensitive">
          <el-form label-width="120px">
            <p v-if="!isAdmin" class="form-hint block-hint">
              添加敏感词需提交审核，由管理员同意后生效；删除与批量保存词库仅管理员可用。
            </p>
            <el-form-item v-if="isAdmin" label="待审核申请">
              <div v-if="pendingRequests.length === 0" class="form-hint">暂无待审核申请</div>
              <div v-else class="pending-requests-scroll">
                <el-table
                  :data="pendingRequests"
                  size="small"
                  max-height="260"
                  stripe
                  class="pending-requests-table"
                >
                  <el-table-column prop="requester_username" label="申请人" min-width="160" />
                  <el-table-column prop="word" label="词" min-width="220" />
                  <el-table-column label="操作" width="168" fixed="right">
                    <template #default="scope">
                      <el-button type="success" size="small" link @click="handleApproveRequest(scope.row)">同意</el-button>
                      <el-button type="danger" size="small" link @click="handleRejectRequest(scope.row)">拒绝</el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </div>
            </el-form-item>
            <el-form-item v-else label="我的申请">
              <el-table v-if="myRequests.length" :data="myRequests" size="small" max-height="200" stripe>
                <el-table-column prop="word" label="词" min-width="90" />
                <el-table-column label="状态" width="96">
                  <template #default="scope">
                    {{ requestStatusLabel(scope.row.status) }}
                  </template>
                </el-table-column>
              </el-table>
              <span v-else class="form-hint">暂无申请记录</span>
            </el-form-item>
            <el-form-item label="敏感词检索">
              <el-input
                v-model="searchKeyword"
                placeholder="搜索敏感词"
                style="width: 200px;"
                @keyup.enter="handleSearchSensitiveWord"
              >
              </el-input>
              <el-button type="primary" @click="clearSearch" style="margin-left: 10px;">清除搜索</el-button>
            </el-form-item>
            <el-form-item label="敏感词列表">
              <div v-if="filteredSensitiveWords.length === 0" class="no-results">
                <el-empty description="未找到匹配的敏感词" />
              </div>
              <div v-else>
                <el-tag
                  v-for="(word, index) in filteredSensitiveWords"
                  :key="index"
                  :closable="isAdmin"
                  @close="handleRemoveSensitiveWord(config.sensitive_words.indexOf(word))"
                  class="sensitive-tag"
                >
                  {{ word }}
                </el-tag>
              </div>
              <el-input
                v-model="newSensitiveWord"
                :placeholder="isAdmin ? '添加敏感词（直接入库）' : '填写后提交管理员审核'"
                style="width: 260px; margin-left: 10px; margin-top: 10px;"
                @keyup.enter="handleAddSensitiveWord"
              >
                <template #append>
                  <el-button @click="handleAddSensitiveWord">{{ isAdmin ? '添加' : '提交审核' }}</el-button>
                </template>
              </el-input>
            </el-form-item>
          </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane v-if="isAdmin" label="管理员 UID" name="admin">
          <div class="tab-pane-body">
            <el-form label-width="120px">
              <p class="form-hint block-hint">
                在此维护<strong>库内</strong>管理员 B 站 UID（数字 mid，多个用英文逗号分隔）。将与服务端环境变量
                <code>ADMIN_BILIBILI_MIDS</code> 合并；用户<strong>下次扫码登录</strong>时按合并列表同步为管理员或普通用户。
              </p>
              <el-form-item label="环境变量 UID">
                <span class="form-hint">{{ adminMidsEnv.length ? adminMidsEnv.join(', ') : '（未配置）' }}</span>
              </el-form-item>
              <el-form-item label="合并后 UID">
                <span class="form-hint">{{ adminMidsEffective.length ? adminMidsEffective.join(', ') : '—' }}</span>
              </el-form-item>
              <el-form-item label="库内配置">
                <el-input
                  v-model="adminMidsStoredInput"
                  type="textarea"
                  :rows="3"
                  placeholder="例：123456789, 987654321"
                  class="admin-mids-textarea"
                />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" :loading="adminMidsSaving" @click="saveAdminMids">
                  保存管理员 UID
                </el-button>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="下载设置" name="download">
          <div class="tab-pane-body">
          <el-form :model="config.download_settings" label-width="120px">
            <el-form-item label="视频质量"> 
              <el-select v-model="config.download_settings.video_quality" style="width: 150px;"> 
                  <el-option label="自动" value="auto" /> 
                  <el-option label="超高清 8K" value="127" /> 
                  <el-option label="杜比视界" value="126" /> 
                  <el-option label="真彩 HDR" value="125" /> 
                  <el-option label="超清 4K" value="120" /> 
                  <el-option label="高清 1080P60" value="116" /> 
                  <el-option label="高清 1080P+" value="112" /> 
                  <el-option label="高清 1080P" value="80" /> 
                  <el-option label="高清 720P60" value="74" /> 
                  <el-option label="高清 720P" value="64" /> 
                  <el-option label="清晰 480P" value="32" /> 
                  <el-option label="流畅 360P" value="16" /> 
                  <el-option label="极速 240P" value="6" /> 
              </el-select> 
              <span class="form-hint">选择下载视频的质量</span> 
            </el-form-item>   

            <el-form-item label="默认视频质量"> 
              <el-select v-model="config.download_settings.default_video_quality" style="width: 150px;"> 
                  <el-option label="自动" value="auto" /> 
                  <el-option label="超高清 8K" value="127" /> 
                  <el-option label="杜比视界" value="126" /> 
                  <el-option label="真彩 HDR" value="125" /> 
                  <el-option label="超清 4K" value="120" /> 
                  <el-option label="高清 1080P60" value="116" /> 
                  <el-option label="高清 1080P+" value="112" /> 
                  <el-option label="高清 1080P" value="80" /> 
                  <el-option label="高清 720P60" value="74" /> 
                  <el-option label="高清 720P" value="64" /> 
                  <el-option label="清晰 480P" value="32" /> 
                  <el-option label="流畅 360P" value="16" /> 
                  <el-option label="极速 240P" value="6" /> 
              </el-select> 
              <span class="form-hint">当未找到指定质量时，返回不超过此质量的最高质量视频</span> 
            </el-form-item>   

            <el-form-item label="音频质量"> 
                <el-select v-model="config.download_settings.audio_quality" style="width: 150px;"> 
                    <el-option label="高质量 (FLAC)" value="high" /> 
                    <el-option label="中等质量" value="medium" /> 
                    <el-option label="低质量" value="low" /> 
                </el-select> 
                <span class="form-hint">选择下载音频的质量</span> 
            </el-form-item>

            <el-form-item label="默认音频质量"> 
                <el-select v-model="config.download_settings.default_audio_quality" style="width: 150px;"> 
                    <el-option label="高质量 (FLAC)" value="high" /> 
                    <el-option label="中等质量" value="medium" /> 
                    <el-option label="低质量" value="low" /> 
                </el-select> 
                <span class="form-hint">当未找到指定质量时，返回不超过此质量的最高质量音频</span> 
            </el-form-item>
            
            <el-form-item label="下载弹幕">
              <el-switch v-model="config.download_settings.download_danmaku" />
              <span class="form-hint">是否下载视频弹幕</span>
            </el-form-item>
            

          </el-form>
          </div>
        </el-tab-pane>
      </el-tabs>
      
      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button type="primary" @click="saveConfig" :loading="loading">
          <i class="fa-solid fa-save"></i>
          保存配置
        </el-button>
        <el-button @click="resetConfig">
          <i class="fa-solid fa-undo"></i>
          重置
        </el-button>
      </div>
    </div>
  </el-card>
  
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue';
import { ElMessage } from 'element-plus';
import {
  getConfig,
  updateConfig,
  getSensitiveWords,
  getCurrentUser,
  addSensitiveWord,
  deleteSensitiveWord,
  updateSensitiveWords,
  requestSensitiveWord,
  getPendingSensitiveWordRequests,
  getMySensitiveWordRequests,
  approveSensitiveWordRequest,
  rejectSensitiveWordRequest,
  getApiErrorDetail,
  getAdminBilibiliMids,
  updateAdminBilibiliMids,
} from '../api/index.js';

const activeTab = ref('video');
const loading = ref(false);
const newSensitiveWord = ref('');
const searchKeyword = ref('');
const isAdmin = ref(false);
const pendingRequests = ref([]);
const myRequests = ref([]);
const adminMidsEnv = ref([]);
const adminMidsEffective = ref([]);
const adminMidsStoredInput = ref('');
const adminMidsSaving = ref(false);

const requestStatusLabel = (s) =>
  ({ pending: '待审核', approved: '已通过', rejected: '已拒绝' }[s] || s);

// 与 backend/config.py 中 DEFAULT_CONFIG 保持字段与默认值一致（含 resetConfig）
const config = reactive({
  video_processing: {
    num_clips: 125,
    frames_per_clip: 5,
    target_size: [224, 224]
  },
  audio_processing: {
    sample_rate: 16000
  },
  sensitive_words: [],
  download_settings: {
    video_quality: "auto",
    default_video_quality: "80",
    audio_quality: "high",
    default_audio_quality: "high",
    download_danmaku: true
  }
});

// 过滤后的敏感词列表
const filteredSensitiveWords = computed(() => {
  if (!searchKeyword.value) {
    return config.sensitive_words;
  }
  return config.sensitive_words.filter(word => 
    word.toLowerCase().includes(searchKeyword.value.toLowerCase())
  );
});

// 原始配置（用于重置）
let originalConfig = { ...config };

const syncUserRole = async () => {
  try {
    const { data } = await getCurrentUser();
    isAdmin.value = data.role === 'admin';
    try {
      const raw = localStorage.getItem('user');
      const u = raw ? JSON.parse(raw) : {};
      u.role = data.role;
      localStorage.setItem('user', JSON.stringify(u));
    } catch (_) {
      /* ignore */
    }
  } catch (_) {
    isAdmin.value = false;
  }
};

const loadPendingRequests = async () => {
  if (!isAdmin.value) return;
  try {
    const { data } = await getPendingSensitiveWordRequests();
    pendingRequests.value = Array.isArray(data) ? data : [];
  } catch (_) {
    pendingRequests.value = [];
  }
};

const loadMyRequests = async () => {
  if (isAdmin.value) return;
  try {
    const { data } = await getMySensitiveWordRequests();
    myRequests.value = Array.isArray(data) ? data : [];
  } catch (_) {
    myRequests.value = [];
  }
};

const loadAdminMids = async () => {
  if (!isAdmin.value) return;
  try {
    const { data } = await getAdminBilibiliMids();
    adminMidsEnv.value = Array.isArray(data.env_mids) ? data.env_mids : [];
    adminMidsEffective.value = Array.isArray(data.effective_mids) ? data.effective_mids : [];
    adminMidsStoredInput.value = typeof data.stored_mids === 'string' ? data.stored_mids : '';
  } catch (_) {
    adminMidsEnv.value = [];
    adminMidsEffective.value = [];
  }
};

const saveAdminMids = async () => {
  adminMidsSaving.value = true;
  try {
    const { data } = await updateAdminBilibiliMids(adminMidsStoredInput.value);
    adminMidsEnv.value = Array.isArray(data.env_mids) ? data.env_mids : [];
    adminMidsEffective.value = Array.isArray(data.effective_mids) ? data.effective_mids : [];
    adminMidsStoredInput.value = typeof data.stored_mids === 'string' ? data.stored_mids : '';
    ElMessage.success('已保存。相关用户需在下次扫码登录后更新管理员身份。');
  } catch (error) {
    ElMessage.error(getApiErrorDetail(error, '保存失败'));
  } finally {
    adminMidsSaving.value = false;
  }
};

// 加载配置
const loadConfig = async () => {
  try {
    loading.value = true;
    // 获取基本配置
    const configResponse = await getConfig();
    const configData = configResponse.data || configResponse;
    if (configData) {
      // 更新视频处理配置
      if (configData.video_processing) {
        Object.assign(config.video_processing, configData.video_processing);
      }
      // 更新音频处理配置
      if (configData.audio_processing) {
        Object.assign(config.audio_processing, configData.audio_processing);
        const sr = config.audio_processing.sample_rate;
        config.audio_processing.sample_rate =
          typeof sr === 'string' ? parseInt(sr, 10) || 16000 : Number(sr) || 16000;
      }
      // 更新下载设置
      if (configData.download_settings) {
        Object.assign(config.download_settings, configData.download_settings);
      }
    }
    
    // 获取敏感词列表
    const wordsResponse = await getSensitiveWords();
    if (wordsResponse.data && wordsResponse.data.sensitive_words) {
      config.sensitive_words = [...wordsResponse.data.sensitive_words];
    }

    await loadPendingRequests();
    await loadMyRequests();
    await loadAdminMids();

    originalConfig = JSON.parse(JSON.stringify(config));
  } catch (error) {
    console.error('加载配置失败:', error);
    ElMessage.error(`加载配置失败: ${getApiErrorDetail(error)}`);
  } finally {
    loading.value = false;
  }
};

// 保存配置
const saveConfig = async () => {
  try {
    loading.value = true;
    
    // 创建不包含敏感词的配置对象
    const configWithoutWords = {
      video_processing: config.video_processing,
      audio_processing: config.audio_processing,
      download_settings: config.download_settings
    };
    
    // 保存基本配置
    await updateConfig(configWithoutWords);
    
    if (isAdmin.value) {
      await updateSensitiveWords(config.sensitive_words);
    }

    originalConfig = JSON.parse(JSON.stringify(config));
    ElMessage.success(isAdmin.value ? '配置保存成功' : '已保存处理/下载设置（敏感词仅管理员可批量保存）');
  } catch (error) {
    console.error('保存配置失败:', error);
    ElMessage.error(`保存配置失败: ${getApiErrorDetail(error)}`);
  } finally {
    loading.value = false;
  }
};

// 重置配置
const resetConfig = async () => {
  // 恢复默认配置
  config.video_processing = {
    num_clips: 125,
    frames_per_clip: 5,
    target_size: [224, 224]
  };
  
  config.audio_processing = {
    sample_rate: 16000
  };
  
  config.download_settings = {
    video_quality: "auto",
    default_video_quality: "80",
    audio_quality: "high",
    default_audio_quality: "high",
    download_danmaku: true
  };
  
  // 自动保存到数据库
  try {
    loading.value = true;
    
    // 创建不包含敏感词的配置对象
    const configWithoutWords = {
      video_processing: config.video_processing,
      audio_processing: config.audio_processing,
      download_settings: config.download_settings
    };
    
    // 保存基本配置
    await updateConfig(configWithoutWords);
    
    ElMessage.success('配置已重置为默认值并保存');
  } catch (error) {
    console.error('重置配置失败:', error);
    ElMessage.error(`配置重置失败: ${getApiErrorDetail(error)}`);
  } finally {
    loading.value = false;
  }
};

// 添加敏感词（管理员直接入库；普通用户走审核）
const handleAddSensitiveWord = async () => {
  const w = newSensitiveWord.value.trim();
  if (!w) return;
  if (config.sensitive_words.includes(w)) {
    ElMessage.warning('敏感词已在词库中');
    return;
  }
  const body = {
    word: w,
    category: '',
    is_regex: false,
    is_whitelist: false,
    action: 'block',
  };
  try {
    if (isAdmin.value) {
      await addSensitiveWord(body);
      config.sensitive_words.push(w);
      newSensitiveWord.value = '';
      ElMessage.success('敏感词已添加');
    } else {
      await requestSensitiveWord(body);
      newSensitiveWord.value = '';
      ElMessage.success('已提交审核，请等待管理员处理');
      await loadMyRequests();
    }
  } catch (error) {
    console.error('敏感词操作失败:', error);
    ElMessage.error(getApiErrorDetail(error, '操作失败'));
  }
};

const handleApproveRequest = async (row) => {
  try {
    await approveSensitiveWordRequest(row.id);
    ElMessage.success('已同意并加入词库');
    await loadPendingRequests();
    const wordsResponse = await getSensitiveWords();
    if (wordsResponse.data?.sensitive_words) {
      config.sensitive_words = [...wordsResponse.data.sensitive_words];
    }
  } catch (error) {
    ElMessage.error(getApiErrorDetail(error, '操作失败'));
  }
};

const handleRejectRequest = async (row) => {
  try {
    await rejectSensitiveWordRequest(row.id);
    ElMessage.success('已拒绝该申请');
    await loadPendingRequests();
  } catch (error) {
    ElMessage.error(getApiErrorDetail(error, '操作失败'));
  }
};

// 删除敏感词
const handleRemoveSensitiveWord = async (index) => {
  if (!isAdmin.value) {
    ElMessage.warning('仅管理员可删除敏感词');
    return;
  }
  const word = config.sensitive_words[index];
  try {
    await deleteSensitiveWord(word);
    config.sensitive_words.splice(index, 1);
    ElMessage.success('敏感词删除成功');
  } catch (error) {
    console.error('删除敏感词失败:', error);
    ElMessage.error(`删除敏感词失败: ${getApiErrorDetail(error)}`);
  }
};

// 搜索敏感词
const handleSearchSensitiveWord = () => {
  // 搜索逻辑已在 computed 中实现，无匹配时列表区已有空状态，不再弹窗
};

// 清除搜索
const clearSearch = () => {
  searchKeyword.value = '';
};

// 生命周期钩子
onMounted(async () => {
  await syncUserRole();
  await loadConfig();
});
</script>

<style scoped>
.settings-card {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.settings-card :deep(.el-card__body) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 12px 16px;
}

.settings-inner {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.settings-tabs {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.settings-tabs :deep(.el-tabs__header) {
  flex-shrink: 0;
  margin: 0 0 8px;
}

.settings-tabs :deep(.el-tabs__content) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.settings-tabs :deep(.el-tab-pane) {
  height: 100%;
  overflow: hidden;
}

.tab-pane-body {
  height: 100%;
  overflow-y: auto;
  padding-right: 4px;
  box-sizing: border-box;
}

.tab-pane-body--sensitive :deep(.el-form-item) {
  margin-bottom: 12px;
}

.admin-mids-textarea {
  width: 100%;
  max-width: 520px;
}

.block-hint code {
  font-size: 12px;
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
}

/* 待审核表：申请人、词条单行完整展示，必要时横向滚动 */
.pending-requests-scroll {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: visible;
}

.pending-requests-table {
  min-width: max(100%, 520px);
}

.pending-requests-table :deep(.el-table__header .cell),
.pending-requests-table :deep(.el-table__body .cell) {
  white-space: nowrap;
  word-break: keep-all;
  overflow: visible;
  line-height: 1.4;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: #0f172a;
}

.form-hint {
  margin-left: 10px;
  color: #606266;
  font-size: 12px;
}

.block-hint {
  display: block;
  margin: 0 0 12px 0;
  margin-left: 0;
}

.resolution-input {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 300px;
}

.resolution-separator {
  font-size: 16px;
  font-weight: bold;
  color: #606266;
}

.sensitive-tag {
  margin: 5px;
}

.action-buttons {
  flex-shrink: 0;
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--el-border-color-lighter);
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.no-results {
  margin: 20px 0;
  padding: 20px;
  background-color: #f1f5f9;
  border: 1px solid rgba(15, 23, 42, 0.06);
  border-radius: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>