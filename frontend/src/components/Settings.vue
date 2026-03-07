<template>
  <el-card shadow="hover" class="settings-card">
    <template #header>
      <div class="card-header">
        <i class="fa-solid fa-gear"></i>
        <span>系统设置</span>
      </div>
    </template>
    
    <div class="settings-content">
      <!-- 视频处理设置 -->
      <el-collapse v-model="activeNames">
        <el-collapse-item title="视频处理设置" name="1">
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
        </el-collapse-item>
        
        <!-- 音频处理设置 -->
        <el-collapse-item title="音频处理设置" name="2">
          <el-form :model="config.audio_processing" label-width="120px">
            <el-form-item label="采样率">
              <el-select v-model="config.audio_processing.sample_rate" style="width: 150px;">
                <el-option label="16000 Hz" value="16000" />
                <el-option label="22050 Hz" value="22050" />
                <el-option label="44100 Hz" value="44100" />
              </el-select>
              <span class="form-hint">音频采样率</span>
            </el-form-item>
          </el-form>
        </el-collapse-item>
        
        <!-- 敏感词设置 -->
        <el-collapse-item title="敏感词设置" name="3">
          <el-form label-width="120px">
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
                  closable
                  @close="handleRemoveSensitiveWord(config.sensitive_words.indexOf(word))"
                  class="sensitive-tag"
                >
                  {{ word }}
                </el-tag>
              </div>
              <el-input
                v-model="newSensitiveWord"
                placeholder="添加敏感词"
                style="width: 200px; margin-left: 10px; margin-top: 10px;"
                @keyup.enter="handleAddSensitiveWord"
              >
                <template #append>
                  <el-button @click="handleAddSensitiveWord">添加</el-button>
                </template>
              </el-input>
            </el-form-item>
          </el-form>
        </el-collapse-item>
        
        <!-- 下载设置 -->
        <el-collapse-item title="下载设置" name="4">
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
        </el-collapse-item>
      </el-collapse>
      
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
import { getConfig, updateConfig, getSensitiveWords, addSensitiveWord, deleteSensitiveWord, updateSensitiveWords } from '../api/index.js';

const activeNames = ref(['1']);
const loading = ref(false);
const newSensitiveWord = ref('');
const searchKeyword = ref('');

// 配置数据
const config = reactive({
  video_processing: {
    num_clips: 5,
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
    
    originalConfig = JSON.parse(JSON.stringify(config));
  } catch (error) {
    console.error('加载配置失败:', error);
    ElMessage.error('加载配置失败');
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
    
    // 保存敏感词
    await updateSensitiveWords(config.sensitive_words);
    
    originalConfig = JSON.parse(JSON.stringify(config));
    ElMessage.success('配置保存成功');
  } catch (error) {
    console.error('保存配置失败:', error);
    ElMessage.error('保存配置失败');
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
    await saveConfig();
    ElMessage.success('配置已重置为默认值并保存');
  } catch (error) {
    console.error('保存配置失败:', error);
    ElMessage.error('配置重置失败');
  } finally {
    loading.value = false;
  }
};

// 添加敏感词
const handleAddSensitiveWord = async () => {
  if (newSensitiveWord.value.trim()) {
    if (!config.sensitive_words.includes(newSensitiveWord.value.trim())) {
      try {
        await addSensitiveWord(newSensitiveWord.value.trim());
        config.sensitive_words.push(newSensitiveWord.value.trim());
        newSensitiveWord.value = '';
        ElMessage.success('敏感词添加成功');
      } catch (error) {
        console.error('添加敏感词失败:', error);
        ElMessage.error('添加敏感词失败');
      }
    } else {
      ElMessage.warning('敏感词已存在');
    }
  }
};

// 删除敏感词
const handleRemoveSensitiveWord = async (index) => {
  const word = config.sensitive_words[index];
  try {
    await deleteSensitiveWord(word);
    config.sensitive_words.splice(index, 1);
    ElMessage.success('敏感词删除成功');
  } catch (error) {
    console.error('删除敏感词失败:', error);
    ElMessage.error('删除敏感词失败');
  }
};

// 搜索敏感词
const handleSearchSensitiveWord = () => {
  // 搜索逻辑已在computed属性中实现
  if (searchKeyword.value && filteredSensitiveWords.value.length === 0) {
    ElMessage.info('未找到匹配的敏感词');
  }
};

// 清除搜索
const clearSearch = () => {
  searchKeyword.value = '';
  ElMessage.info('搜索已清除');
};

// 生命周期钩子
onMounted(() => {
  loadConfig();
});
</script>

<style scoped>

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
}



.settings-content {
  padding: 0 20px;
  max-height: calc(100vh - 200px);
  overflow-y: auto;
}

.form-hint {
  margin-left: 10px;
  color: #606266;
  font-size: 12px;
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
  margin-top: 30px;
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.no-results {
  margin: 20px 0;
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 4px;
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>