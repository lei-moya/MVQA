<template>
  <el-card shadow="hover" class="help-card">
    <template #header>
      <div class="card-header">
        <i class="fa-solid fa-question-circle header-icon"></i>
        <span class="header-title">帮助中心</span>
      </div>
    </template>
    
    <div class="help-content">
      <!-- 项目介绍 -->
      <el-collapse v-model="activeNames">
        <el-collapse-item title="项目介绍" name="1">
          <div class="help-section">
            <p>视频质量分析平台是一个用于分析视频质量的综合系统，支持视频上传、分析和评分。</p>
            <p>该平台使用先进的深度学习技术，从视频和音频多个维度对视频质量进行评估，为用户提供详细的质量分析报告。</p>
          </div>
        </el-collapse-item>
        
        <!-- 功能说明 -->
        <el-collapse-item title="功能说明" name="2">
          <div class="help-section">
            <h4>主要功能</h4>
            <ul>
              <li><strong>视频上传</strong>：支持本地视频文件上传和B站视频URL上传</li>
              <li><strong>视频分析</strong>：对视频进行多维度质量分析，包括清晰度、色彩、饱和度、稳定性、亮度等</li>
              <li><strong>音频分析</strong>：对音频进行质量评估，包括音量、音质、噪音、清晰度、均衡等</li>
              <li><strong>弹幕分析</strong>：分析视频弹幕内容，评估弹幕质量</li>
              <li><strong>质量评分</strong>：根据分析结果生成综合质量评分</li>
              <li><strong>趋势图表</strong>：展示视频质量的时间趋势</li>
              <li><strong>配置管理</strong>：支持自定义视频处理、音频处理和下载设置</li>
              <li><strong>敏感词管理</strong>：管理弹幕分析中的敏感词列表</li>
            </ul>
          </div>
        </el-collapse-item>
        
        <!-- 使用指南 -->
        <el-collapse-item title="使用指南" name="3">
          <div class="help-section">
            <h4>1. 登录系统</h4>
            <p>首次使用需要使用B站账号扫码登录，登录后系统会自动创建用户账户。</p>
            
            <h4>2. 上传视频</h4>
            <p>有两种上传方式：</p>
            <ul>
              <li><strong>本地文件上传</strong>：点击"选择文件"按钮，选择本地视频文件，支持MP4、AVI、MOV等格式</li>
              <li><strong>URL上传</strong>：在输入框中输入B站视频URL，系统会自动下载视频</li>
            </ul>
            
            <h4>3. 查看分析结果</h4>
            <p>视频上传后，系统会自动开始分析。分析完成后，您可以：</p>
            <ul>
              <li>在主页面查看视频质量趋势图表</li>
              <li>查看视频和音频的雷达图分析</li>
              <li>查看核心质量指标和弹幕状态</li>
              <li>播放视频并查看实时弹幕</li>
            </ul>
            
            <h4>4. 管理设置</h4>
            <p>点击顶部导航栏的"设置"按钮，您可以：</p>
            <ul>
              <li>调整视频处理参数，如片段数、每片段帧数、目标分辨率等</li>
              <li>调整音频处理参数，如采样率</li>
              <li>管理敏感词列表</li>
              <li>调整下载设置，如视频质量、音频质量等</li>
            </ul>
          </div>
        </el-collapse-item>
        
        <!-- 常见问题 -->
        <el-collapse-item title="常见问题" name="4">
          <div class="help-section">
            <h4>Q: 视频上传后为什么没有分析结果？</h4>
            <p>A: 视频分析需要一定时间，特别是长视频。请耐心等待，分析完成后系统会自动更新结果。</p>
            
            <h4>Q: 支持哪些视频格式？</h4>
            <p>A: 支持MP4、AVI、MOV、WMV、FLV、MKV等常见视频格式。</p>
            
            <h4>Q: 视频大小有限制吗？</h4>
            <p>A: 建议上传不超过200MB的视频，过大的视频可能会导致分析时间过长或失败。</p>
            
            <h4>Q: 为什么B站URL上传失败？</h4>
            <p>A: 可能的原因包括：URL格式不正确、视频已被删除、网络连接问题等。请检查URL是否正确，或尝试使用本地文件上传。</p>
            
            <h4>Q: 分析结果的评分标准是什么？</h4>
            <p>A: 评分基于多个维度的综合评估，包括视频清晰度、色彩、饱和度、稳定性、亮度以及音频质量等。满分100分，分数越高表示质量越好。</p>
          </div>
        </el-collapse-item>
        
        <!-- 技术支持 -->
        <el-collapse-item title="技术支持" name="5">
          <div class="help-section">
            <p>如果您在使用过程中遇到问题，请联系技术支持团队：</p>
            <ul>
              <li>邮箱：support@video-quality-analysis.com</li>
              <li>电话：400-123-4567</li>
              <li>工作时间：周一至周五 9:00-18:00</li>
            </ul>
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>
  </el-card>
</template>

<script setup>
import { ref } from 'vue';

const activeNames = ref(['1']);
</script>

<style scoped>
.help-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.help-card .el-card__body {
  flex: 1;
  overflow: auto;
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
}

.header-icon {
  color: #3b82f6;
  font-size: 18px;
}

.help-content {
  height: 100%;
  overflow: auto;
}

.help-section {
  padding: 10px 0;
}

.help-section h4 {
  margin: 15px 0 10px 0;
  color: #3b82f6;
  font-size: 16px;
}

.help-section p {
  margin: 10px 0;
  line-height: 1.6;
  color: #333;
}

.help-section ul {
  margin: 10px 0 10px 20px;
}

.help-section li {
  margin: 5px 0;
  line-height: 1.5;
  color: #333;
}

/* 自定义滚动条 */
.help-card .el-card__body::-webkit-scrollbar {
  width: 6px;
}

.help-card .el-card__body::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.help-card .el-card__body::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 3px;
  transition: background 0.3s ease;
}

.help-card .el-card__body::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}

/* 折叠面板样式 */
:deep(.el-collapse-item__header) {
  font-size: 16px;
  font-weight: 500;
  color: #1f2937;
  padding: 12px 15px;
}

:deep(.el-collapse-item__content) {
  padding: 15px;
}

:deep(.el-collapse-item__header.is-active) {
  color: #3b82f6;
}

:deep(.el-collapse-item__header:hover) {
  background-color: #f9fafb;
}
</style>