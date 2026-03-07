/**
 * API请求模块
 * 封装了所有与后端API的交互方法
 */
import axios from 'axios';

/**
 * 创建axios实例
 * @baseURL 后端API基础路径
 * @timeout 请求超时时间
 */
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 60000, // 增加超时时间到60秒
});

// 请求拦截器，添加认证令牌
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器，处理认证失败
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response && error.response.status === 401) {
      // 登录过期，清除本地存储并刷新页面
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

/**
 * 上传视频文件
 * @param {FormData} formData - 包含视频文件和弹幕文件的FormData对象
 * @param {Function} onUploadProgress - 上传进度回调函数
 * @returns {Promise} - 返回Promise对象
 */
export const uploadVideo = (formData, onUploadProgress) => {
  return api.post('/videos/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress, // 用于获取上传进度
  });
};

/**
 * 通过URL上传视频
 * @param {string} url - B站视频URL
 * @returns {Promise} - 返回Promise对象
 */
export const uploadVideoByUrl = (url) => {
  const formData = new FormData();
  formData.append('url', url);
  return api.post('/videos/upload/url', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

/**
 * 获取视频详情
 * @param {number} id - 视频ID
 * @returns {Promise} - 返回Promise对象
 */
export const getVideoDetail = (id) => api.get(`/videos/${id}`);

/**
 * 获取视频列表
 * @returns {Promise} - 返回Promise对象
 */
export const getVideoList = () => api.get('/videos');

/**
 * 删除视频
 * @param {number} id - 视频ID
 * @returns {Promise} - 返回Promise对象
 */
export const deleteVideo = (id) => api.delete(`/videos/${id}`);

/**
 * 获取系统配置
 * @returns {Promise} - 返回Promise对象
 */
export const getConfig = () => api.get('/config');

/**
 * 更新系统配置
 * @param {Object} config - 配置对象
 * @returns {Promise} - 返回Promise对象
 */
export const updateConfig = (config) => api.put('/config', config);

/**
 * 获取敏感词列表
 * @returns {Promise} - 返回Promise对象
 */
export const getSensitiveWords = () => api.get('/sensitive-words');

/**
 * 添加敏感词
 * @param {string} word - 敏感词
 * @returns {Promise} - 返回Promise对象
 */
export const addSensitiveWord = (word) => api.post('/sensitive-words', { word });

/**
 * 删除敏感词
 * @param {string} word - 敏感词
 * @returns {Promise} - 返回Promise对象
 */
export const deleteSensitiveWord = (word) => api.delete(`/sensitive-words/${word}`);

/**
 * 批量更新敏感词
 * @param {Array} words - 敏感词列表
 * @returns {Promise} - 返回Promise对象
 */
export const updateSensitiveWords = (words) => api.put('/sensitive-words', words);

/**
 * 获取登录二维码
 * @returns {Promise} - 返回Promise对象
 */
export const getLoginQRCode = () => api.get('/auth/qr-code');

/**
 * 检查二维码状态
 * @param {string} qrKey - 二维码key
 * @returns {Promise} - 返回Promise对象
 */
export const checkQRStatus = (qrKey) => api.get(`/auth/qr-status/${qrKey}`);

/**
 * 获取当前用户信息
 * @returns {Promise} - 返回Promise对象
 */
export const getCurrentUser = () => api.get('/auth/me');

/**
 * 登出
 * @returns {Promise} - 返回Promise对象
 */
export const logout = () => api.post('/auth/logout');


