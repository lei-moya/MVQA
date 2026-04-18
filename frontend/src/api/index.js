/**
 * 后端 API 封装（axios）
 *
 * - 开发：不设 VITE_BACKEND_ORIGIN 时使用相对路径 `/api`，由 Vite proxy 转发。
 * - 生产：构建前设置 VITE_BACKEND_ORIGIN=https://你的后端域名（无尾部斜杠）。
 * JWT 存于 localStorage `token`；401 时清空凭证并回首页（响应拦截器）。
 */
import axios from 'axios';

const rawOrigin = import.meta.env.VITE_BACKEND_ORIGIN;
/** 后端根地址，空字符串表示与当前页面同源（依赖 dev 代理或反向代理） */
export const BACKEND_ORIGIN =
  rawOrigin != null && String(rawOrigin).trim() !== ''
    ? String(rawOrigin).replace(/\/$/, '')
    : '';

const api = axios.create({
  baseURL: BACKEND_ORIGIN ? `${BACKEND_ORIGIN}/api` : '/api',
  timeout: 60000,
});

/** 401 时仅跳转一次，避免并发请求触发多次 reload */
let authRedirectScheduled = false;

/**
 * 统一解析 FastAPI/axios 错误文案（detail 可为 string 或校验错误数组）
 * @param {unknown} error
 * @param {string} [fallback]
 * @returns {string}
 */
export function getApiErrorDetail(error, fallback = '网络异常，请稍后重试') {
  const res = error && error.response;
  if (!res || !res.data) {
    if (error && error.message === 'Network Error') {
      return '无法连接服务器，请检查网络或后端是否已启动';
    }
    return fallback;
  }
  const d = res.data.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    try {
      return d
        .map((x) => (x.msg ? `${x.loc?.join?.('.') || ''}: ${x.msg}` : JSON.stringify(x)))
        .join('; ');
    } catch {
      return fallback;
    }
  }
  if (d != null && typeof d === 'object' && d.message) return String(d.message);
  return fallback;
}

/** 带鉴权 query 的视频流地址（供 <video src> 使用） */
export const videoStreamUrl = (videoId, token) => {
  const q = token ? `?token=${encodeURIComponent(token)}` : '';
  const path = `/api/videos/${videoId}/stream${q}`;
  return BACKEND_ORIGIN ? `${BACKEND_ORIGIN}${path}` : path;
};

/** 静态挂载的弹幕等文件（filename 可为完整路径，仅取最后一段） */
export const uploadsPublicUrl = (filePathOrName) => {
  const name = filePathOrName ? String(filePathOrName).split(/[/\\]/).pop() : '';
  const path = `/uploads/${encodeURIComponent(name)}`;
  return BACKEND_ORIGIN ? `${BACKEND_ORIGIN}${path}` : path;
};

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
      if (authRedirectScheduled) {
        return Promise.reject(error);
      }
      authRedirectScheduled = true;
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      if (window.location.pathname !== '/') {
        window.location.href = '/';
      } else {
        window.location.reload();
      }
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

/** 与后端 ``backend/constants.VIDEO_LIST_DEFAULT_LIMIT`` 一致，用于无限滚动分页 */
export const VIDEO_LIST_PAGE_SIZE = 30;

/**
 * 分页获取视频列表（响应为 ``{ items, total, skip, limit }``，非数组）
 * @param {{ skip?: number, limit?: number }} params — 未传 limit 时使用 {@link VIDEO_LIST_PAGE_SIZE}
 */
export const getVideoList = (params = {}) =>
  api.get('/videos', {
    params: { limit: VIDEO_LIST_PAGE_SIZE, ...params },
  });

/**
 * 批量本地上传（仅视频文件，多 part 同名 ``videos``）
 * @param {FormData} formData
 */
export const uploadVideoBatch = (formData, onUploadProgress) =>
  api.post('/videos/upload/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  });

/** 对已有本地文件重新入队分析 */
export const reanalyzeVideo = (id) => api.post(`/videos/${id}/reanalyze`);

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
 * 添加敏感词（管理员；body 与后端 SensitiveWordCreate 一致）
 * @param {string|{word:string,category?:string,is_regex?:boolean,is_whitelist?:boolean,action?:string}} payload
 */
export const addSensitiveWord = (payload) => {
  const body =
    typeof payload === 'string'
      ? { word: payload, category: '', is_regex: false, is_whitelist: false, action: 'block' }
      : payload;
  return api.post('/sensitive-words', body);
};

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

/** 管理员：查看环境变量 / 库内 / 合并后的 B 站管理员 mid */
export const getAdminBilibiliMids = () => api.get('/admin/bilibili-mids');

/** 管理员：保存库内管理员 mid（逗号分隔，与环境变量合并） */
export const updateAdminBilibiliMids = (stored_mids) =>
  api.put('/admin/bilibili-mids', { stored_mids });

/** 校验 B 站 sessdata 是否仍有效 */
export const refreshBilibiliSession = () => api.post('/auth/bilibili/refresh');

/** 管理员：敏感词完整规则 */
export const getSensitiveWordRules = () => api.get('/sensitive-words/rules');

/** 普通用户：提交敏感词添加申请 */
export const requestSensitiveWord = (body) => api.post('/sensitive-words/requests', body);

/** 管理员：待审核申请列表 */
export const getPendingSensitiveWordRequests = () =>
  api.get('/sensitive-words/requests/pending');

/** 当前用户：我的申请记录 */
export const getMySensitiveWordRequests = () => api.get('/sensitive-words/requests/mine');

export const approveSensitiveWordRequest = (id) =>
  api.post(`/sensitive-words/requests/${id}/approve`);

export const rejectSensitiveWordRequest = (id) =>
  api.post(`/sensitive-words/requests/${id}/reject`);

/**
 * 登出
 * @returns {Promise} - 返回Promise对象
 */
export const logout = () => api.post('/auth/logout');


