"""
业务服务层：在 Web 请求与 MVQA 推理之间做衔接。

``analyze_video_quality`` 负责调用 ``mvqa_analyzer.predict_video_qa``，
将张量输出规范为一维/二维列表，并生成面向用户的 ``suggestions``。
"""

import os
import hashlib
import logging
from functools import lru_cache
from typing import Optional

import torch
import numpy as np

_logger = logging.getLogger(__name__)

# 与 ``main.process_video_task`` 约定：仅当 ``ok`` 为 True 时记为分析成功
MIN_VIDEO_SCORE_DIM = 11


@lru_cache(maxsize=256)
def _get_file_hash_cached(file_path: str, st_mtime_ns: int) -> str:
    """按路径 + 修改时间缓存；文件被替换后 mtime 变化即失效。"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_hash(file_path: str) -> str:
    """生成文件内容的 MD5（元数据参与缓存键，避免同路径替换后读到旧哈希）。"""
    if not os.path.exists(file_path):
        return ""
    st = os.stat(file_path)
    return _get_file_hash_cached(file_path, st.st_mtime_ns)


def analyze_video_quality(file_path: str, ass_path: str = "", user_id: Optional[int] = None):
    """
    对视频进行多维度分析
    
    Args:
        file_path (str): 视频文件路径
        ass_path (str): 弹幕文件路径（可选）
        user_id (Optional[int]): 用于加载该用户在库中的视频/音频处理与敏感词等配置；默认与 MVQA 内部默认一致
    
    Returns:
        dict: 包含视频分析结果的字典，包括：
            - ok: 是否推理成功（供后台任务写库，勿依赖中文文案判断）
            - video_score: 视频级评分数据
            - clip_scores: 片段级评分数据
            - suggestions: 改进建议
            - error_code: 失败时可选，如 ``inference_failed`` / ``import_error``
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        from backend.mvqa_analyzer import predict_video_qa

        clip_outputs, video_outputs = predict_video_qa(
            file_path, ass_path, user_id=user_id
        )

        if video_outputs is None:
            return {
                "ok": False,
                "error_code": "inference_failed",
                "video_score": [],
                "clip_scores": [],
                "suggestions": ["分析失败，请重试"],
            }

        # 处理视频评分数据
        if isinstance(video_outputs, torch.Tensor):
            video_score_list = video_outputs.tolist()
        elif isinstance(video_outputs, list):
            video_score_list = video_outputs
        elif isinstance(video_outputs, np.ndarray):
            video_score_list = video_outputs.flatten().tolist()
        else:
            # 尝试转换为列表
            try:
                video_score_list = list(video_outputs)
            except Exception:
                video_score_list = []
        
        # 确保 video_score_list 是一维列表
        if isinstance(video_score_list, list) and video_score_list and isinstance(video_score_list[0], list):
            # 如果是二维列表，且只有一个元素，则取第一个元素
            if len(video_score_list) == 1:
                video_score_list = video_score_list[0]
        
        # 如果没有弹幕文件，确保返回的video_score列表不包含弹幕分
        if not ass_path and len(video_score_list) > 11:
            video_score_list = video_score_list[:11]  # 只保留前11个元素，不包含弹幕分
        
        # 处理片段评分数据
        clip_scores_list = []
        if clip_outputs is not None:
            if isinstance(clip_outputs, torch.Tensor):
                clip_outputs_list = clip_outputs.tolist()
            elif isinstance(clip_outputs, list):
                clip_outputs_list = clip_outputs
            elif isinstance(clip_outputs, np.ndarray):
                clip_outputs_list = clip_outputs.tolist()
            else:
                # 尝试转换为列表
                try:
                    clip_outputs_list = list(clip_outputs)
                except Exception:
                    clip_outputs_list = []
            
            # 确保clip_scores_list是一个二维数组
            if isinstance(clip_outputs_list, list):
                if clip_outputs_list and not isinstance(clip_outputs_list[0], list):
                    clip_scores_list = [clip_outputs_list]
                else:
                    clip_scores_list = clip_outputs_list
            
            # 处理特殊情况：如果是三维数组（例如 [[[1,2,3]]]），则降为二维
            if isinstance(clip_scores_list, list) and clip_scores_list and isinstance(clip_scores_list[0], list):
                if clip_scores_list[0] and isinstance(clip_scores_list[0][0], list) and len(clip_scores_list) == 1:
                    clip_scores_list = clip_scores_list[0]

        suggestions = generate_suggestions(video_score_list, ass_path)
        analysis_ok = len(video_score_list) >= MIN_VIDEO_SCORE_DIM

        if not analysis_ok:
            return {
                "ok": False,
                "error_code": "incomplete_output",
                "video_score": video_score_list,
                "clip_scores": clip_scores_list,
                "suggestions": [
                    "分析失败：输出维度不足，请检查模型权重与 num_clips 等配置是否一致"
                ],
            }

        return {
            "ok": True,
            "error_code": None,
            "video_score": video_score_list,
            "clip_scores": clip_scores_list,
            "suggestions": suggestions,
        }
    except ImportError as e:
        _logger.warning("MVQA 模块导入失败: %s", e, exc_info=True)
        return {
            "ok": False,
            "error_code": "import_error",
            "video_score": [],
            "clip_scores": [],
            "suggestions": ["模型加载失败，请检查模型文件"],
        }
    except Exception as e:
        _logger.error("视频质量分析异常: %s", e, exc_info=True)
        return {
            "ok": False,
            "error_code": "exception",
            "video_score": [],
            "clip_scores": [],
            "suggestions": ["分析失败，请重试"],
        }


def generate_suggestions(video_score_list: list, ass_path: str) -> list:
    """
    根据视频评分生成改进建议
    
    Args:
        video_score_list: 视频评分列表
        ass_path: 弹幕文件路径
    
    Returns:
        list: 改进建议列表
    """
    suggestions = []
    
    # 分析整体评分（第11维为 0~100 综合分）
    if len(video_score_list) > 10:
        overall_score = float(video_score_list[10])
        
        if overall_score < 40:
            suggestions.append("视频质量较差，建议重新拍摄，注意光线和稳定性。")
        elif overall_score < 60:
            suggestions.append("视频质量一般，建议提高拍摄分辨率或注意对焦。")
        elif overall_score < 80:
            suggestions.append("视频质量良好，可进一步优化画面构图。")
        else:
            suggestions.append("视频质量优秀！")
        
        # 分析弹幕质量
        if len(video_score_list) > 11:
            danmu_score = video_score_list[11]
            if danmu_score < -0.5:
                suggestions.append("弹幕质量较差，建议增加互动性或优化弹幕内容。")
            elif danmu_score < 0:
                suggestions.append("弹幕质量一般，可适当增加有意义的弹幕。")
            else:
                suggestions.append("弹幕质量良好，继续保持互动性。")
        elif ass_path:
            suggestions.append("弹幕分析完成，未发现明显问题。")
    
    # 分析视频各维度得分
    if len(video_score_list) >= 5:
        # 视频维度分析（前5个元素）
        video_dimensions = [
            ("清晰度", video_score_list[0], 5),
            ("色彩", video_score_list[1], 5),
            ("饱和度", video_score_list[2], 5),
            ("稳定性", video_score_list[3], 5),
            ("亮度", video_score_list[4], 5)
        ]
        
        for name, score, threshold in video_dimensions:
            if score < threshold:
                suggestions.append(f"{name}得分较低，建议优化拍摄设置。")
    
    if len(video_score_list) >= 10:
        # 音频维度分析（6-10个元素）
        audio_dimensions = [
            ("音量", video_score_list[5], 5),
            ("音质", video_score_list[6], 5),
            ("噪音", video_score_list[7], 5),
            ("清晰度", video_score_list[8], 5),
            ("均衡", video_score_list[9], 5)
        ]
        
        for name, score, threshold in audio_dimensions:
            if score < threshold:
                suggestions.append(f"{name}得分较低，建议优化音频设置。")
    
    return suggestions if suggestions else ["视频分析完成，未发现明显问题。"]
