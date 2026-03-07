import os
import hashlib
from functools import lru_cache


@lru_cache(maxsize=128)
def get_file_hash(file_path: str) -> str:
    """生成文件的哈希值，用于缓存"""
    if not os.path.exists(file_path):
        return ""
    
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def analyze_video_quality(file_path: str, ass_path: str = ""):
    """
    对视频进行多维度分析
    
    Args:
        file_path (str): 视频文件路径
        ass_path (str): 弹幕文件路径（可选）
    
    Returns:
        dict: 包含视频分析结果的字典，包括：
            - video_score: 视频级评分数据
            - clip_scores: 片段级评分数据
            - suggestions: 改进建议
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # 尝试导入并使用模型进行分析
        from backend.mvqa_analyzer import predict_video_qa
        
        # 生成文件哈希，用于后续可能的缓存
        file_hash = get_file_hash(file_path)
        ass_hash = get_file_hash(ass_path) if ass_path else ""
        
        clip_outputs, video_outputs = predict_video_qa(file_path, ass_path)

        if video_outputs is None:
            # 如果模型分析失败，返回默认值
            return {
                "video_score": [],
                "clip_scores": [],
                "suggestions": ["分析失败，请重试"]
            }

        # 处理视频评分数据
        video_score_list = video_outputs.tolist() if hasattr(video_outputs, 'tolist') else list(video_outputs)
        
        # 如果没有弹幕文件，确保返回的video_score列表不包含弹幕分
        if not ass_path and len(video_score_list) > 11:
            video_score_list = video_score_list[:11]  # 只保留前11个元素，不包含弹幕分
        
        # 处理片段评分数据
        clip_scores_list = []
        if clip_outputs is not None:
            if hasattr(clip_outputs, 'tolist'):
                clip_outputs_list = clip_outputs.tolist()
                # 确保clip_scores_list是一个二维数组
                if isinstance(clip_outputs_list, list):
                    if clip_outputs_list and not isinstance(clip_outputs_list[0], list):
                        clip_scores_list = [clip_outputs_list]
                    else:
                        clip_scores_list = clip_outputs_list
            elif isinstance(clip_outputs, list):
                clip_scores_list = clip_outputs

        # 生成建议
        suggestions = generate_suggestions(video_score_list, ass_path)

        return {
            "video_score": video_score_list,
            "clip_scores": clip_scores_list,
            "suggestions": suggestions
        }
    except ImportError as e:
        print(f"模型导入失败: {e}")
        return {
            "video_score": [],
            "clip_scores": [],
            "suggestions": ["模型加载失败，请检查模型文件"]
        }
    except Exception as e:
        # 如果模型加载或分析失败，返回默认值
        print(f"模型分析失败: {e}")
        return {
            "video_score": [],
            "clip_scores": [],
            "suggestions": ["分析失败，请重试"]
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
    
    # 分析整体评分
    if len(video_score_list) > 10:
        overall_score = video_score_list[10]
        
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
