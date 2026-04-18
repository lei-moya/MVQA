"""
MVQA 推理与预处理：加载多模态子模型与 traced 融合网络，对单条视频（及可选弹幕）输出片段级与视频级分数。

``predict_video_qa(..., user_id=...)`` 通过上下文变量将配置解析到对应用户的 ``Setting``；
未传入时回退为 user_id=1（兼容脚本直接调用）。
"""

import contextvars
import os
from typing import Optional, Tuple

import numpy as np
import torch

from backend.model.mvqa import MVQA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MVQA_CONFIG_USER_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "mvqa_config_user_id", default=None
)

# ==========================================
# 配置参数
# ==========================================
from backend.config import config_manager, DEFAULT_CONFIG
from backend.database import SessionLocal

class Config:
    # 融合层维度
    FUSION_DIM = 512
    # 输出维度
    OUTPUT_DIM = 12
    
    @classmethod
    def get_config(cls):
        """从数据库加载与当前 MVQA 上下文对应的用户配置（默认用户 1）。"""
        db = SessionLocal()
        try:
            uid = MVQA_CONFIG_USER_ID.get()
            if uid is None:
                uid = 1
            CONFIG = config_manager.get_config(user_id=uid, db=db)
        except Exception as e:
            print(f"加载配置失败: {e}")
            CONFIG = DEFAULT_CONFIG
        finally:
            db.close()
        return CONFIG
    
    @classmethod
    def get_whisper_model(cls):
        return cls.get_config().get("model_paths", {}).get("whisper_model", os.path.join(BASE_DIR, "model/whisper-medium"))
    
    @classmethod
    def get_audio_model(cls):
        return cls.get_config().get("model_paths", {}).get("audio_model", os.path.join(BASE_DIR, "model/ast-finetuned-audioset-10-10-0.4593"))
    
    @classmethod
    def get_text_model(cls):
        return cls.get_config().get("model_paths", {}).get("text_model", os.path.join(BASE_DIR, "model/roberta-wwm-ext"))
    
    @classmethod
    def get_visual_model(cls):
        return cls.get_config().get("model_paths", {}).get("visual_model", os.path.join(BASE_DIR, "model/vit-base-patch16-224"))
    
    @classmethod
    def get_sensitive_words(cls):
        return cls.get_config().get("sensitive_words", ["违禁词", "暴力", "涉黄", "反动", "测试", "傻叉", "漫展", "三十"])
    
    @classmethod
    def get_num_clips(cls):
        return cls.get_config().get("video_processing", {}).get("num_clips", 125)
    
    @classmethod
    def get_frames_per_clip(cls):
        return cls.get_config().get("video_processing", {}).get("frames_per_clip", 5)
    
    @classmethod
    def get_target_size(cls):
        return tuple(cls.get_config().get("video_processing", {}).get("target_size", [224, 224]))
    
    @classmethod
    def get_sample_rate(cls):
        return cls.get_config().get("audio_processing", {}).get("sample_rate", 16000)
    
    @classmethod
    def get_traced_model_path(cls):
        return cls.get_config().get("model_paths", {}).get("traced_model_path", os.path.join(BASE_DIR, "model/mvqa_traced.pt"))


# ==========================================
# 核心工具类导入
# ==========================================
from backend.utils.text_utils import AsideFilter, ASSParser, FineGrainedAligner
from backend.utils.video_utils import VideoFrameExtractor
from backend.utils.audio_utils import VideoAudioSegmentProcessor

# ==========================================
# 数据预处理器
# ==========================================
from transformers import (
    AutoTokenizer,
    ViTImageProcessor,
    ASTFeatureExtractor
)

class MVQAPreprocessor:
    """MVQA数据预处理器"""

    def __init__(self):
        # 加载各模态的processor
        visual_model = Config.get_visual_model()
        audio_model = Config.get_audio_model()
        text_model = Config.get_text_model()
        self.image_processor = ViTImageProcessor.from_pretrained(visual_model)
        self.audio_processor = ASTFeatureExtractor.from_pretrained(audio_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)

    def process_audio(self, audio_array, sample_rate=16000):
        """
        处理音频数据
        Args:
            audio_array: numpy array (num_clips, samples,)
            sample_rate: 采样率
        Returns:
            处理后的音频特征 (num_clips, 128, 1024)
        """
        try:
            inputs = self.audio_processor(
                list(audio_array),
                sampling_rate=sample_rate,
                return_tensors="pt",
                truncation=True,  # 强制截断
                max_length=1024  # AST 标准长度
            )

            # 【关键修正】
            # AST 模型期望输入形状：(Batch, Frequency=128, Time=1024)
            # ASTFeatureExtractor 默认输出正是：(Batch, 128, 1024)
            # 因此，不需要任何转置操作！

            input_values = inputs["input_values"]

            if input_values.shape[-1] == 128 and input_values.shape[-2] == 1024:
                input_values = input_values.transpose(1, 2)

            return input_values
        except Exception as e:
            print(f"音频处理出错: {e}")
            return torch.zeros((audio_array.shape[0], 128, 1024))

    def process_text(self, text, max_length=512):
        """
        处理文本数据
        Args:
            text: 字符串 或 字符串列表
            max_length: 最大长度
        Returns:
            input_ids, attention_mask
        """
        inputs = self.text_tokenizer(
            text,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        return inputs["input_ids"], inputs["attention_mask"]

    def process_frames(self, frames):
        """
        处理视频帧
        Args:
            frames: numpy array (300, num_frames, H, W, C) 或 (300, num_frames, C, H, W)
        Returns:
            处理后的像素值 (300, num_frames, C, H, W)
        """

        # 1. 通道顺序处理
        # 如果最后一维是3，说明是 (Batch, Time, H, W, C)，需要转为 (Batch, Time, C, H, W)
        if frames.shape[-1] == 3:
            # (125, T, H, W, C) -> (125, T, C, H, W)
            frames = np.transpose(frames, (0, 1, 4, 2, 3))

        # 此时 frames 形状应为 (125, Time, C, H, W)

        # 2. 归一化到 0-255
        # 注意：这里对整个数组判断，如果有任何一个值大于1.0则不缩放。
        # 更安全的做法通常是判断数据类型是否为 float 且最大值小于等于1
        if frames.max() <= 1.0:
            frames = frames * 255.0

        # 3. 展平批次和时间维度
        # ViT处理器通常只接受 (N, C, H, W)，因此需要将前两维合并
        # (125, Time, C, H, W) -> (125 * Time, C, H, W)
        frames_flat = frames.reshape(-1, *frames.shape[2:])

        # 4. 使用ViT处理器
        # image_processor 会处理 resize, normalization 等操作
        # 输入: (Batch*Time, C, H, W) numpy array
        # 输出: pixel_values tensor
        inputs = self.image_processor(
            frames_flat,
            return_tensors="pt",
            do_rescale=False,
        )

        pixel_values = inputs["pixel_values"]  # shape: (Batch*Time, C, H_new, W_new)

        # 5. 恢复形状
        # (125 * Time, C, H, W) -> (125, Time, C, H, W)
        # 注意：这里的 H, W 可能已经被处理器 resize 到模型所需大小 (如 224x224)
        processed_frames = pixel_values.view(Config.get_num_clips(), Config.get_frames_per_clip(), *pixel_values.shape[1:])

        return processed_frames


def _process_single_prediction(pred: np.ndarray, min_diff: float = 1.0) -> np.ndarray:
    """
    处理单个预测样本：前 10 维组内排序并放大差异；第 11 维综合分映射到 0~100；
    第 12 维限制在 [-1, 1]。
    """
    if len(pred) <= 1:
        result = np.array([max(4.0, min(9.0, pred[0]))])
        return np.floor(result).astype(int)

    first_5 = pred[0:5]
    second_5 = pred[5:10]
    score = pred[10]
    mood = pred[11]

    sorted_first = np.sort(first_5)
    sorted_second = np.sort(second_5)
    sorted_indices_first = np.argsort(first_5)
    sorted_indices_second = np.argsort(second_5)

    processed_first_sorted = np.zeros_like(first_5)
    processed_second_sorted = np.zeros_like(second_5)
    max_count = 0

    if len(processed_first_sorted) > 0:
        processed_first_sorted[0] = max(4.0, sorted_first[0])
        for i in range(1, len(processed_first_sorted)):
            target_diff = max(min_diff, (sorted_first[i] - sorted_first[i - 1]) * 1.5)
            next_val = processed_first_sorted[i - 1] + target_diff
            if next_val >= 9.0:
                if max_count == 0:
                    processed_first_sorted[i] = 9.0
                    max_count += 1
                else:
                    processed_first_sorted[i] = 8.0
            else:
                processed_first_sorted[i] = next_val

    if len(processed_second_sorted) > 0:
        processed_second_sorted[0] = max(4.0, sorted_second[0])
        for i in range(1, len(processed_second_sorted)):
            target_diff = max(min_diff, (sorted_second[i] - sorted_second[i - 1]) * 1.5)
            next_val = processed_second_sorted[i - 1] + target_diff
            if next_val >= 9.0:
                if max_count == 0:
                    processed_second_sorted[i] = 9.0
                    max_count += 1
                else:
                    processed_second_sorted[i] = 8.0
            else:
                processed_second_sorted[i] = next_val

    processed_score = float(score)
    if processed_score > 100.0:
        processed_score = 100.0
    elif processed_score > 10.0:
        processed_score = min(100.0, processed_score)
    else:
        processed_score = processed_score * 10.0
    processed_score = max(0.0, min(100.0, processed_score))

    result_first_5 = np.zeros_like(first_5)
    result_second_5 = np.zeros_like(second_5)
    if len(result_first_5) > 0 and len(processed_first_sorted) > 0:
        result_first_5[sorted_indices_first] = processed_first_sorted
    if len(result_second_5) > 0 and len(processed_second_sorted) > 0:
        result_second_5[sorted_indices_second] = processed_second_sorted

    processed_mood = max(-1.0, min(1.0, float(mood)))

    result = np.concatenate([result_first_5, result_second_5, [processed_score], [processed_mood]])
    result[:11] = np.floor(result[:11]).astype(int)
    result[-1] = max(-1.0, min(1.0, result[-1]))
    return result


def postprocess_predictions(predictions: np.ndarray, min_diff: float = 1.0) -> np.ndarray:
    """对 batch 或单条回归向量做与训练展示一致的后处理。"""
    if predictions.ndim == 2:
        return np.array([_process_single_prediction(p, min_diff) for p in predictions])
    return _process_single_prediction(predictions, min_diff)


def _postprocess_mvqa_dict_like(
    video_outputs: torch.Tensor, clip_outputs: Optional[torch.Tensor] = None
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """对 ``MVQA`` 输出的 video / clip 张量做后处理，返回 (clip_processed, video_processed)。"""
    video_np = video_outputs.detach().cpu().numpy()
    if video_np.ndim == 2 and video_np.shape[0] == 1:
        video_np = video_np[0]

    processed_video = postprocess_predictions(video_np, min_diff=1.0)
    video_t = torch.from_numpy(np.asarray(processed_video))

    if clip_outputs is None:
        return None, video_t

    clip_rows = [item for sublist in clip_outputs.detach().cpu().numpy() for item in sublist]
    processed_clip = np.array(
        [postprocess_predictions(np.asarray(row), min_diff=1.0) for row in clip_rows]
    )
    return torch.from_numpy(processed_clip), video_t


# ==========================================
# 核心预测函数
# ==========================================
def predict_video_qa(
    video_path: str,
    ass_path: str = "",
    device: str = "cpu",
    *,
    user_id: Optional[int] = None,
):
    ctx_token = None
    if user_id is not None:
        ctx_token = MVQA_CONFIG_USER_ID.set(user_id)

    try:
        return _predict_video_qa_impl(video_path, ass_path, device)
    finally:
        if ctx_token is not None:
            MVQA_CONFIG_USER_ID.reset(ctx_token)


def _predict_video_qa_impl(video_path: str, ass_path: str = "", device: str = "cpu"):
    # 1. 初始化预处理器
    preprocessor = MVQAPreprocessor()

    # 2. 初始化模型
    # 注意：freeze_pretrained 参数在推理时不影响权重加载，但需要结构与保存时一致
    model = MVQA(
        fusion_dim=Config.FUSION_DIM,
        output_dim=Config.OUTPUT_DIM,
        freeze_pretrained=True
    )
    
    # 先创建模型，不立即移动到设备
    # 后续加载权重后再移动

    # 3. 加载模型权重
    traced_model_path = Config.get_traced_model_path()
    if os.path.exists(traced_model_path):
        try:
            # 尝试加载 state_dict (推荐方式)
            state_dict = torch.load(traced_model_path, map_location=device, weights_only=False)
            # 兼容处理：如果加载的是整个模型对象而非字典
            if isinstance(state_dict, dict):
                # 先加载权重，再移动到设备
                model.load_state_dict(state_dict)
                # 移动模型到设备，使用try-except处理meta tensor错误
                try:
                    model.to(device)
                except RuntimeError:
                    # 如果遇到meta tensor错误，使用to_empty()
                    model.to_empty(device=device)
            else:
                # 如果 torch.load 返回的不是字典，可能已经是模型对象或 JIT 模块
                model = state_dict
                # 确保模型在正确的设备上
                try:
                    model.to(device)
                except RuntimeError:
                    # 如果遇到meta tensor错误，使用to_empty()
                    model.to_empty(device=device)
        except Exception as e:
            try:
                model = torch.jit.load(traced_model_path, map_location=device)
                # 确保模型在正确的设备上
                try:
                    model.to(device)
                except RuntimeError:
                    # 如果遇到meta tensor错误，使用to_empty()
                    model.to_empty(device=device)
            except Exception as je:
                return None, None
    else:
        # 移动随机初始化的模型到设备，使用try-except处理meta tensor错误
        try:
            model.to(device)
        except RuntimeError:
            # 如果遇到meta tensor错误，使用to_empty()
            model.to_empty(device=device)

    model.eval()

    # ==========================================
    # 数据预处理流程
    # ==========================================

    # 4. 视觉模态
    video_extractor = VideoFrameExtractor()
    frames, timestamps = video_extractor.extract_frames_with_timestamps(video_path)
    # frames: (num_clips, T, H, W, 3) -> process_frames -> (num_clips, T, C, H, W)
    processed_frames = preprocessor.process_frames(frames)
    # 增加 batch 维度 -> (1, num_clips, T, C, H, W)
    processed_frames = processed_frames.unsqueeze(0).to(device)

    # 5. 音频模态
    # 【修正3】使用修正后的 VideoAudioSegmentProcessor
    audio_extractor = VideoAudioSegmentProcessor()
    # process_video 现在返回原始波形 (num_clips, samples)
    waveforms = audio_extractor.process_video(video_path)
    # process_audio 接收波形并转换为 AST 输入格式 (num_clips, 128, 1024)
    processed_audio = preprocessor.process_audio(waveforms)
    # 增加 batch 维度 -> (1, num_clips, 128, 1024)
    processed_audio = processed_audio.unsqueeze(0).to(device)

    # 6. 文本模态 - 旁白
    try:
        # 【修正4】AsideFilter 初始化需要传入敏感词列表
        aside_filter = AsideFilter()
        # process_video 返回 (matches, text)，我们只需要 text
        _, aside_text = aside_filter.process_video(video_path)
    except Exception as e:
        aside_text = ""

    # process_text 返回 (1, seq_len)，已经是二维，符合模型输入 (batch, seq_len)
    aside_ids, aside_mask = preprocessor.process_text(aside_text)
    aside_ids, aside_mask = aside_ids.to(device), aside_mask.to(device)

    # 7. 文本模态 - 弹幕
    num_clips = Config.get_num_clips()
    danmu_texts = [""] * num_clips
    if os.path.exists(ass_path):
        parser = ASSParser()
        danmaku_list = parser.parse(ass_path)
        # 【修正5】使用 FineGrainedAligner，替代 DanmuSegmentAligner
        aligner = FineGrainedAligner(danmaku_list)
        # align 方法接收二维 timestamps，返回 List[str] (长度为 num_clips)
        danmu_texts = aligner.align(timestamps)

    # process_text 接收字符串列表，返回
    danmu_ids, danmu_mask = preprocessor.process_text(danmu_texts)
    # 增加 batch 维度 -> (1, num_clips, seq_len)
    danmu_ids, danmu_mask = danmu_ids.unsqueeze(0).to(device), danmu_mask.unsqueeze(0).to(device)

    # ==========================================
    # 模型推理
    # ==========================================
    with torch.no_grad():
        outputs = model(
            processed_audio,
            aside_ids,
            aside_mask,
            danmu_ids,
            danmu_mask,
            processed_frames
        )
    # 兼容 dict（eager 版 MVQA）或 tuple（部分 TorchScript 导出为 (clip_outputs, video_outputs)）
    if isinstance(outputs, dict):
        clip_t = outputs.get("clip_outputs")
        return _postprocess_mvqa_dict_like(outputs["video_outputs"], clip_t)

    if isinstance(outputs, tuple) and len(outputs) == 2:
        a, b = outputs[0], outputs[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # 视频级 (B,12) 与片段级 (B,T,12) 可区分；否则默认 (clip, video) 与常见导出一致
            if a.ndim == 2 and b.ndim == 3 and a.shape[-1] == 12 and b.shape[-1] == 12:
                return _postprocess_mvqa_dict_like(a, b)
            if b.ndim == 2 and a.ndim == 3 and a.shape[-1] == 12 and b.shape[-1] == 12:
                return _postprocess_mvqa_dict_like(b, a)
            return _postprocess_mvqa_dict_like(b, a)

    return None, None

