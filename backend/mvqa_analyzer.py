import os
import re
import cv2
import wave
import torch
import numpy as np

# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 第三方库导入
# ==========================================
from moviepy import VideoFileClip
from transformers import (
    AutoTokenizer,
    ViTImageProcessor,
    ASTFeatureExtractor
)


# ==========================================
# 1. 配置参数
# ==========================================
from backend.config import CONFIG

class Config:
    # 模型路径 - 使用配置文件中的路径
    VIDEO_MODEL = CONFIG["model_paths"]["video_model"]
    VIDEO_SCORER = CONFIG["model_paths"]["video_scorer"]
    AUDIO_MODEL = CONFIG["model_paths"]["audio_model"]
    TEXT_MODEL = CONFIG["model_paths"]["text_model"]
    VISUAL_MODEL = CONFIG["model_paths"]["visual_model"]

    # 敏感词
    SENSITIVE_WORDS = CONFIG["sensitive_words"]

    # 视频处理参数
    NUM_CLIPS = CONFIG["video_processing"]["num_clips"]
    FRAMES_PER_CLIP = CONFIG["video_processing"]["frames_per_clip"]
    TARGET_SIZE = tuple(CONFIG["video_processing"]["target_size"])

    # 音频处理参数
    SAMPLE_RATE = CONFIG["audio_processing"]["sample_rate"]

    # Traced模型路径
    TRACED_MODEL_PATH = CONFIG["model_paths"]["traced_model_path"]


# ==========================================
# 2. 核心工具类定义
# ==========================================

# --- 文本处理工具 ---
class DFAModel:
    """基于DFA算法的敏感词过滤"""

    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'

    def add_word(self, word):
        word = word.strip().lower()
        if not word: return
        level = self.keyword_chains
        for char in word:
            if char not in level: level[char] = {}
            level = level[char]
        level[self.delimit] = word

    def parse(self, words_list):
        for word in words_list: self.add_word(word)

    def filter_match(self, text):
        text = text.lower()
        matched_words = []
        start = 0
        while start < len(text):
            level = self.keyword_chains
            step_ins = 0
            flag = False
            current_word = ""
            for i in range(start, len(text)):
                char = text[i]
                if char in level:
                    step_ins += 1
                    current_word += char
                    level = level[char]
                    if self.delimit in level:
                        matched_words.append(current_word)
                        flag = True
                        break
                else:
                    break
            if flag:
                start += step_ins
            else:
                start += 1
        return list(set(matched_words))


class AsideFilter:
    def __init__(self, sensitive_words):
        self.dfa_model = DFAModel()
        self.dfa_model.parse(words_list=sensitive_words)
        self.ds = None
        try:
            from deepspeech import Model
            # if os.path.exists(Config.VIDEO_MODEL):
            #     self.ds = Model(Config.VIDEO_MODEL)
            #     self.ds.enableExternalScorer(Config.VIDEO_SCORER)
            #     print("Deepspeech 模型加载成功。")
            # else:
            #     print(f"[警告] 未找到语音模型: {Config.VIDEO_MODEL}，旁白功能将被禁用。")
        except ImportError:
            print("[警告] 未安装 deepspeech 库，旁白功能将被禁用。")

    def extract_audio(self, video_path, audio_save_path=None):
        try:
            # 生成唯一的临时文件名，避免并发处理时的冲突
            if audio_save_path is None:
                import uuid
                audio_save_path = f"temp_audio_{uuid.uuid4().hex}.wav"
            
            clip = VideoFileClip(video_path)
            if clip.audio is None: return None
            clip.audio.write_audiofile(
                audio_save_path, fps=16000, nbytes=2, codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"], logger=None
            )
            return audio_save_path
        except Exception as e:
            print(f"音频提取失败: {e}")
            return None

    def load_wav(self, path):
        with wave.open(path, "r") as wf:
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16)

    def transcribe_audio(self, audio_path):
        if not self.ds: return ""
        try:
            audio = self.load_wav(audio_path)
            return self.ds.stt(audio)
        except Exception as e:
            print(f"语音识别出错: {e}")
            return ""

    def process_video(self, video_path):
        audio_path = self.extract_audio(video_path)
        if not audio_path: return [], ""
        text = self.transcribe_audio(audio_path)
        matches = self.dfa_model.filter_match(text)
        if os.path.exists(audio_path): os.remove(audio_path)
        return matches, text


class ASSParser:
    def __init__(self):
        self.dialogue_pattern = re.compile(
            r"Dialogue:\s*(.*?)\s*,(\d:\d{2}:\d{2}\.\d{2}),(\d:\d{2}:\d{2}\.\d{2}),.*?,.*?,.*?,.*?,.*?,.*?,(.*)")
        self.tag_pattern = re.compile(r"\{.*?\}")

    def _time_to_seconds(self, time_str):
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def _clean_text(self, text):
        text = self.tag_pattern.sub('', text)
        return text.replace('\\N', ' ').strip()

    def parse(self, file_path):
        danmaku_list = []
        if not os.path.exists(file_path): return danmaku_list
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("Dialogue:"):
                    match = self.dialogue_pattern.match(line)
                    if match:
                        danmaku_list.append({
                            'start': self._time_to_seconds(match.group(2)),
                            'end': self._time_to_seconds(match.group(3)),
                            'text': self._clean_text(match.group(4))
                        })
        return danmaku_list


class DanmuSegmentAligner:
    def __init__(self, danmaku_list):
        self.danmaku_list = sorted(danmaku_list, key=lambda x: x['start'])

    def align_by_clips(self, clip_timestamps):
        num_clips = clip_timestamps.shape[0]
        clip_texts = []
        for i in range(num_clips):
            t_start = clip_timestamps[i][0]
            t_end = clip_timestamps[i][-1]
            segment_text = []
            for d in self.danmaku_list:
                if d['start'] <= t_end and d['end'] >= t_start:
                    segment_text.append(d['text'])
                if d['start'] > t_end: break
            clip_texts.append(" ".join(segment_text) if segment_text else "")
        return clip_texts


# --- 视频处理工具 ---
class VideoFrameExtractor:
    def __init__(self):
        self.num_clips = Config.NUM_CLIPS
        self.frames_per_clip = Config.FRAMES_PER_CLIP
        self.target_size = Config.TARGET_SIZE

    def _calculate_sample_indices(self, total_frames):
        if total_frames <= self.num_clips:
            indices = np.linspace(0, total_frames - 1, self.num_clips * self.frames_per_clip, dtype=int)
            return indices.reshape(self.num_clips, self.frames_per_clip)
        clip_boundaries = np.linspace(0, total_frames, self.num_clips + 1, dtype=int)
        all_indices = []
        for i in range(self.num_clips):
            start_f, end_f = clip_boundaries[i], clip_boundaries[i + 1]
            clip_len = end_f - start_f
            if clip_len == 0:
                segment_indices = np.full(self.frames_per_clip, start_f - 1 if start_f > 0 else 0)
            else:
                segment_indices = np.linspace(start_f, end_f - 1, self.frames_per_clip, dtype=int)
            all_indices.append(segment_indices)
        return np.array(all_indices)

    def extract_frames_with_timestamps(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f"无法打开视频: {video_path}")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            sample_indices = self._calculate_sample_indices(total_frames)
            frames_list, timestamps_list = [], []
            for clip_idx in range(self.num_clips):
                clip_frames, clip_timestamps = [], []
                for frame_idx in sample_indices[clip_idx]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                    else:
                        frame = cv2.resize(frame, self.target_size)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = frame.astype(np.float32) / 255.0
                    clip_frames.append(frame)
                    clip_timestamps.append(frame_idx / fps)
                frames_list.append(clip_frames)
                timestamps_list.append(clip_timestamps)
            return np.array(frames_list).astype(np.float32), np.array(timestamps_list).astype(np.float32)
        finally:
            cap.release()


# --- 音频处理工具 ---
class AudioSegmentWaveformExtractor:
    def __init__(self):
        self.n_segments = Config.NUM_CLIPS

    def extract_segment_waveforms(self, video_path):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is None:
            video_clip.close()
            raise ValueError("视频无音轨")

        duration = audio_clip.duration
        segment_length = duration / self.n_segments
        waveforms = []

        try:
            for i in range(self.n_segments):
                start = i * segment_length
                end = start + segment_length if i < self.n_segments - 1 else duration
                segment_clip = audio_clip[start:end]
                y_segment = segment_clip.to_soundarray(fps=Config.SAMPLE_RATE)
                if y_segment.ndim > 1: y_segment = y_segment.mean(axis=1)
                waveforms.append(y_segment.astype(np.float32))
            return waveforms, duration
        finally:
            video_clip.close()


# --- 数据预处理器 ---
class MVQAPreprocessor:
    def __init__(self):
        print("初始化预处理器...")
        try:
            self.image_processor = ViTImageProcessor.from_pretrained(Config.VISUAL_MODEL)
            print(f"✓ 视觉模型处理器加载成功: {Config.VISUAL_MODEL}")
        except Exception as e:
            print(f"✗ 视觉模型处理器加载失败: {e}")
            raise
        
        try:
            self.audio_processor = ASTFeatureExtractor.from_pretrained(Config.AUDIO_MODEL)
            print(f"✓ 音频模型处理器加载成功: {Config.AUDIO_MODEL}")
        except Exception as e:
            print(f"✗ 音频模型处理器加载失败: {e}")
            raise
        
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL)
            print(f"✓ 文本模型处理器加载成功: {Config.TEXT_MODEL}")
        except Exception as e:
            print(f"✗ 文本模型处理器加载失败: {e}")
            raise
        
        print("预处理器初始化完成\n")

    def process_audio(self, audio_array, sample_rate=16000):
        """
        处理音频数据
        Args:
            audio_array: List of numpy arrays (waveforms)
        Returns:
            Tensor: (num_clips, 128, 1024)
        """
        inputs = self.audio_processor(
            list(audio_array),
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        input_values = inputs["input_values"]

        # 【修正2】移除错误的转置逻辑
        # ASTFeatureExtractor 输出格式为: (Batch, Time=1024, Freq=128)
        # AST 模型期望输入格式: (Batch, Time=1024, Freq=128)
        # 两者一致，无需转置。
        # 原代码中的转置会导致将时间和频率维度颠倒，破坏特征。

        return input_values

    def process_text(self, text, max_length=512):
        inputs = self.text_tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return inputs["input_ids"], inputs["attention_mask"]

    def process_frames(self, frames):
        num_frames = frames.shape[1]
        # (N, T, H, W, C) -> (N, T, C, H, W)
        if frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 1, 4, 2, 3))
        if frames.max() <= 1.0: frames = frames * 255.0
        frames_flat = frames.reshape(-1, *frames.shape[2:])
        inputs = self.image_processor(frames_flat, return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"]
        # 恢复 (N, T, C, H, W)
        processed_frames = pixel_values.view(Config.NUM_CLIPS, num_frames, *pixel_values.shape[1:])
        return processed_frames


# ==========================================
# 3. 核心预测函数
# ==========================================
def predict_video_qa(video_path: str, ass_path: str = "", device: str = 'cpu'):
    print(f"\n{'=' * 60}")
    print(f"开始处理视频: {video_path}")
    print(f"{'=' * 60}")

    # 1. 初始化
    preprocessor = MVQAPreprocessor()
    model_path = Config.TRACED_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return None, None

    print(f"加载追踪模型: {model_path}")
    try:
        # 加载 TorchScript 模型
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return None, None

    # 2. 视频帧
    print("[步骤 1] 提取视频帧...")
    video_extractor = VideoFrameExtractor()
    frames, timestamps = video_extractor.extract_frames_with_timestamps(video_path)
    processed_frames = preprocessor.process_frames(frames).unsqueeze(0).to(device)

    # 3. 音频
    print("[步骤 2] 提取音频片段...")
    audio_extractor = AudioSegmentWaveformExtractor()
    waveforms, _ = audio_extractor.extract_segment_waveforms(video_path)
    processed_audio = preprocessor.process_audio(waveforms).unsqueeze(0).to(device)

    # 4. 旁白
    print("[步骤 3] 提取旁白...")
    aside_text = ""
    try:
        aside_filter = AsideFilter(Config.SENSITIVE_WORDS)
        _, aside_text = aside_filter.process_video(video_path)
        print(f"  -> 旁白: {aside_text[:30]}...")
    except Exception as e:
        print(f"  -> 旁白提取跳过 ({e})")

    aside_ids, aside_mask = preprocessor.process_text(aside_text)
    aside_ids, aside_mask = aside_ids.to(device), aside_mask.to(device)

    # 5. 弹幕
    print("[步骤 4] 处理弹幕...")
    danmu_texts = [""] * Config.NUM_CLIPS
    if os.path.exists(ass_path):
        parser = ASSParser()
        danmaku_list = parser.parse(ass_path)
        aligner = DanmuSegmentAligner(danmaku_list)
        danmu_texts = aligner.align_by_clips(timestamps)

    danmu_ids, danmu_mask = preprocessor.process_text(danmu_texts)
    danmu_ids, danmu_mask = danmu_ids.unsqueeze(0).to(device), danmu_mask.unsqueeze(0).to(device)

    # 6. 推理
    print("[步骤 5] 模型推理...")
    with torch.no_grad():
        # TorchScript 模型调用通常按位置传参
        outputs = model(
            processed_audio, aside_ids, aside_mask,
            danmu_ids, danmu_mask, processed_frames
        )

    # 兼容处理：JIT 模型输出可能是字典也可能是元组，需根据实际情况解析
    # 假设输出为字典格式 (与 train.py 中的结构一致)
    if isinstance(outputs, dict):
        clip_outputs = outputs['clip_outputs'].cpu().numpy()[0]
        video_outputs = outputs['video_outputs'].cpu().numpy()[0]
    elif isinstance(outputs, tuple):
        # 如果是元组，假设顺序为 (clip_outputs, video_outputs) 或类似，需根据实际模型结构调整
        # 通常 traced model 如果返回 dict，在 C++ 环境下可能表现为 tuple 或需特殊处理
        # 这里假设 PyTorch JIT 保留了字典返回
        clip_outputs = outputs[0].cpu().numpy()[0]
        video_outputs = outputs[1].cpu().numpy()[0]
    else:
        # 无法解析
        clip_outputs, video_outputs = None, None

    print("\n" + "=" * 30)
    print("预测完成!")
    print(f"视频级预测: {video_outputs}")
    print(f"片段级预测: {clip_outputs}")
    print("=" * 30)

    return clip_outputs, video_outputs


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    # 扫描 lab 目录
    LAB_DIR = "./dataset"
    if not os.path.exists(LAB_DIR):
        print(f"错误: 目录不存在 {LAB_DIR}")
    else:
        found_video = False
        for file in os.listdir(LAB_DIR):
            if file.endswith('.mp4'):
                found_video = True
                video_file = os.path.join(LAB_DIR, file)
                base_name = os.path.splitext(file)[0]
                ass_file = os.path.join(LAB_DIR, base_name + '.ass')

                predict_video_qa(
                    video_path=video_file,
                    ass_path=ass_file if os.path.exists(ass_file) else "",
                    device=DEVICE
                )

        if not found_video:
            print(f"在 {LAB_DIR} 中未找到 .mp4 文件。")
