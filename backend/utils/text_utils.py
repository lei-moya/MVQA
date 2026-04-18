"""
文本与弹幕：ASS/XML 解析、粗粒度对齐、DFA 敏感词过滤及与 ASR 文本的细粒度对齐，供文本侧特征使用。
"""

from moviepy.video.io.VideoFileClip import VideoFileClip
import wave
from backend.mvqa_analyzer import Config
import re
import os
import numpy as np
from typing import List, Dict, Union
from pathlib import Path

# ---------------------------------------------------------
# 敏感词过滤（DFA）
# ---------------------------------------------------------
class DFAModel:
    """
    基于DFA算法的敏感词过滤
    """

    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'

    def add_word(self, word):
        """添加单个敏感词到字典树"""
        word = word.strip().lower()
        if not word:
            return
        level = self.keyword_chains
        for char in word:
            if char not in level:
                level[char] = {}
            level = level[char]
        level[self.delimit] = word

    def parse(self, path=None, words_list=None):
        """加载敏感词库"""
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for word in f:
                    self.add_word(word)
        if words_list:
            for word in words_list:
                self.add_word(word)

    def filter_match(self, text):
        """匹配文本中的敏感词"""
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


# ---------------------------------------------------------
# 核心业务类
# ---------------------------------------------------------
class AsideFilter:
    def __init__(self):
        self.dfa_model = DFAModel()
        from backend.database import SessionLocal
        from backend.utils.sensitive_rules import literal_words_for_dfa

        db = SessionLocal()
        try:
            self.dfa_model.parse(words_list=literal_words_for_dfa(db))
        finally:
            db.close()
        # 从本地加载模型
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import torch
            local_model_path = Config.get_whisper_model()
            self.processor = WhisperProcessor.from_pretrained(local_model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(local_model_path)
            # 使用to_empty()代替to()来避免meta tensor错误
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model.to(device)
            except RuntimeError:
                # 如果遇到meta tensor错误，使用to_empty()
                self.model.to_empty(device=device)
            self.use_transformers = True
        except ImportError:
            self.model = None
            self.processor = None
            self.use_transformers = False
        except Exception as e:
            self.model = None
            self.processor = None
            self.use_transformers = False

    def extract_audio(self, video_path, audio_save_path="temp_audio.wav"):
        """从视频或音频中提取/转换音频，强制转为 16kHz 单声道"""
        try:
            # 使用 VideoFileClip 可以处理视频，也可以直接处理 mp3 等音频文件
            clip = VideoFileClip(video_path)

            if clip.audio is None:
                return None

            # 关键修改：设置音频参数
            # fps=16000 对应 -ar 16000
            # ffmpeg_params=["-ac", "1"] 对应 -ac 1 (强制单声道)
            # moviepy 默认输出 16bit PCM WAV，对应 -sample_fmt s16
            clip.audio.write_audiofile(
                audio_save_path,
                fps=16000,
                nbytes=2,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"],  # 强制单声道
                logger=None
            )
            return audio_save_path
        except Exception as e:
            return None

    def load_wav(self, path):
        with wave.open(path, "r") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("需要单声道 WAV")
            if wf.getsampwidth() != 2:
                raise ValueError("需要 16bit PCM")
            if wf.getframerate() != 16000:
                raise ValueError("需要 16kHz 采样率")
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16)

    def transcribe_audio(self, audio_path):
        """使用 Whisper 进行语音识别"""
        try:
            if self.model is None or self.processor is None:
                return ""
            
            # 使用 Transformers 的 Whisper 进行语音识别
            import librosa
            import torch
            audio, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # 处理输入
            inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            
            # 生成时添加详细参数以解决警告
            generated_ids = self.model.generate(
                inputs["input_features"],
                language="zh",
                task="transcribe",
                attention_mask=inputs.get("attention_mask"),
                suppress_tokens=None,
                begin_suppress_tokens=None
            )
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text

        except Exception as e:
            return ""

    def process_video(self, video_path):
        """完整流程：视频 -> 音频 -> 识别 -> 敏感词检测"""
        # 1. 提取音频
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return [], ""

        # 2. 语音转文字
        try:
            text = self.transcribe_audio(audio_path)
        except Exception as e:
            text = ""

        # 3. 敏感词：白名单剥离 + DFA + 正则（正则命中可累计 hit_count）
        from backend.database import SessionLocal
        from backend.utils import sensitive_rules as sr

        db = SessionLocal()
        try:
            text_for_match = sr.strip_whitelist_spans(text, db)
            matches = self.dfa_model.filter_match(text_for_match)
            regex_hits = sr.match_regex_on_text(db, text_for_match)
            sr.bump_hit_counts(db, regex_hits)
        finally:
            db.close()

        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return matches, text

class ASSParser:
    """
    解析弹幕文件：ASS（Dialogue 行）或 B 站导出的 XML（``<d p="秒,时长,...">``）。
    """

    def __init__(self):
        # 正则匹配 Dialogue 行
        # 格式: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        self.dialogue_pattern = re.compile(
            r"Dialogue:\s*(.*?)\s*,(\d:\d{2}:\d{2}\.\d{2}),(\d:\d{2}:\d{2}\.\d{2}),.*?,.*?,.*?,.*?,.*?,.*?,(.*)")
        # 正则匹配 ASS 样式标签，如 {\move(...)}, {\c&H...}, {\a6...} 等
        self.tag_pattern = re.compile(r"\{.*?\}")

    def _time_to_seconds(self, time_str: str) -> float:
        """
        将 ASS 时间格式 (H:MM:SS.CC) 转换为秒数
        """
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def _clean_text(self, text: str) -> str:
        """
        清洗弹幕文本，移除 ASS 标签和换行符
        """
        text = self.tag_pattern.sub('', text)
        text = text.replace('\\N', ' ')  # ASS 换行符替换为空格
        return text.strip()

    def _parse_bilibili_xml(self, file_path: Union[str, Path]) -> List[Dict]:
        """解析 B 站 XML 弹幕（与 ``bilidown.convert_xml_to_ass`` 时间字段一致）。"""
        import xml.etree.ElementTree as ET

        danmaku_list = []
        tree = ET.parse(file_path)
        root = tree.getroot()
        for d in root.findall("d"):
            p_attr = d.get("p")
            if not p_attr:
                continue
            p = p_attr.split(",")
            if len(p) < 1:
                continue
            try:
                start_time = float(p[0])
            except ValueError:
                continue
            duration = float(p[1]) if len(p) > 1 else 2.0
            end_time = start_time + max(duration, 0.05)
            raw = d.text if d.text else ""
            clean_text = raw.replace("\n", " ").strip()
            if clean_text:
                danmaku_list.append({"start": start_time, "end": end_time, "text": clean_text})
        return danmaku_list

    def parse(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        解析弹幕文件，返回统一结构的弹幕列表。
        返回格式: [{'start': float, 'end': float, 'text': str}, ...]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"弹幕文件不存在: {file_path}")

        suffix = Path(file_path).suffix.lower()
        if suffix == ".xml":
            return self._parse_bilibili_xml(file_path)

        danmaku_list = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Dialogue:"):
                    match = self.dialogue_pattern.match(line)
                    if match:
                        start_time = self._time_to_seconds(match.group(2))
                        end_time = self._time_to_seconds(match.group(3))
                        raw_text = match.group(4)

                        clean_text = self._clean_text(raw_text)

                        if clean_text:
                            danmaku_list.append({
                                'start': start_time,
                                'end': end_time,
                                'text': clean_text
                            })
        return danmaku_list

# ==========================================
# 3. 细粒度对齐逻辑
# ==========================================

class FineGrainedAligner:
    """
    将视频帧时间戳与弹幕时间轴进行对齐
    支持二维时间戳输入 (num_clips, frames_per_clip)
    """

    def __init__(self, danmaku_list: List[Dict]):
        self.danmaku_list = danmaku_list
        # 按照开始时间排序，便于优化查询
        self.danmaku_list.sort(key=lambda x: x['start'])

    def align(self, timestamps: np.ndarray) -> List[str]:
        """
        对齐函数：将弹幕文本聚合到片段级别

        参数:
            timestamps: 帧时间戳数组，形状为 (num_clips, frames_per_clip)

        返回:
            List[str]: 长度为 num_clips 的列表，每个元素是该片段内所有弹幕拼接后的文本
        """
        # 如果是一维数组，升维以便统一处理
        if timestamps.ndim == 1:
            timestamps = timestamps.reshape(1, -1)

        num_clips = timestamps.shape[0]
        aligned_text_list = []

        # 遍历每个片段
        for clip_idx in range(num_clips):
            clip_timestamps = timestamps[clip_idx]

            # 获取当前片段的时间范围 (最小值到最大值)
            # 为了容错，稍微扩展一下范围
            t_min = clip_timestamps.min()
            t_max = clip_timestamps.max()

            clip_danmaku_set = set()  # 使用集合去重

            # 遍历弹幕列表查找匹配
            # 优化：因为已排序，如果弹幕开始时间已经超过片段结束时间，可以停止内层循环
            for d in self.danmaku_list:
                # 优化截止条件：如果弹幕开始时间已经晚于当前片段最大时间，后续弹幕肯定不匹配
                if d['start'] > t_max:
                    break

                # 匹配条件：弹幕的时间区间 [start, end] 与 片段时间区间 [t_min, t_max] 有交集
                # 交集条件：d['start'] <= t_max and d['end'] >= t_min
                if d['start'] <= t_max and d['end'] >= t_min:
                    clip_danmaku_set.add(d['text'])

            # 将该片段内的弹幕拼接成一个字符串
            # 注意：如果弹幕为空，返回空字符串
            merged_text = " ".join(list(clip_danmaku_set))
            aligned_text_list.append(merged_text)

        return aligned_text_list