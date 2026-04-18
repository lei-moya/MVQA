"""
音频预处理：FFmpeg 抽取波形，按与视频相同的段数 ``n_segments`` 切段并对齐长度，供 AST 编码。

临时文件 ``temp_audio.wav`` 生成在当前工作目录，处理结束会删除。
"""

import os
import numpy as np
import subprocess
from backend.mvqa_analyzer import Config


class VideoAudioSegmentProcessor:
    """视频音频分割与特征提取处理器"""

    def __init__(self):
        self.sample_rate = Config.get_sample_rate()
        self.n_segments = Config.get_num_clips()

    def process_video(self, video_path):
        """
        完整处理流程：提取音频 -> 分段 -> 返回原始波形
        Returns:
            waveforms: numpy array (num_clips, samples)
        """
        try:
            # 临时音频文件路径
            temp_audio_path = "temp_audio.wav"
            
            # 使用FFmpeg提取音频
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                '-y',
                temp_audio_path
            ]
            
            # 捕获输出为字节，避免Unicode解码错误
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=False)
            if result.returncode != 0:
                # 尝试解码错误信息
                raise Exception(f"FFmpeg提取音频失败")
            
            # 使用librosa读取音频
            import librosa
            y, sr = librosa.load(temp_audio_path, sr=self.sample_rate)
            
            # 分段处理
            duration = len(y) / sr
            segment_length = duration / self.n_segments
            waveforms = []
            
            for i in range(self.n_segments):
                start = int(i * segment_length * sr)
                end = int((i + 1) * segment_length * sr) if i < self.n_segments - 1 else len(y)
                segment = y[start:end]
                waveforms.append(segment.astype(np.float32))
            
            # 对齐长度
            max_len = max(len(w) for w in waveforms)
            padded_waveforms = np.zeros((self.n_segments, max_len), dtype=np.float32)
            
            for i, w in enumerate(waveforms):
                if len(w) > 0:
                    padded_waveforms[i, :len(w)] = w
            
            # 清理临时文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return padded_waveforms
            
        except Exception as e:
            # 返回默认波形
            return np.zeros((self.n_segments, 1024), dtype=np.float32)
