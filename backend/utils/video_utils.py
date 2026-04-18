"""
视频采样：按配置将视频均分为 ``num_clips`` 段，每段采 ``frames_per_clip`` 帧，供 ViT 视觉编码。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Any
from backend.mvqa_analyzer import Config

class VideoFrameExtractor:
    """
    视频帧提取器：将视频均分为指定数量的片段，并在每个片段内提取指定数量的帧
    输出形状: (num_clips, frames_per_clip, H, W, 3)
    """

    def __init__(self):
        """从 ``mvqa_analyzer.Config`` 读取 ``num_clips`` / ``frames_per_clip`` / ``target_size``。"""
        self.num_clips = Config.get_num_clips()
        self.frames_per_clip = Config.get_frames_per_clip()
        self.target_size = Config.get_target_size()

    def _calculate_sample_indices(self, total_frames: int) -> np.ndarray:
        """
        计算双层采样的帧索引

        返回:
            indices: 形状为 (num_clips, frames_per_clip) 的索引数组
        """
        # 如果视频总帧数不足以分割，则退化为全局均匀采样
        if total_frames <= self.num_clips:
            # 这种情况下无法按片段分割，直接重复采样
            indices = np.linspace(0, total_frames - 1, self.num_clips * self.frames_per_clip, dtype=int)
            return indices.reshape(self.num_clips, self.frames_per_clip)

        # 1. 计算每个片段的边界帧索引
        # 将视频看作连续流，计算每个片段的起止位置
        clip_boundaries = np.linspace(0, total_frames, self.num_clips + 1, dtype=int)

        all_indices = []

        for i in range(self.num_clips):
            start_f = clip_boundaries[i]
            end_f = clip_boundaries[i + 1]

            # 当前片段的实际帧数
            clip_len = end_f - start_f

            if clip_len == 0:
                # 如果片段长度为0（极少见），填充前一帧
                segment_indices = np.full(self.frames_per_clip, start_f - 1 if start_f > 0 else 0)
            else:
                # 2. 在当前片段内部均匀采样 frames_per_clip 帧
                # endpoint=False 避免包含下一个片段的起始帧
                segment_indices = np.linspace(start_f, end_f - 1, self.frames_per_clip, dtype=int)

            all_indices.append(segment_indices)

        return np.array(all_indices)

    def extract_frames_with_timestamps(self, video_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取视频帧及对应的时间戳

        返回:
            frames: (num_clips, frames_per_clip, H, W, 3) float32 numpy array
            timestamps: (num_clips, frames_per_clip) float32 numpy array
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0 or fps == 0:
                raise RuntimeError("视频信息无效")

            # 计算采样索引 (N_clips, N_frames)
            sample_indices = self._calculate_sample_indices(total_frames)

            # 预分配内存，避免动态扩容
            frames_array = np.zeros((self.num_clips, self.frames_per_clip, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            timestamps_array = np.zeros((self.num_clips, self.frames_per_clip), dtype=np.float32)

            # 优化：为了避免频繁的 cap.set 跳转开销，我们可以逐个片段处理
            # 但为了保证逻辑清晰，这里按片段循环
            for clip_idx in range(self.num_clips):
                for frame_idx_in_clip, frame_idx in enumerate(sample_indices[clip_idx]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        # 如果读取失败，使用零帧
                        frame = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                    else:
                        # 图像处理
                        frame = cv2.resize(frame, self.target_size)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 归一化
                        frame = frame.astype(np.float32) / 255.0

                    # 计算时间戳
                    current_time = frame_idx / fps

                    # 直接存储到预分配的数组中
                    frames_array[clip_idx, frame_idx_in_clip] = frame
                    timestamps_array[clip_idx, frame_idx_in_clip] = current_time

            return frames_array, timestamps_array

        finally:
            cap.release()


