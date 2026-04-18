"""从视频文件抽取首帧缩略图（OpenCV）。"""

from __future__ import annotations

import logging
import os
import uuid

_log = logging.getLogger(__name__)


def write_thumbnail_jpeg(video_path: str, upload_dir: str) -> str | None:
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        name = f"thumb_{uuid.uuid4().hex}.jpg"
        out = os.path.join(upload_dir, name)
        cv2.imwrite(out, frame)
        return out
    except Exception as e:
        _log.debug("缩略图失败: %s", e)
        return None
    finally:
        cap.release()
