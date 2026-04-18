"""
新用户入库时的默认 ``Setting`` 行：与 ``config.DEFAULT_CONFIG`` 对齐；
前端 ``Settings.vue`` 表单初值与重置逻辑也应与同一默认值集一致。
"""

from __future__ import annotations

from backend.config import DEFAULT_CONFIG
from backend.models import Setting


def seed_default_settings_for_user(db, user_id: int) -> None:
    """为新注册用户写入与 ``DEFAULT_CONFIG`` 一致的 EAV 配置行。"""
    rows: list[Setting] = [
        Setting(
            key="video_processing",
            value=dict(DEFAULT_CONFIG["video_processing"]),
            description="视频处理设置",
            user_id=user_id,
        ),
        Setting(
            key="audio_processing",
            value=dict(DEFAULT_CONFIG["audio_processing"]),
            description="音频处理设置",
            user_id=user_id,
        ),
        Setting(
            key="download_settings",
            value=dict(DEFAULT_CONFIG["download_settings"]),
            description="下载设置",
            user_id=user_id,
        ),
        Setting(
            key="model_paths",
            value=dict(DEFAULT_CONFIG["model_paths"]),
            description="模型与权重路径",
            user_id=user_id,
        ),
    ]
    for row in rows:
        db.add(row)
    db.commit()
