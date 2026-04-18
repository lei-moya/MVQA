"""通过 FastAPI ``BackgroundTasks`` 调度视频分析任务。"""

from __future__ import annotations

from typing import Any, Optional


def schedule_video_analysis(
    background_tasks: Any,
    video_id: int,
    file_path: str,
    danmu_path: str = "",
    old_score: Optional[float] = None,
) -> None:
    from backend.main import process_video_task

    background_tasks.add_task(process_video_task, video_id, file_path, danmu_path, old_score)
