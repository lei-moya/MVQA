"""
应用级常量：与上传校验、流式传输、分页等相关。

分页默认值须与 ``frontend/src/api/index.js`` 中 ``VIDEO_LIST_PAGE_SIZE`` 保持一致。
"""

from typing import FrozenSet

# 依赖 OpenCV / MoviePy 解码；若某环境下无法打开，请转码为 mp4 后再传
ALLOWED_VIDEO_EXTENSIONS: FrozenSet[str] = frozenset(
    {".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"}
)
# XML 按 B 站弹幕（``d`` 节点 ``p`` 属性）解析，见 ``utils/text_utils.ASSParser``
ALLOWED_DANMU_EXTENSIONS: FrozenSet[str] = frozenset({".ass", ".xml"})

VIDEO_STREAM_CHUNK_SIZE = 8192

VIDEO_LIST_DEFAULT_LIMIT = 30

# 单次批量本地上传最大文件数
BATCH_UPLOAD_MAX_FILES = 10
