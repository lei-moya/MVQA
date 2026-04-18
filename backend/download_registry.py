"""
视频链接下载调度：按域名选择解析器，默认走 B 站 ``bilidown.download_video``。
扩展时在 ``REGISTRY`` 中注册 ``host_suffix -> callable(url, out_dir, **kwargs)``。
"""

from __future__ import annotations

from typing import Any, Callable, Dict
from urllib.parse import urlparse

from backend.bilidown import download_video as bili_download_video

Downloader = Callable[..., Any]

REGISTRY: Dict[str, Downloader] = {
    "bilibili.com": bili_download_video,
    "b23.tv": bili_download_video,
}


def _hosts_for(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host


def dispatch_download(url: str, output_dir: str, **kwargs) -> None:
    host = _hosts_for(url)
    for suffix, fn in REGISTRY.items():
        if host == suffix or host.endswith("." + suffix):
            fn(url, output_dir, **kwargs)
            return
    # 默认尝试 B 站解析（历史行为：含 BV 的 URL 由 bilidown 内部抛错）
    bili_download_video(url, output_dir, **kwargs)
