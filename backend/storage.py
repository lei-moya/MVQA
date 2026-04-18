"""
存储抽象：本地路径与可选 ``s3://`` 对象键；流播放时可返回预签名 URL 重定向。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
from urllib.parse import urlparse


def is_s3_uri(path: Optional[str]) -> bool:
    return bool(path) and str(path).startswith("s3://")


def presigned_video_get_url(s3_uri: str, expires_in: int = 3600) -> Optional[str]:
    """若配置 ``AWS_ACCESS_KEY_ID`` / 区域 / Bucket，则返回 GET 预签名 URL。"""
    if not is_s3_uri(s3_uri):
        return None
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        return None
    try:
        import boto3  # type: ignore
    except ImportError:
        return None
    client = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def stream_redirect_or_local(file_path: str) -> Tuple[Optional[str], str]:
    """
    Returns:
        (redirect_url, local_path): ``redirect_url`` 非空时应对浏览器返回 307；
        否则用 ``local_path`` 走本地 FileResponse。
    """
    if is_s3_uri(file_path):
        url = presigned_video_get_url(file_path)
        if url:
            return url, ""
        return None, ""
    return None, file_path
