"""
运行时配置：优先从环境变量读取，便于部署隔离密钥与超时。

- ``JWT_SECRET``：JWT 签名密钥（生产环境务必设置强随机串）。
- ``ACCESS_TOKEN_EXPIRE_MINUTES``：访问令牌有效期（分钟）。
- ``CORS_ORIGINS``：逗号分隔的允许来源；未设置或 ``*`` 时与 ``allow_credentials`` 同用时浏览器可能拒收 Cookie，
  生产环境建议设为前端绝对 URL 列表（如 ``https://app.example.com``）。
"""

import os
from typing import List

_DEFAULT_JWT_SECRET = "your-secret-key-for-jwt-token-generation-2026"

JWT_SECRET = os.environ.get("JWT_SECRET", _DEFAULT_JWT_SECRET)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

_raw_cors = os.environ.get("CORS_ORIGINS", "").strip()
if _raw_cors:
    CORS_ORIGINS: List[str] = [o.strip() for o in _raw_cors.split(",") if o.strip()]
else:
    CORS_ORIGINS = ["*"]

# B 站 mid 列表（逗号分隔），与库内配置（见 ``system_kv`` / 设置页）合并；登录后与 ``User.role`` 同步
_raw_admins = os.environ.get("ADMIN_BILIBILI_MIDS", "").strip()
ADMIN_BILIBILI_MIDS: set[str] = {x.strip() for x in _raw_admins.split(",") if x.strip()}
