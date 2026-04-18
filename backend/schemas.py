"""
Pydantic 请求/响应模型：与 ``backend.models`` ORM 对应，供 FastAPI 序列化与校验。
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str
    bilibili_mid: str

class UserResponse(UserBase):
    id: int
    bilibili_mid: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class VideoBase(BaseModel):
    filename: str

class VideoCreate(VideoBase):
    pass

class VideoResponse(VideoBase):
    id: int
    status: str
    file_path: Optional[str] = None
    danmu_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    progress: float = 0.0
    progress_stage: Optional[str] = None
    source_url: Optional[str] = None
    video_score: List[Any] = Field(default_factory=list)
    clip_scores: List[Any] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    score_change: Optional[float] = None
    created_at: datetime
    user_id: Optional[int] = None

    @field_validator("video_score", "clip_scores", "suggestions", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: Any) -> Any:
        return [] if v is None else v

    class Config:
        from_attributes = True


class SensitiveWordCreate(BaseModel):
    word: str
    category: str = ""
    is_regex: bool = False
    is_whitelist: bool = False
    action: str = "block"


class SensitiveWordRuleResponse(BaseModel):
    id: int
    word: str
    category: str = ""
    is_regex: bool = False
    is_whitelist: bool = False
    action: str = "block"
    hit_count: int = 0

    class Config:
        from_attributes = True


class SensitiveWordRequestResponse(BaseModel):
    id: int
    word: str
    category: str = ""
    is_regex: bool = False
    is_whitelist: bool = False
    action: str = "block"
    user_id: int
    requester_username: str = ""
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime] = None


class VideoListResponse(BaseModel):
    """分页列表：``GET /api/videos`` 响应体；与前端 ``getVideoList`` / 无限滚动约定一致。"""
    items: List[VideoResponse]
    total: int
    skip: int
    limit: int


class AdminBilibiliMidsResponse(BaseModel):
    """管理员 mid：环境变量、库内保存、合并结果（均为字符串，与 ``User.bilibili_mid`` 一致）。"""

    env_mids: List[str]
    stored_mids: str
    effective_mids: List[str]


class AdminBilibiliMidsUpdate(BaseModel):
    stored_mids: str = ""
