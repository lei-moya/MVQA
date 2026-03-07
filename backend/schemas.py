from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr

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
    video_score: Optional[List] = []
    clip_scores: Optional[List] = []
    suggestions: Optional[List[str]] = []
    score_change: Optional[float] = None
    created_at: datetime
    user_id: Optional[int] = None

    class Config:
        from_attributes = True
