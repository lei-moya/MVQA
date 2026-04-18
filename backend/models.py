"""
SQLAlchemy ORM 模型：用户（B 站扫码登录）、视频任务、按用户划分的设置、全局敏感词表。
"""

from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class VideoStatus(str, enum.Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"  # 链接下载落盘完成，待分析（可单独重试分析）
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    bilibili_mid = Column(String, unique=True, index=True, nullable=False)  # B站用户ID
    username = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    sessdata = Column(String, nullable=False)  # B站登录凭证
    level = Column(Integer, nullable=True)  # 等级
    role = Column(String, default="user", nullable=False)  # user | admin
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 关联关系
    videos = relationship("Video", back_populates="user")
    settings = relationship("Setting", back_populates="user")

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)  # 本地路径或 s3://bucket/key
    danmu_path = Column(String, nullable=True)  # 弹幕文件路径
    thumbnail_path = Column(String, nullable=True)  # 封面 jpeg 路径（uploads 下）
    progress = Column(Float, default=0.0)  # 0–100
    progress_stage = Column(String, nullable=True)  # 如 decode / vision / fusion
    source_url = Column(String, nullable=True)  # 链接下载时的原始 URL
    status = Column(String, default=VideoStatus.PENDING.value)
    # 评分指标
    video_score = Column(JSON, nullable=True)  # 视频级预测整个数据
    clip_scores = Column(JSON, nullable=True)  # 片段级预测整个数据
    suggestions = Column(JSON, nullable=True)  # 改进建议列表
    score_change = Column(Float, nullable=True)  # 视频级评分涨幅
    created_at = Column(DateTime, default=datetime.now)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 关联关系
    user = relationship("User", back_populates="videos")

class SystemKV(Base):
    """全局键值（与 ``Setting`` 按用户区分不同，如管理员 UID 列表）。"""

    __tablename__ = "system_kv"

    key = Column(String, primary_key=True, index=True)
    value = Column(String, nullable=True)


class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, index=True)  # 配置键
    value = Column(JSON)  # 配置值
    description = Column(String, nullable=True)  # 配置描述
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 关联关系
    user = relationship("User", back_populates="settings")

class SensitiveWord(Base):
    __tablename__ = "sensitive_words"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, unique=True, index=True)  # 敏感词或正则模式
    category = Column(String, default="", nullable=False)
    is_regex = Column(Boolean, default=False, nullable=False)
    is_whitelist = Column(Boolean, default=False, nullable=False)
    action = Column(String, default="block", nullable=False)  # block | warn
    hit_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class SensitiveWordRequest(Base):
    """普通用户提交的敏感词添加申请，经管理员同意后写入 ``SensitiveWord``。"""

    __tablename__ = "sensitive_word_requests"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, index=True, nullable=False)
    category = Column(String, default="", nullable=False)
    is_regex = Column(Boolean, default=False, nullable=False)
    is_whitelist = Column(Boolean, default=False, nullable=False)
    action = Column(String, default="block", nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="pending", index=True)  # pending | approved | rejected
    reviewed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
