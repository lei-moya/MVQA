from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class VideoStatus(str, enum.Enum):
    PENDING = "pending"
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
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 关联关系
    videos = relationship("Video", back_populates="user")
    settings = relationship("Setting", back_populates="user")

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)  # 存储路径
    danmu_path = Column(String, nullable=True)  # 弹幕文件路径
    status = Column(String, default=VideoStatus.PENDING)
    # 评分指标
    video_score = Column(JSON, nullable=True)  # 视频级预测整个数据
    clip_scores = Column(JSON, nullable=True)  # 片段级预测整个数据
    suggestions = Column(JSON, nullable=True)  # 改进建议列表
    score_change = Column(Float, nullable=True)  # 视频级评分涨幅
    created_at = Column(DateTime, default=datetime.now)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 关联关系
    user = relationship("User", back_populates="videos")

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
    word = Column(String, unique=True, index=True)  # 敏感词
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
