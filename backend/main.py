from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form, Request, Depends, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from sqlalchemy.orm import Session
import shutil
import os
import uuid
import mimetypes
from datetime import datetime
import jwt
from datetime import datetime, timedelta

from backend.database import SessionLocal, engine
from backend.models import Base, Video, VideoStatus, User
from backend.schemas import VideoResponse, UserCreate, UserResponse
from backend.services import analyze_video_quality
from backend.bilidown import download_video
from backend.config import config_manager, CONFIG, ConfigManager
from backend.bililogin import BiliClient, generate_qr_code


# 创建数据库表
Base.metadata.create_all(bind=engine)

# 初始化敏感词数据
def init_sensitive_words():
    from backend.database import SessionLocal
    from backend.models import SensitiveWord
    from backend.config import DEFAULT_CONFIG
    
    db = SessionLocal()
    try:
        # 检查是否已有敏感词
        existing_words = db.query(SensitiveWord).count()
        if existing_words == 0:
            # 添加默认敏感词
            for word in DEFAULT_CONFIG.get('sensitive_words', []):
                if word:
                    sensitive_word = SensitiveWord(word=word)
                    db.add(sensitive_word)
            db.commit()
            print("敏感词初始化成功")
    except Exception as e:
        print(f"初始化敏感词失败: {e}")
        db.rollback()
    finally:
        db.close()

# 初始化敏感词
init_sensitive_words()

app = FastAPI(title="短视频质量评级系统")

# 配置
SECRET_KEY = "your-secret-key-for-jwt-token-generation-2026"  # 生产环境中应使用环境变量
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 认证相关函数
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

# 认证依赖
def get_current_user(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    
    token = token.replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    
    return user_id

# 可选认证依赖
def get_current_user_optional(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        return None
    
    token = token.replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    return user_id

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置上传文件存储目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# 依赖注入：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 工具函数
def generate_unique_filename(original_filename):
    """生成唯一的文件名"""
    # 清理文件名，移除路径信息和特殊字符
    safe_filename = os.path.basename(original_filename)
    # 移除可能的恶意字符
    safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in '._-')
    ext = os.path.splitext(safe_filename)[1]
    return f"{uuid.uuid4()}{ext}"


def safe_remove_file(file_path, max_attempts=3):
    """安全删除文件，处理文件占用的情况"""
    if not file_path:
        return
    
    # 验证文件路径是否在允许的目录内
    if not is_safe_path(file_path):
        print(f"不安全的文件路径: {file_path}")
        return
    
    if not os.path.exists(file_path):
        return
    
    import time
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            print(f"文件删除成功: {file_path}")
            return
        except Exception as e:
            print(f"删除文件失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
            time.sleep(0.5)


def is_safe_path(file_path):
    """验证文件路径是否安全，防止路径遍历攻击"""
    try:
        # 规范化路径
        normalized_path = os.path.normpath(file_path)
        # 获取uploads目录的绝对路径
        uploads_abs_path = os.path.abspath(UPLOAD_DIR)
        # 获取文件的绝对路径
        file_abs_path = os.path.abspath(normalized_path)
        # 检查文件是否在uploads目录内
        return file_abs_path.startswith(uploads_abs_path)
    except Exception:
        return False


def validate_file_extension(filename, allowed_extensions):
    """验证文件扩展名是否允许"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions


# 后台任务：处理视频分析
def process_video_task(video_id: int, file_path: str, danmu_path: str = "", old_score: float = None):
    """后台处理视频分析任务"""
    from backend.database import SessionLocal
    
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return
        
        video.status = VideoStatus.PROCESSING
        db.commit()

        results = analyze_video_quality(file_path, danmu_path)

        score_change = None
        if old_score is not None and results['video_score'] and len(results['video_score']) > 10:
            new_score = results['video_score'][10]
            score_change = new_score - old_score

        video.status = VideoStatus.COMPLETED
        video.video_score = results['video_score']
        video.clip_scores = results.get('clip_scores', [])
        video.suggestions = results['suggestions']
        video.score_change = score_change
        db.commit()

        print(f"Video {video_id} processed successfully.")
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = VideoStatus.FAILED
            db.commit()
    finally:
        db.close()


def process_video_file(db: Session, video_filename: str, video_path: str, danmu_path: str, background_tasks: BackgroundTasks, user_id: int = None):
    """处理视频文件和数据库记录"""
    # 只查找当前用户的视频
    existing_video = db.query(Video).filter(Video.filename == video_filename, Video.user_id == user_id).first()
    
    if existing_video:
        old_score = None
        if existing_video.video_score and len(existing_video.video_score) > 10:
            old_score = existing_video.video_score[10]
        
        # 删除原文件
        safe_remove_file(existing_video.file_path)
        if existing_video.danmu_path:
            safe_remove_file(existing_video.danmu_path)
        
        # 更新数据库记录
        existing_video.file_path = video_path
        existing_video.danmu_path = danmu_path
        existing_video.status = VideoStatus.PENDING
        existing_video.video_score = None
        existing_video.clip_scores = []
        existing_video.suggestions = None
        existing_video.score_change = None
        existing_video.created_at = datetime.now()
        existing_video.user_id = user_id  # 设置用户ID
        db.commit()
        db.refresh(existing_video)
        
        background_tasks.add_task(process_video_task, existing_video.id, video_path, danmu_path, old_score)
        return existing_video
    else:
        # 创建新记录
        db_video = Video(
            filename=video_filename,
            file_path=video_path,
            danmu_path=danmu_path,
            status=VideoStatus.PENDING,
            user_id=user_id  # 设置用户ID
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)

        background_tasks.add_task(process_video_task, db_video.id, video_path, danmu_path)
        return db_video


# 路由处理函数
@app.post("/api/videos/upload/url", response_model=VideoResponse)
async def upload_video_by_url(
        background_tasks: BackgroundTasks,
        url: str = Form(...),
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
):
    """通过URL上传视频"""
    try:
        temp_dir = f"temp_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 获取当前用户的sessdata
        user = db.query(User).filter(User.id == int(current_user)).first()
        sessdata = user.sessdata if user else None
        
        print(f"开始下载视频: {url}")
        # 从配置中获取下载设置
        download_settings = CONFIG.get('download_settings', {})
        video_quality = download_settings.get('video_quality', 'auto')
        default_video_quality = download_settings.get('default_video_quality', '80')
        audio_quality = download_settings.get('audio_quality', 'high')
        default_audio_quality = download_settings.get('default_audio_quality', 'high')
        
        download_video(url, temp_dir, quality=video_quality, default_video_quality=default_video_quality, audio_quality=audio_quality, default_audio_quality=default_audio_quality, session_token=sessdata)
        print("视频下载完成")
        
        video_file = None
        danmu_file = None
        for file in os.listdir(temp_dir):
            if file.endswith('.mp4'):
                video_file = os.path.join(temp_dir, file)
            elif file.endswith('.ass'):
                danmu_file = os.path.join(temp_dir, file)
            elif file.endswith('.xml'):
                danmu_file = os.path.join(temp_dir, file)
        
        if not video_file:
            raise HTTPException(status_code=400, detail="视频下载失败")
        
        video_filename = os.path.basename(video_file)
        unique_video_filename = generate_unique_filename(video_filename)
        video_path = os.path.join(UPLOAD_DIR, unique_video_filename)
        shutil.move(video_file, video_path)
        
        danmu_path = ""
        if danmu_file:
            unique_danmu_filename = generate_unique_filename(os.path.basename(danmu_file))
            danmu_path = os.path.join(UPLOAD_DIR, unique_danmu_filename)
            shutil.move(danmu_file, danmu_path)
        
        shutil.rmtree(temp_dir)
        
        return process_video_file(db, video_filename, video_path, danmu_path, background_tasks, int(current_user))
    except Exception as e:
        print(f"URL上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"URL上传失败: {str(e)}")


@app.post("/api/videos/upload", response_model=VideoResponse)
async def upload_video(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        danmu: UploadFile = File(None),
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
):
    """上传本地视频文件"""
    try:
        # 验证视频文件扩展名
        allowed_video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']
        if not validate_file_extension(video.filename, allowed_video_extensions):
            raise HTTPException(status_code=400, detail="不支持的视频文件格式")

        unique_video_filename = generate_unique_filename(video.filename)
        video_path = os.path.join(UPLOAD_DIR, unique_video_filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        danmu_path = ""
        if danmu:
            # 验证弹幕文件扩展名
            allowed_danmu_extensions = ['.ass', '.xml']
            if not validate_file_extension(danmu.filename, allowed_danmu_extensions):
                raise HTTPException(status_code=400, detail="不支持的弹幕文件格式")
            
            unique_danmu_filename = generate_unique_filename(danmu.filename)
            danmu_path = os.path.join(UPLOAD_DIR, unique_danmu_filename)

            with open(danmu_path, "wb") as buffer:
                shutil.copyfileobj(danmu.file, buffer)

        return process_video_file(db, video.filename, video_path, danmu_path, background_tasks, int(current_user))
    except HTTPException:
        raise
    except Exception as e:
        print(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.get("/api/videos/{video_id}", response_model=VideoResponse)
def get_video_detail(video_id: int, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """获取视频详情"""
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == int(current_user)).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.get("/api/videos", response_model=list[VideoResponse])
def list_videos(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """获取视频列表"""
    videos = db.query(Video).filter(Video.user_id == int(current_user)).order_by(Video.created_at.desc()).offset(skip).limit(limit).all()
    return videos


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """删除视频"""
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == int(current_user)).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video.file_path
    danmu_path = video.danmu_path
    
    db.delete(video)
    db.commit()
    
    # 直接删除文件
    safe_remove_file(video_path)
    if danmu_path:
        safe_remove_file(danmu_path)
    
    return {"message": "Video deleted successfully"}


async def async_safe_remove(video_path: str, danmu_path: str):
    """异步删除文件"""
    safe_remove_file(video_path)
    if danmu_path:
        safe_remove_file(danmu_path)


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: int, request: Request, db: Session = Depends(get_db)):
    """流式播放视频，支持断点续传"""
    # 从URL参数中获取token
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    
    # 验证token
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == int(user_id)).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    file_path = video.file_path
    file_size = os.path.getsize(file_path)
    
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = "video/mp4"
    
    range_header = request.headers.get("Range")
    if range_header:
        range_str = range_header.replace("bytes=", "")
        start, end = range_str.split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        
        if start >= file_size or end >= file_size:
            raise HTTPException(status_code=416, detail="Range Not Satisfiable")
        
        content_length = end - start + 1
        
        async def file_iterator():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 8192
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
            "Accept-Ranges": "bytes",
            "Content-Type": content_type
        }
        
        return StreamingResponse(
            file_iterator(),
            status_code=206,
            headers=headers
        )
    else:
        return FileResponse(
            file_path,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"}
        )


@app.get("/api/config")
def get_config(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """获取当前配置"""
    # 使用数据库会话重新加载配置
    config_manager = ConfigManager(db)
    return config_manager.get_config(int(current_user))


@app.put("/api/config")
def update_config(config: dict, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """更新配置"""
    try:
        # 使用数据库会话更新配置
        config_manager = ConfigManager(db)
        config_manager.update_config(config, db, int(current_user))
        return {"message": "配置更新成功", "config": config_manager.get_config(int(current_user))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


# 敏感词相关API接口
@app.get("/api/sensitive-words")
def get_sensitive_words(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """获取敏感词列表"""
    try:
        from backend.models import SensitiveWord
        words = db.query(SensitiveWord.word).all()
        return {"sensitive_words": [word[0] for word in words]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取敏感词失败: {str(e)}")


@app.post("/api/sensitive-words")
def add_sensitive_word(word: str = Body(..., embed=True), db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """添加敏感词"""
    try:
        from backend.models import SensitiveWord
        
        if not word:
            raise HTTPException(status_code=400, detail="敏感词不能为空")
        
        # 检查敏感词是否已存在
        existing_word = db.query(SensitiveWord).filter(SensitiveWord.word == word).first()
        if existing_word:
            raise HTTPException(status_code=400, detail="敏感词已存在")
        
        new_word = SensitiveWord(word=word)
        db.add(new_word)
        db.commit()
        db.refresh(new_word)
        
        return {"message": "敏感词添加成功", "word": new_word.word}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加敏感词失败: {str(e)}")


@app.delete("/api/sensitive-words/{word}")
def delete_sensitive_word(word: str, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """删除敏感词"""
    try:
        from backend.models import SensitiveWord
        sensitive_word = db.query(SensitiveWord).filter(SensitiveWord.word == word).first()
        if not sensitive_word:
            raise HTTPException(status_code=404, detail="敏感词不存在")
        
        db.delete(sensitive_word)
        db.commit()
        
        return {"message": "敏感词删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除敏感词失败: {str(e)}")


from fastapi import Body

@app.put("/api/sensitive-words")
def update_sensitive_words(words: list = Body(...), db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """批量更新敏感词"""
    try:
        from backend.models import SensitiveWord
        # 先删除所有现有敏感词
        db.query(SensitiveWord).delete()
        # 添加新的敏感词
        for word in words:
            if word:
                new_word = SensitiveWord(word=word)
                db.add(new_word)
        db.commit()
        
        return {"message": "敏感词更新成功", "sensitive_words": words}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新敏感词失败: {str(e)}")


# 登录相关路由
@app.get("/api/auth/qr-code")
def get_login_qr_code():
    """获取登录二维码"""
    client = BiliClient()
    qr_info = client.get_qr_info()
    qr_url = qr_info["url"]
    qr_key = qr_info["qrcode_key"]
    img_base64 = generate_qr_code(qr_url)
    
    return {"qr_key": qr_key, "qr_code": img_base64, "qr_url": qr_url}

@app.get("/api/auth/qr-status/{qr_key}")
def check_qr_status(qr_key: str, db: Session = Depends(get_db)):
    """检查二维码状态"""
    try:
        client = BiliClient()
        qr_status, sessdata = client.get_qr_status(qr_key)
        
        if qr_status["code"] == 0:
            # 登录成功，创建或更新用户
            if not sessdata:
                # 如果没有获取到SESSDATA，返回错误信息
                return {"status": "error", "code": qr_status["code"], "message": "无法获取登录凭证，请重新登录"}
            
            bili_client = BiliClient(sessdata)
            user_info_response = bili_client.simple_get("https://api.bilibili.com/x/space/myinfo")
            user_info = user_info_response.json()
            
            if user_info["code"] == 0:
                data = user_info["data"]
                username = data.get("name", "")
                bilibili_mid = str(data.get('mid', ''))
                email = f"{bilibili_mid}@bilibili.com"
                
                # 查找现有用户（使用B站用户ID作为唯一标识）
                existing_user = db.query(User).filter(User.bilibili_mid == bilibili_mid).first()
                
                if existing_user:
                    # 更新用户
                    existing_user.username = username  # 更新用户名（用户可能改名）
                    existing_user.email = email
                    existing_user.sessdata = sessdata  # 更新sessdata
                    # 更新B站用户信息
                    existing_user.level = data.get("level")
                    existing_user.updated_at = datetime.now()
                    db.commit()
                    user_id = existing_user.id
                else:
                    # 创建新用户
                    new_user = User(
                        bilibili_mid=bilibili_mid,
                        username=username,
                        email=email,
                        sessdata=sessdata,  # 存储sessdata
                        # 存储B站用户信息
                        level=data.get("level")
                    )
                    db.add(new_user)
                    db.commit()
                    db.refresh(new_user)
                    user_id = new_user.id
                    
                    # 为新用户创建设置数据
                    from backend.models import Setting
                    
                    # 视频处理设置
                    video_processing_setting = Setting(
                        key="video_processing",
                        value={
                            "num_clips": 5,
                            "frames_per_clip": 5,
                            "target_size": [224, 224]
                        },
                        description="视频处理设置",
                        user_id=user_id
                    )
                    db.add(video_processing_setting)
                    
                    # 音频处理设置
                    audio_processing_setting = Setting(
                        key="audio_processing",
                        value={
                            "sample_rate": 16000
                        },
                        description="音频处理设置",
                        user_id=user_id
                    )
                    db.add(audio_processing_setting)
                    
                    # 下载设置
                    download_settings_setting = Setting(
                        key="download_settings",
                        value={
                            "video_quality": "auto",
                            "default_video_quality": "80",
                            "audio_quality": "high",
                            "default_audio_quality": "high",
                            "download_danmaku": True
                        },
                        description="下载设置",
                        user_id=user_id
                    )
                    db.add(download_settings_setting)
                    
                    db.commit()
                
                # 生成访问令牌
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                access_token = create_access_token(
                    data={"sub": str(user_id)},
                    expires_delta=access_token_expires
                )
                
                # 获取用户对象以获取更多字段
                user_obj = db.query(User).filter(User.id == user_id).first()
                
                return {
                    "status": "success",
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "user": {
                        "id": user_id,
                        "username": username,
                        "email": email,
                        "bilibili_mid": bilibili_mid,
                        "level": user_obj.level,
                        "created_at": user_obj.created_at,
                        "updated_at": user_obj.updated_at
                    }
                }
            else:
                # 获取用户信息失败
                return {"status": "error", "code": user_info["code"], "message": user_info.get("message", "获取用户信息失败")}
        elif qr_status["code"] == -1:
            # 网络错误
            return {"status": "error", "code": qr_status["code"], "message": qr_status.get("message", "网络错误，请检查网络连接后重试")}
        
        return {"status": "pending", "code": qr_status["code"], "message": qr_status.get("message", "")}
    except Exception as e:
        # 捕获其他异常
        return {"status": "error", "code": -1, "message": f"登录失败: {str(e)}"}

@app.get("/api/auth/me")
def get_current_user_info(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取当前用户信息"""
    user = db.query(User).filter(User.id == int(current_user)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "bilibili_mid": user.bilibili_mid,
        "level": user.level,
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }

@app.post("/api/auth/logout")
def logout(current_user: str = Depends(get_current_user)):
    """登出"""
    # JWT是无状态的，客户端删除令牌即可
    return {"message": "登出成功"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
