"""
FastAPI 应用入口：短视频多模态质量分析 API。

提供视频上传（本地上传 / B 站 URL）、异步分析任务、JWT 认证、
用户配置与敏感词管理、视频流式播放等能力。分析 pipeline 见
``backend.services.analyze_video_quality`` 与 ``backend.mvqa_analyzer``。
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form, Request, Depends, Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
import shutil
import os
import uuid
import mimetypes
import tempfile
import threading
import jwt
import logging
from datetime import datetime, timedelta
from typing import List as TypingList, Optional

from sqlalchemy import func, desc, asc, nullslast

from backend.constants import (
    ALLOWED_DANMU_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    BATCH_UPLOAD_MAX_FILES,
    VIDEO_LIST_DEFAULT_LIMIT,
    VIDEO_STREAM_CHUNK_SIZE,
)
from backend.settings import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ADMIN_BILIBILI_MIDS,
    CORS_ORIGINS,
    JWT_SECRET,
)
from backend.user_defaults import seed_default_settings_for_user

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 日志过滤器：抑制前端热重载等场景下常见的 WebSocket 断开噪声日志。
class WebSocketErrorFilter(logging.Filter):
    def filter(self, record):
        # 过滤掉包含 WebSocket 相关错误的日志
        if 'socket.send()' in str(record.msg) or 'WebSocket' in str(record.msg):
            return False
        return True

# 抑制 transformers / PyTorch 初始化时冗长的「部分权重未初始化」类警告。
class ModelWarningFilter(logging.Filter):
    def filter(self, record):
        # 过滤模型加载和初始化过程中的已知提示性警告
        warning_messages = [
            'generation_config default values have been modified',
            'Some weights of WhisperForConditionalGeneration were not initialized',
            "'torch.load' received a zip file that looks like a TorchScript archive",
            'Some weights of the model checkpoint at',
            'Some weights of ViTModel were not initialized',
            'You should probably TRAIN this model'
        ]
        for msg in warning_messages:
            if msg in str(record.msg):
                return False
        return True

# 为所有处理器添加过滤器
for handler in logging.getLogger().handlers:
    handler.addFilter(WebSocketErrorFilter())
    handler.addFilter(ModelWarningFilter())

# 为uvicorn日志添加过滤器
uvicorn_logger = logging.getLogger('uvicorn')
for handler in uvicorn_logger.handlers:
    handler.addFilter(WebSocketErrorFilter())
    handler.addFilter(ModelWarningFilter())

uvicorn_access_logger = logging.getLogger('uvicorn.access')
for handler in uvicorn_access_logger.handlers:
    handler.addFilter(WebSocketErrorFilter())
    handler.addFilter(ModelWarningFilter())

# 为transformers和torch日志添加过滤器
transformers_logger = logging.getLogger('transformers')
for handler in transformers_logger.handlers:
    handler.addFilter(ModelWarningFilter())

# 为torch日志添加过滤器
torch_logger = logging.getLogger('torch')
for handler in torch_logger.handlers:
    handler.addFilter(ModelWarningFilter())

from backend.database import SessionLocal, engine, ensure_schema
from backend.models import Base, Video, VideoStatus, User, SensitiveWordRequest

_VIDEO_LIST_STATUSES = frozenset(s.value for s in VideoStatus)
_VIDEO_LIST_SORTS = frozenset({"created_desc", "created_asc", "score_desc", "score_asc"})
from backend.schemas import (
    VideoResponse,
    VideoListResponse,
    SensitiveWordRuleResponse,
    SensitiveWordRequestResponse,
    SensitiveWordCreate,
    AdminBilibiliMidsResponse,
    AdminBilibiliMidsUpdate,
)
from backend.admin_mids_store import (
    effective_admin_mids,
    get_stored_admin_mids_raw,
    set_stored_admin_mids_raw,
)
from backend.services import analyze_video_quality
from backend.download_registry import dispatch_download
from backend.config import ConfigManager, config_manager
from backend.bililogin import BiliClient, generate_qr_code
from backend.job_queue import schedule_video_analysis
from backend.storage import stream_redirect_or_local, is_s3_uri
from backend.utils.video_thumb import write_thumbnail_jpeg

# 创建数据库表
Base.metadata.create_all(bind=engine)
ensure_schema()


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
    except Exception as e:
        db.rollback()
        logging.getLogger(__name__).warning("初始化敏感词失败: %s", e, exc_info=True)
    finally:
        db.close()


# 初始化敏感词
init_sensitive_words()

app = FastAPI(title="短视频质量评级系统")

# JWT：密钥与过期时间见 ``backend/settings.py``（可用环境变量覆盖）
ALGORITHM = "HS256"

# MVQA 推理全局互斥：避免多任务并行占满 GPU 显存（单机部署常见场景）
_INFERENCE_LOCK = threading.Lock()

# 认证相关函数
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """未显式处理的异常：统一 JSON；``HTTPException`` 优先走框架处理器，此处防御性透传。"""
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logging.getLogger(__name__).exception(
        "未处理异常 %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"},
    )


def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
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
# 与浏览器规范一致：``allow_origins=[*]`` 时不应再带 credentials。
_cors_allow_credentials = "*" not in CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置上传文件存储目录
# 使用绝对路径，确保无论从哪个目录运行都指向同一个位置
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# 依赖注入：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin(
    current_user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    u = db.query(User).filter(User.id == int(current_user)).first()
    if not u or u.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


def _validate_admin_mids_stored(raw: str) -> None:
    """库内保存的 UID 行：逗号分隔，每段须为数字串。"""
    s = (raw or "").strip()
    if not s:
        return
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            raise HTTPException(status_code=400, detail=f"UID 须为数字（B 站 mid）：{p}")


def _admin_mids_response(db: Session) -> AdminBilibiliMidsResponse:
    stored = get_stored_admin_mids_raw(db)
    eff = effective_admin_mids(db, ADMIN_BILIBILI_MIDS)
    return AdminBilibiliMidsResponse(
        env_mids=sorted(ADMIN_BILIBILI_MIDS),
        stored_mids=stored,
        effective_mids=sorted(eff),
    )


@app.get("/api/admin/bilibili-mids", response_model=AdminBilibiliMidsResponse)
def get_admin_bilibili_mids_endpoint(
    db: Session = Depends(get_db),
    _admin: str = Depends(require_admin),
):
    """管理员：查看环境变量、库内配置及合并后的管理员 UID 列表。"""
    return _admin_mids_response(db)


@app.put("/api/admin/bilibili-mids", response_model=AdminBilibiliMidsResponse)
def put_admin_bilibili_mids_endpoint(
    body: AdminBilibiliMidsUpdate,
    db: Session = Depends(get_db),
    _admin: str = Depends(require_admin),
):
    """管理员：写入库内管理员 UID（与环境变量合并后，在下次登录时同步 ``User.role``）。"""
    _validate_admin_mids_stored(body.stored_mids)
    set_stored_admin_mids_raw(db, body.stored_mids)
    return _admin_mids_response(db)


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
    log = logging.getLogger(__name__)
    if not file_path:
        return

    # 验证文件路径是否在允许的目录内
    if not is_safe_path(file_path):
        return

    if not os.path.exists(file_path):
        return

    import time
    last_err = None
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            return
        except OSError as e:
            last_err = e
            time.sleep(0.5)
    if last_err:
        log.warning("删除文件失败（已重试 %s 次）: %s — %s", max_attempts, file_path, last_err)


def is_safe_path(file_path):
    """验证文件路径是否安全，防止路径遍历攻击（用 commonpath 判定在 uploads 根内）。"""
    try:
        uploads_root = os.path.realpath(os.path.abspath(UPLOAD_DIR))
        candidate = os.path.realpath(os.path.abspath(os.path.normpath(file_path)))
        common = os.path.commonpath([uploads_root, candidate])
        return common == uploads_root
    except (ValueError, OSError):
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

        fp = file_path or video.file_path or ""
        dp = danmu_path if danmu_path is not None else (video.danmu_path or "")

        if is_s3_uri(fp):
            logging.getLogger(__name__).error(
                "video_id=%s 为对象存储路径，当前分析管线仅支持本地文件", video_id
            )
            video.status = VideoStatus.FAILED.value
            video.suggestions = ["当前版本仅支持本地文件分析，请使用本地上传"]
            video.progress = 0.0
            video.progress_stage = None
            db.commit()
            return

        video.status = VideoStatus.PROCESSING.value
        video.progress = 5.0
        video.progress_stage = "prepare"
        db.commit()

        if fp and os.path.exists(fp):
            thumb = write_thumbnail_jpeg(fp, UPLOAD_DIR)
            if thumb:
                video.thumbnail_path = thumb
                db.commit()

        video.progress = 20.0
        video.progress_stage = "inference"
        db.commit()

        with _INFERENCE_LOCK:
            results = analyze_video_quality(fp, dp, user_id=video.user_id)

        video.progress = 100.0
        video.progress_stage = "done"

        if not results.get("ok"):
            video.status = VideoStatus.FAILED.value
            video.video_score = results.get("video_score") or []
            video.clip_scores = results.get("clip_scores") or []
            video.suggestions = results.get("suggestions") or ["分析失败，请重试"]
            db.commit()
        else:
            score_change = None
            vscore = results.get("video_score") or []
            if old_score is not None and len(vscore) > 10:
                new_score = vscore[10]
                score_change = new_score - old_score

            video.status = VideoStatus.COMPLETED.value
            video.video_score = vscore
            video.clip_scores = results.get("clip_scores", [])
            video.suggestions = results.get("suggestions")
            video.score_change = score_change
            db.commit()
    except Exception as e:
        logging.getLogger(__name__).error("视频分析任务失败 video_id=%s: %s", video_id, e, exc_info=True)
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = VideoStatus.FAILED.value
            video.progress = 0.0
            video.progress_stage = None
            db.commit()
    finally:
        db.close()


def process_video_file(
    db: Session,
    video_filename: str,
    video_path: str,
    danmu_path: str,
    background_tasks: BackgroundTasks,
    user_id: int = None,
    initial_status: VideoStatus = VideoStatus.PENDING,
    source_url: Optional[str] = None,
):
    """处理视频文件和数据库记录；链接下载可用 ``DOWNLOADED`` 表示已落盘待分析。"""
    existing_video = db.query(Video).filter(Video.filename == video_filename, Video.user_id == user_id).first()

    if existing_video:
        old_score = None
        if existing_video.video_score and len(existing_video.video_score) > 10:
            old_score = existing_video.video_score[10]

        safe_remove_file(existing_video.file_path)
        if existing_video.danmu_path:
            safe_remove_file(existing_video.danmu_path)
        if existing_video.thumbnail_path:
            safe_remove_file(existing_video.thumbnail_path)

        existing_video.file_path = video_path
        existing_video.danmu_path = danmu_path
        existing_video.status = initial_status.value
        if source_url:
            existing_video.source_url = source_url
        existing_video.video_score = None
        existing_video.clip_scores = []
        existing_video.suggestions = None
        existing_video.score_change = None
        existing_video.thumbnail_path = None
        existing_video.progress = 0.0
        existing_video.progress_stage = None
        existing_video.created_at = datetime.now()
        existing_video.user_id = user_id
        db.commit()
        db.refresh(existing_video)

        schedule_video_analysis(
            background_tasks,
            existing_video.id,
            video_path,
            danmu_path,
            old_score,
        )
        return existing_video

    db_video = Video(
        filename=video_filename,
        file_path=video_path,
        danmu_path=danmu_path,
        status=initial_status.value,
        user_id=user_id,
        source_url=source_url,
        progress=0.0,
        progress_stage=None,
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    schedule_video_analysis(background_tasks, db_video.id, video_path, danmu_path)
    return db_video


# 路由处理函数
@app.post("/api/videos/upload/url", response_model=VideoResponse)
async def upload_video_by_url(
        background_tasks: BackgroundTasks,
        url: str = Form(...),
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
):
    """通过 B 站等链接下载视频并入库，随后触发与本地上传相同的分析流程。"""
    temp_dir = tempfile.mkdtemp(prefix="mvqa_url_")
    try:
        # 获取当前用户的sessdata
        user = db.query(User).filter(User.id == int(current_user)).first()
        sessdata = user.sessdata if user else None

        # 从配置中获取下载设置
        download_settings = config_manager.get_config(int(current_user))
        video_quality = download_settings.get('video_quality', 'auto')
        default_video_quality = download_settings.get('default_video_quality', '80')
        audio_quality = download_settings.get('audio_quality', 'high')
        default_audio_quality = download_settings.get('default_audio_quality', 'high')

        dispatch_download(
            url,
            temp_dir,
            quality=video_quality,
            default_video_quality=default_video_quality,
            audio_quality=audio_quality,
            default_audio_quality=default_audio_quality,
            session_token=sessdata,
        )

        video_file = None
        danmu_file = None
        for file in os.listdir(temp_dir):
            path_joined = os.path.join(temp_dir, file)
            lower = file.lower()
            if any(lower.endswith(ext) for ext in ALLOWED_VIDEO_EXTENSIONS):
                if video_file is None:
                    video_file = path_joined
            elif lower.endswith(".ass") or lower.endswith(".xml"):
                danmu_file = path_joined

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

        return process_video_file(
            db,
            video_filename,
            video_path,
            danmu_path,
            background_tasks,
            int(current_user),
            initial_status=VideoStatus.DOWNLOADED,
            source_url=url,
        )
    except HTTPException:
        raise
    except Exception:
        logging.getLogger(__name__).exception("链接下载或保存视频失败")
        raise HTTPException(status_code=500, detail="视频下载失败，请稍后重试")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
        if not validate_file_extension(video.filename, ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(status_code=400, detail="不支持的视频文件格式")

        unique_video_filename = generate_unique_filename(video.filename)
        video_path = os.path.join(UPLOAD_DIR, unique_video_filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        danmu_path = ""
        if danmu:
            # 验证弹幕文件扩展名
            if not validate_file_extension(danmu.filename, ALLOWED_DANMU_EXTENSIONS):
                raise HTTPException(status_code=400, detail="不支持的弹幕文件格式")

            unique_danmu_filename = generate_unique_filename(danmu.filename)
            danmu_path = os.path.join(UPLOAD_DIR, unique_danmu_filename)

            with open(danmu_path, "wb") as buffer:
                shutil.copyfileobj(danmu.file, buffer)

        return process_video_file(db, video.filename, video_path, danmu_path, background_tasks, int(current_user))
    except HTTPException:
        raise
    except Exception:
        logging.getLogger(__name__).exception("本地视频上传失败")
        raise HTTPException(status_code=500, detail="文件上传失败，请稍后重试")


@app.get("/api/videos/{video_id}", response_model=VideoResponse)
def get_video_detail(video_id: int, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """获取视频详情"""
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == int(current_user)).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.post("/api/videos/{video_id}/reanalyze", response_model=VideoResponse)
def reanalyze_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """使用已有本地文件重新入队分析（需状态非 processing）。"""
    video = db.query(Video).filter(
        Video.id == video_id, Video.user_id == int(current_user)
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status == VideoStatus.PROCESSING.value:
        raise HTTPException(status_code=400, detail="任务进行中，请稍后再试")
    if is_s3_uri(video.file_path):
        raise HTTPException(status_code=400, detail="对象存储视频暂不支持服务端重分析")
    old_score = None
    if video.video_score and len(video.video_score) > 10:
        old_score = video.video_score[10]

    video.status = VideoStatus.PENDING.value
    video.video_score = None
    video.clip_scores = []
    video.suggestions = None
    video.score_change = None
    video.progress = 0.0
    video.progress_stage = None
    db.commit()
    db.refresh(video)

    schedule_video_analysis(
        background_tasks,
        video.id,
        video.file_path,
        video.danmu_path or "",
        old_score,
    )
    return video


@app.post("/api/videos/upload/batch", response_model=TypingList[VideoResponse])
async def upload_video_batch(
    background_tasks: BackgroundTasks,
    videos: TypingList[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """批量本地上传（无共享弹幕；单次数上限见 ``BATCH_UPLOAD_MAX_FILES``）。"""
    if len(videos) > BATCH_UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {BATCH_UPLOAD_MAX_FILES} 个文件",
        )
    out: TypingList[Video] = []
    for vidf in videos:
        if not validate_file_extension(vidf.filename, ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(status_code=400, detail=f"不支持的视频格式: {vidf.filename}")
        unique_video_filename = generate_unique_filename(vidf.filename)
        video_path = os.path.join(UPLOAD_DIR, unique_video_filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(vidf.file, buffer)
        out.append(
            process_video_file(
                db,
                vidf.filename,
                video_path,
                "",
                background_tasks,
                int(current_user),
            )
        )
    return out


@app.get("/api/videos", response_model=VideoListResponse)
def list_videos(
    skip: int = Query(0, ge=0, description="分页偏移"),
    limit: int = Query(
        VIDEO_LIST_DEFAULT_LIMIT,
        ge=1,
        le=1000,
        description="每页条数，默认与 VIDEO_LIST_DEFAULT_LIMIT 一致",
    ),
    status: Optional[str] = Query(None, description="按状态筛选：pending/downloaded/processing/completed/failed"),
    filename: Optional[str] = Query(None, description="文件名包含"),
    sort: str = Query(
        "created_desc",
        description="created_desc|created_asc|score_desc|score_asc",
    ),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """获取当前用户的视频列表（分页、筛选、按创建时间或总分排序）。"""
    st = (status or "").strip() or None
    fn = (filename or "").strip() or None
    so = (sort or "created_desc").strip()
    if so not in _VIDEO_LIST_SORTS:
        raise HTTPException(status_code=400, detail="无效的 sort 参数，允许: created_desc|created_asc|score_desc|score_asc")
    if st is not None and st not in _VIDEO_LIST_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 status 参数，允许: {', '.join(sorted(_VIDEO_LIST_STATUSES))}",
        )
    base = db.query(Video).filter(Video.user_id == int(current_user))
    if st:
        base = base.filter(Video.status == st)
    if fn:
        base = base.filter(Video.filename.contains(fn))
    total = base.count()
    score_ex = func.json_extract(Video.video_score, "$[10]")
    if so == "score_desc":
        base = base.order_by(nullslast(desc(score_ex)), desc(Video.created_at))
    elif so == "score_asc":
        base = base.order_by(nullslast(asc(score_ex)), desc(Video.created_at))
    elif so == "created_asc":
        base = base.order_by(Video.created_at.asc())
    else:
        base = base.order_by(Video.created_at.desc())
    videos = base.offset(skip).limit(limit).all()
    return VideoListResponse(items=videos, total=total, skip=skip, limit=limit)


@app.get("/api/health")
def health_check():
    """进程存活探测，供网关或本地脚本使用。"""
    return {"status": "ok"}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """删除视频"""
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == int(current_user)).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video.file_path
    danmu_path = video.danmu_path
    thumb_path = video.thumbnail_path

    db.delete(video)
    db.commit()

    # 直接删除文件
    safe_remove_file(video_path)
    if danmu_path:
        safe_remove_file(danmu_path)
    if thumb_path:
        safe_remove_file(thumb_path)

    return {"message": "Video deleted successfully"}


async def async_safe_remove(video_path: str, danmu_path: str):
    """异步删除文件"""
    safe_remove_file(video_path)
    if danmu_path:
        safe_remove_file(danmu_path)


def _parse_stream_range(range_header: str, file_size: int) -> tuple[int, int]:
    """解析单个 ``Range: bytes=…``，返回含端点的 ``(start, end)``。"""
    if file_size <= 0:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")
    if not range_header.startswith("bytes="):
        raise HTTPException(status_code=400, detail="无效的 Range 请求头")
    range_str = range_header[6:].strip()
    if not range_str or "," in range_str:
        raise HTTPException(status_code=400, detail="无效的 Range 请求头")
    if "-" not in range_str:
        raise HTTPException(status_code=400, detail="无效的 Range 请求头")
    start_s, end_s = range_str.split("-", 1)
    start_s, end_s = start_s.strip(), end_s.strip()
    try:
        if start_s == "":
            if not end_s.isdigit():
                raise ValueError
            suffix_len = int(end_s)
            if suffix_len < 1:
                raise HTTPException(status_code=400, detail="无效的 Range 请求头")
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        else:
            start = int(start_s)
            if end_s == "":
                end = file_size - 1
            else:
                end = int(end_s)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的 Range 请求头")
    if start < 0 or start > end:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")
    if start >= file_size:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")
    if end >= file_size:
        end = file_size - 1
    return start, end


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

    redirect_url, file_path = stream_redirect_or_local(video.file_path)
    if redirect_url:
        return RedirectResponse(url=redirect_url, status_code=307)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    file_size = os.path.getsize(file_path)

    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = "video/mp4"

    range_header = request.headers.get("Range")
    if range_header:
        start, end = _parse_stream_range(range_header, file_size)

        content_length = end - start + 1

        async def file_iterator():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = VIDEO_STREAM_CHUNK_SIZE
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
    """更新当前登录用户的 Setting 记录；与敏感词表无直接耦合（敏感词走独立 API）。"""
    config_manager = ConfigManager(db)
    config_manager.update_config(config, db, int(current_user))
    return {"message": "配置更新成功", "config": config_manager.get_config(int(current_user))}


# 敏感词相关API接口（增删改需管理员；读取任登录用户可拉字面量列表）
@app.get("/api/sensitive-words")
def get_sensitive_words(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """获取敏感词字面量列表（与管理页展示一致）。"""
    from backend.models import SensitiveWord

    try:
        words = db.query(SensitiveWord.word).all()
        return {"sensitive_words": [word[0] for word in words]}
    except Exception:
        logging.getLogger(__name__).exception("获取敏感词失败")
        raise HTTPException(status_code=500, detail="获取敏感词失败")


@app.get("/api/sensitive-words/rules", response_model=TypingList[SensitiveWordRuleResponse])
def list_sensitive_word_rules(
    db: Session = Depends(get_db), _admin: str = Depends(require_admin)
):
    """管理员：敏感词完整规则（类别、正则、白名单、命中统计等）。"""
    from backend.models import SensitiveWord

    rows = db.query(SensitiveWord).order_by(SensitiveWord.id).all()
    return rows


@app.post("/api/sensitive-words")
def add_sensitive_word(
    item: SensitiveWordCreate,
    db: Session = Depends(get_db),
    _admin: str = Depends(require_admin),
):
    """管理员：添加敏感词规则。"""
    from backend.models import SensitiveWord

    w = (item.word or "").strip()
    if not w:
        raise HTTPException(status_code=400, detail="敏感词不能为空")
    if db.query(SensitiveWord).filter(SensitiveWord.word == w).first():
        raise HTTPException(status_code=400, detail="敏感词已存在")
    row = SensitiveWord(
        word=w,
        category=item.category or "",
        is_regex=item.is_regex,
        is_whitelist=item.is_whitelist,
        action=item.action or "block",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"message": "敏感词添加成功", "word": row.word, "id": row.id}


@app.delete("/api/sensitive-words/{word}")
def delete_sensitive_word(
    word: str, db: Session = Depends(get_db), _admin: str = Depends(require_admin)
):
    """管理员：删除敏感词。"""
    from backend.models import SensitiveWord

    sensitive_word = db.query(SensitiveWord).filter(SensitiveWord.word == word).first()
    if not sensitive_word:
        raise HTTPException(status_code=404, detail="敏感词不存在")
    db.delete(sensitive_word)
    db.commit()
    return {"message": "敏感词删除成功"}


@app.put("/api/sensitive-words")
def update_sensitive_words(
    words: list = Body(...),
    db: Session = Depends(get_db),
    _admin: str = Depends(require_admin),
):
    """管理员：批量重置敏感词（项可为字符串或对象）。"""
    from backend.models import SensitiveWord

    db.query(SensitiveWord).delete()
    for item in words:
        if not item:
            continue
        if isinstance(item, str):
            db.add(SensitiveWord(word=item))
        elif isinstance(item, dict):
            w = (item.get("word") or "").strip()
            if not w:
                continue
            db.add(
                SensitiveWord(
                    word=w,
                    category=(item.get("category") or "") or "",
                    is_regex=bool(item.get("is_regex")),
                    is_whitelist=bool(item.get("is_whitelist")),
                    action=(item.get("action") or "block") or "block",
                )
            )
    db.commit()
    flat = [x if isinstance(x, str) else (x.get("word") or "") for x in words]
    return {"message": "敏感词更新成功", "sensitive_words": [x for x in flat if x]}


def _sensitive_request_to_response(db: Session, r: SensitiveWordRequest) -> SensitiveWordRequestResponse:
    from backend.models import User

    u = db.query(User).filter(User.id == r.user_id).first()
    return SensitiveWordRequestResponse(
        id=r.id,
        word=r.word,
        category=r.category or "",
        is_regex=bool(r.is_regex),
        is_whitelist=bool(r.is_whitelist),
        action=r.action or "block",
        user_id=r.user_id,
        requester_username=u.username if u else "",
        status=r.status,
        created_at=r.created_at,
        reviewed_at=r.reviewed_at,
    )


@app.post("/api/sensitive-words/requests", response_model=SensitiveWordRequestResponse)
def create_sensitive_word_request(
    item: SensitiveWordCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """普通用户：提交敏感词添加申请（待管理员审核）。"""
    from backend.models import SensitiveWord, SensitiveWordRequest

    w = (item.word or "").strip()
    if not w:
        raise HTTPException(status_code=400, detail="敏感词不能为空")
    if db.query(SensitiveWord).filter(SensitiveWord.word == w).first():
        raise HTTPException(status_code=400, detail="该词已在词库中")
    dup = (
        db.query(SensitiveWordRequest)
        .filter(SensitiveWordRequest.word == w, SensitiveWordRequest.status == "pending")
        .first()
    )
    if dup:
        raise HTTPException(status_code=400, detail="该词已有待审核申请，请等待管理员处理")
    row = SensitiveWordRequest(
        word=w,
        category=item.category or "",
        is_regex=bool(item.is_regex),
        is_whitelist=bool(item.is_whitelist),
        action=(item.action or "block"),
        user_id=int(current_user),
        status="pending",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _sensitive_request_to_response(db, row)


@app.get(
    "/api/sensitive-words/requests/pending",
    response_model=TypingList[SensitiveWordRequestResponse],
)
def list_pending_sensitive_word_requests(
    db: Session = Depends(get_db), _admin: str = Depends(require_admin)
):
    """管理员：待审核的敏感词申请列表（先提交先审）。"""
    from backend.models import SensitiveWordRequest

    rows = (
        db.query(SensitiveWordRequest)
        .filter(SensitiveWordRequest.status == "pending")
        .order_by(SensitiveWordRequest.created_at.asc())
        .all()
    )
    return [_sensitive_request_to_response(db, r) for r in rows]


@app.get(
    "/api/sensitive-words/requests/mine",
    response_model=TypingList[SensitiveWordRequestResponse],
)
def list_my_sensitive_word_requests(
    db: Session = Depends(get_db), current_user: str = Depends(get_current_user)
):
    """当前用户已提交的申请记录（含已通过/已拒绝）。"""
    from backend.models import SensitiveWordRequest

    rows = (
        db.query(SensitiveWordRequest)
        .filter(SensitiveWordRequest.user_id == int(current_user))
        .order_by(SensitiveWordRequest.created_at.desc())
        .limit(100)
        .all()
    )
    return [_sensitive_request_to_response(db, r) for r in rows]


@app.post(
    "/api/sensitive-words/requests/{request_id}/approve",
    response_model=SensitiveWordRequestResponse,
)
def approve_sensitive_word_request(
    request_id: int,
    db: Session = Depends(get_db),
    admin_uid: str = Depends(require_admin),
):
    """管理员：同意申请并写入词库（若词已存在则仅标记通过）。"""
    from backend.models import SensitiveWord, SensitiveWordRequest

    req = db.query(SensitiveWordRequest).filter(SensitiveWordRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="申请不存在")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail="该申请已处理")
    exists = db.query(SensitiveWord).filter(SensitiveWord.word == req.word).first()
    if not exists:
        db.add(
            SensitiveWord(
                word=req.word,
                category=req.category or "",
                is_regex=bool(req.is_regex),
                is_whitelist=bool(req.is_whitelist),
                action=req.action or "block",
            )
        )
    req.status = "approved"
    req.reviewed_by_user_id = int(admin_uid)
    req.reviewed_at = datetime.now()
    db.commit()
    db.refresh(req)
    return _sensitive_request_to_response(db, req)


@app.post(
    "/api/sensitive-words/requests/{request_id}/reject",
    response_model=SensitiveWordRequestResponse,
)
def reject_sensitive_word_request(
    request_id: int,
    db: Session = Depends(get_db),
    admin_uid: str = Depends(require_admin),
):
    """管理员：拒绝申请。"""
    from backend.models import SensitiveWordRequest

    req = db.query(SensitiveWordRequest).filter(SensitiveWordRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="申请不存在")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail="该申请已处理")
    req.status = "rejected"
    req.reviewed_by_user_id = int(admin_uid)
    req.reviewed_at = datetime.now()
    db.commit()
    db.refresh(req)
    return _sensitive_request_to_response(db, req)


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

                effective_mids = effective_admin_mids(db, ADMIN_BILIBILI_MIDS)
                role = "admin" if bilibili_mid in effective_mids else "user"

                if existing_user:
                    # 更新用户
                    existing_user.username = username  # 更新用户名（用户可能改名）
                    existing_user.email = email
                    existing_user.sessdata = sessdata  # 更新sessdata
                    # 更新B站用户信息
                    existing_user.level = data.get("level")
                    existing_user.updated_at = datetime.now()
                    existing_user.role = role
                    db.commit()
                    user_id = existing_user.id
                else:
                    # 创建新用户
                    new_user = User(
                        bilibili_mid=bilibili_mid,
                        username=username,
                        email=email,
                        sessdata=sessdata,  # 存储sessdata
                        level=data.get("level"),
                        role=role,
                    )
                    db.add(new_user)
                    db.commit()
                    db.refresh(new_user)
                    user_id = new_user.id

                    seed_default_settings_for_user(db, user_id)

                # 生成访问令牌
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                user_obj = db.query(User).filter(User.id == user_id).first()
                token_role = user_obj.role if user_obj and user_obj.role else role
                access_token = create_access_token(
                    data={"sub": str(user_id), "role": token_role},
                    expires_delta=access_token_expires,
                )

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
                        "role": token_role,
                        "created_at": user_obj.created_at,
                        "updated_at": user_obj.updated_at
                    }
                }
            else:
                # 获取用户信息失败
                return {"status": "error", "code": user_info["code"],
                        "message": user_info.get("message", "获取用户信息失败")}
        elif qr_status["code"] == -1:
            # 网络错误
            return {"status": "error", "code": qr_status["code"],
                    "message": qr_status.get("message", "网络错误，请检查网络连接后重试")}

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
        "role": user.role or "user",
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }


@app.post("/api/auth/bilibili/refresh")
def refresh_bilibili_session(
    current_user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    """校验当前用户 ``sessdata`` 是否仍被 B 站接受；失效时需重新扫码。"""
    user = db.query(User).filter(User.id == int(current_user)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        client = BiliClient(user.sessdata)
        nav = client.simple_get("https://api.bilibili.com/x/web-interface/nav")
        data = nav.json()
        if data.get("code") != 0:
            raise HTTPException(
                status_code=401,
                detail="B 站会话已失效，请重新扫码登录",
            )
        return {"ok": True, "message": "会话有效"}
    except HTTPException:
        raise
    except Exception:
        logging.getLogger(__name__).exception("B 站会话校验失败")
        raise HTTPException(status_code=500, detail="会话校验失败")


@app.post("/api/auth/logout")
def logout(current_user: str = Depends(get_current_user)):
    """登出"""
    # JWT是无状态的，客户端删除令牌即可
    return {"message": "登出成功"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
