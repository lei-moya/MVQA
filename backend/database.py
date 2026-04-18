"""
数据库连接：单文件 SQLite，供开发与单机部署使用。

启用 WAL 减轻并发读写下的锁等待；生产高并发可改为 PostgreSQL/MySQL。
"""

from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker

# 项目根目录（backend 的上一级），与进程 cwd 无关
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_FILE = PROJECT_ROOT / "video_rating.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_FILE.as_posix()}"
# SQLite 需要特殊的 connect_args
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(engine, "connect")
def _sqlite_on_connect(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def ensure_schema() -> None:
    """为已有 SQLite 库追加 ORM 新增列（create_all 不会 ALTER）。"""
    try:
        insp = inspect(engine)
        if "videos" not in insp.get_table_names():
            return
        vcols = {c["name"] for c in insp.get_columns("videos")}
        ucols = {c["name"] for c in insp.get_columns("users")} if "users" in insp.get_table_names() else set()
        scols = (
            {c["name"] for c in insp.get_columns("sensitive_words")}
            if "sensitive_words" in insp.get_table_names()
            else set()
        )
        stmts = []
        if "thumbnail_path" not in vcols:
            stmts.append("ALTER TABLE videos ADD COLUMN thumbnail_path VARCHAR")
        if "progress" not in vcols:
            stmts.append("ALTER TABLE videos ADD COLUMN progress FLOAT DEFAULT 0")
        if "progress_stage" not in vcols:
            stmts.append("ALTER TABLE videos ADD COLUMN progress_stage VARCHAR")
        if "source_url" not in vcols:
            stmts.append("ALTER TABLE videos ADD COLUMN source_url VARCHAR")
        if "role" not in ucols:
            stmts.append("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user'")
        if "category" not in scols:
            stmts.append("ALTER TABLE sensitive_words ADD COLUMN category VARCHAR DEFAULT ''")
        if "is_regex" not in scols:
            stmts.append("ALTER TABLE sensitive_words ADD COLUMN is_regex INTEGER DEFAULT 0")
        if "is_whitelist" not in scols:
            stmts.append("ALTER TABLE sensitive_words ADD COLUMN is_whitelist INTEGER DEFAULT 0")
        if "action" not in scols:
            stmts.append("ALTER TABLE sensitive_words ADD COLUMN action VARCHAR DEFAULT 'block'")
        if "hit_count" not in scols:
            stmts.append("ALTER TABLE sensitive_words ADD COLUMN hit_count INTEGER DEFAULT 0")
        if not stmts:
            return
        with engine.begin() as conn:
            for s in stmts:
                conn.execute(text(s))
    except Exception:
        pass
