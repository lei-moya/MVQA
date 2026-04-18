"""管理员 B 站 mid：库内配置与环境变量 ``ADMIN_BILIBILI_MIDS`` 合并，登录时同步 ``User.role``。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.models import SystemKV

ADMIN_MIDS_KEY = "admin_bilibili_mids"


def parse_mids_csv(csv: str) -> set[str]:
    return {x.strip() for x in (csv or "").split(",") if x.strip()}


def get_stored_admin_mids_raw(db: Session) -> str:
    row = db.query(SystemKV).filter(SystemKV.key == ADMIN_MIDS_KEY).first()
    if not row or row.value is None:
        return ""
    return str(row.value).strip()


def set_stored_admin_mids_raw(db: Session, value: str) -> None:
    v = (value or "").strip()
    row = db.query(SystemKV).filter(SystemKV.key == ADMIN_MIDS_KEY).first()
    if row:
        row.value = v
    else:
        db.add(SystemKV(key=ADMIN_MIDS_KEY, value=v))
    db.commit()


def effective_admin_mids(db: Session, env_mids: set[str]) -> set[str]:
    out = set(env_mids)
    out.update(parse_mids_csv(get_stored_admin_mids_raw(db)))
    return out
