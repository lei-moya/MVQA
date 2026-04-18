"""
敏感词策略：DFA 用字面量库；正则 / 白名单 / 告警在旁路匹配并可累计命中（best-effort）。
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

from sqlalchemy.orm import Session

from backend.models import SensitiveWord

_log = logging.getLogger(__name__)


def literal_words_for_dfa(db: Session) -> List[str]:
    """参与 DFA 的字面量：非白名单、非正则；block / warn 均参与匹配（影响检测强度）。"""
    rows = (
        db.query(SensitiveWord)
        .filter(
            (SensitiveWord.is_whitelist != True),  # noqa: E712
            (SensitiveWord.is_regex != True),
        )
        .all()
    )
    return [r.word for r in rows if r.word]


def regex_rules(db: Session) -> List[Tuple[re.Pattern, SensitiveWord]]:
    out: List[Tuple[re.Pattern, SensitiveWord]] = []
    for r in db.query(SensitiveWord).filter(SensitiveWord.is_regex == True).all():  # noqa: E712
        try:
            out.append((re.compile(r.word), r))
        except re.error:
            _log.warning("跳过无效正则敏感规则 id=%s", r.id)
    return out


def match_regex_on_text(db: Session, text: str) -> List[SensitiveWord]:
    if not text:
        return []
    hits: List[SensitiveWord] = []
    for pat, row in regex_rules(db):
        if pat.search(text):
            hits.append(row)
    return hits


def bump_hit_counts(db: Session, rows: List[SensitiveWord]) -> None:
    if not rows:
        return
    try:
        for r in rows:
            r.hit_count = (r.hit_count or 0) + 1
        db.commit()
    except Exception:
        db.rollback()


def strip_whitelist_spans(text: str, db: Session) -> str:
    """从转写文本中移除白名单字面短语（简单子串替换）。"""
    if not text:
        return text
    spans = [w.word for w in db.query(SensitiveWord).filter(SensitiveWord.is_whitelist.is_(True)).all() if w.word]
    out = text
    for w in sorted(spans, key=len, reverse=True):
        out = out.replace(w, "")
    return out
