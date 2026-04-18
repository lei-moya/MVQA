#!/usr/bin/env python3
"""
初始化敏感词数据到数据库
"""

from backend.database import SessionLocal
from backend.models import SensitiveWord
from backend.config import DEFAULT_CONFIG

def init_sensitive_words():
    """初始化敏感词数据"""
    db = SessionLocal()
    try:
        # 检查是否已有敏感词
        existing_words = db.query(SensitiveWord).count()
        
        if existing_words == 0:
            # 添加默认敏感词
            default_words = DEFAULT_CONFIG.get('sensitive_words', [])
            
            for word in default_words:
                if word:
                    sensitive_word = SensitiveWord(word=word)
                    db.add(sensitive_word)
            
            db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_sensitive_words()
