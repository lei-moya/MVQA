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
        print(f"现有敏感词数量: {existing_words}")
        
        if existing_words == 0:
            # 添加默认敏感词
            print("开始初始化敏感词...")
            default_words = DEFAULT_CONFIG.get('sensitive_words', [])
            print(f"默认敏感词列表: {default_words}")
            
            for word in default_words:
                if word:
                    sensitive_word = SensitiveWord(word=word)
                    db.add(sensitive_word)
                    print(f"添加敏感词: {word}")
            
            db.commit()
            print("敏感词初始化成功!")
        else:
            # 显示现有敏感词
            print("敏感词已存在，显示当前敏感词:")
            words = db.query(SensitiveWord.word).all()
            for word in words:
                print(f"- {word[0]}")
    except Exception as e:
        print(f"初始化敏感词失败: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_sensitive_words()
