import sqlite3
from pathlib import Path

# 与 database.py 一致：项目根目录下的 video_rating.db
_db_path = Path(__file__).resolve().parent.parent / "video_rating.db"
conn = sqlite3.connect(str(_db_path))
cursor = conn.cursor()

# 检查sensitive_words表是否存在
try:
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sensitive_words';")
    table_exists = cursor.fetchone()
    
    if table_exists:
        # 检查表中的数据
        cursor.execute("SELECT * FROM sensitive_words;")
        words = cursor.fetchall()
    else:
        words = []
        
        # 检查所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
except Exception as e:
    print(f"错误: {e}")
finally:
    conn.close()
