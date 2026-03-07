import sqlite3

# 连接到SQLite数据库
conn = sqlite3.connect('video_rating.db')
cursor = conn.cursor()

# 检查sensitive_words表是否存在
try:
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sensitive_words';")
    table_exists = cursor.fetchone()
    
    if table_exists:
        print("sensitive_words表存在")
        
        # 检查表中的数据
        cursor.execute("SELECT * FROM sensitive_words;")
        words = cursor.fetchall()
        
        if words:
            print(f"找到{len(words)}个敏感词:")
            for word in words:
                print(f"- {word[1]}")
        else:
            print("sensitive_words表为空")
    else:
        print("sensitive_words表不存在")
        
        # 检查所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("数据库中的表:")
        for table in tables:
            print(f"- {table[0]}")
            
except Exception as e:
    print(f"错误: {e}")
finally:
    conn.close()
