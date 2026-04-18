"""
用户级配置与默认配置：处理参数、模型路径、下载选项等。

``ConfigManager`` 将 ``Setting`` 表与 ``SensitiveWord`` 表读写成扁平 dict；
注意 ``save_config`` 在保存时会根据 dict 中的 ``sensitive_words`` 全量重写敏感词表，
若通过 ``update_config`` 更新，会先 ``get_config`` 再合并，通常仍会带上当前词表。
"""

import os
from typing import Dict, Any
from sqlalchemy.orm import Session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 应用内默认值；首启无 Setting 行时会落库一份。
# 修改时请同步：backend/user_defaults.seed_default_settings_for_user、
# frontend/src/components/Settings.vue（reactive 初值与 resetConfig）。
DEFAULT_CONFIG = {
    "video_processing": {
        "num_clips": 125,
        "frames_per_clip": 5,
        "target_size": [224, 224]
    },
    "audio_processing": {
        "sample_rate": 16000
    },
    "model_paths": {
        "whisper_model": os.path.join(BASE_DIR, "model/whisper-medium"),
        "audio_model": os.path.join(BASE_DIR, "model/ast-finetuned-audioset-10-10-0.4593"),
        "text_model": os.path.join(BASE_DIR, "model/roberta-wwm-ext"),
        "visual_model": os.path.join(BASE_DIR, "model/vit-base-patch16-224"),
        "traced_model_path": os.path.join(BASE_DIR, "model/mvqa_traced.pt")
    },
    "sensitive_words": ["违禁词", "暴力", "涉黄", "反动", "测试", "傻叉", "漫展", "三十"],
    "download_settings": {
        "video_quality": "auto",  # auto, high, medium, low
        "default_video_quality": "80",  # 默认视频质量
        "audio_quality": "high",  # high, medium, low
        "default_audio_quality": "high",  # 默认音频质量
        "download_danmaku": True  # 是否下载弹幕
    }
}

class ConfigManager:
    """按 ``user_id`` 读写 ``Setting``；敏感词从全局 ``SensitiveWord`` 表聚合进返回 dict。"""

    def __init__(self, db: Session = None):
        self.db = db
    
    def load_config(self, db: Session = None, user_id: int = None) -> Dict[str, Any]:
        """从数据库加载配置"""
        from backend.database import SessionLocal
        from backend.models import Setting, SensitiveWord
        
        if db is None:
            db = SessionLocal()
        
        try:
            # 从数据库获取当前用户的配置
            settings = db.query(Setting).filter(Setting.user_id == user_id).all()
            config = {}
            
            for setting in settings:
                config[setting.key] = setting.value
            
            # 从敏感词表加载 DFA 用字面量（排除白名单/正则行）
            from backend.utils.sensitive_rules import literal_words_for_dfa

            config["sensitive_words"] = literal_words_for_dfa(db)
            
            # 如果数据库中没有配置（settings表为空），使用默认配置并保存到数据库
            if not settings:
                config = DEFAULT_CONFIG
                self.save_config(config, db, user_id)
            
            return config
        except Exception as e:
            return DEFAULT_CONFIG
        finally:
            # 不要关闭从外部传入的数据库会话，让调用者负责关闭
            if db is None:
                db.close()
    
    def save_config(self, config: Dict[str, Any], db: Session = None, user_id: int = None):
        """保存配置到数据库"""
        from backend.database import SessionLocal
        from backend.models import Setting, SensitiveWord
        
        if db is None:
            db = SessionLocal()
        
        try:
            # 保存每个配置项，排除敏感词
            sensitive_words = config.pop('sensitive_words', [])
            
            for key, value in config.items():
                # 查找现有配置
                setting = db.query(Setting).filter(Setting.key == key, Setting.user_id == user_id).first()
                
                if setting:
                    # 更新现有配置
                    setting.value = value
                else:
                    # 创建新配置
                    setting = Setting(
                        key=key,
                        value=value,
                        user_id=user_id
                    )
                    db.add(setting)
            
            # 处理敏感词
            # 先删除所有现有敏感词
            db.query(SensitiveWord).delete()
            for item in sensitive_words:
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
        except Exception as e:
            db.rollback()
        finally:
            # 不要关闭从外部传入的数据库会话，让调用者负责关闭
            if db is None:
                db.close()

    def get_config(self, user_id: int = None, db: Session = None) -> Dict[str, Any]:
        """获取当前配置"""
        from backend.database import SessionLocal
        from backend.models import Setting, SensitiveWord
        
        # 使用传入的数据库会话，如果没有则创建新的
        use_external_db = db is not None
        if not db:
            db = SessionLocal()
        try:
            # 从数据库获取所有配置
            settings = db.query(Setting).filter(Setting.user_id == user_id).all()
            config = {}
            
            for setting in settings:
                config[setting.key] = setting.value
            
            # 从敏感词表加载 DFA 用字面量（排除白名单/正则行）
            from backend.utils.sensitive_rules import literal_words_for_dfa

            config["sensitive_words"] = literal_words_for_dfa(db)
            
            # 如果数据库中没有配置（settings表为空），使用默认配置并保存到数据库
            if not settings:
                config = DEFAULT_CONFIG
                self.save_config(config, db, user_id)
            
            return config
        except Exception as e:
            return DEFAULT_CONFIG
        finally:
            # 只有当数据库会话是在该方法内部创建的时才关闭
            if not use_external_db:
                db.close()

    def update_config(self, new_config: Dict[str, Any], db: Session = None, user_id: int = None):
        """更新配置"""
        # 合并配置
        current_config = self.get_config(user_id, db)
        current_config.update(new_config)
        # 保存到数据库
        self.save_config(current_config, db, user_id)

# 创建全局配置管理器实例（用于系统级操作，不影响用户特定配置）
from backend.database import SessionLocal
db = SessionLocal()
config_manager = ConfigManager(db)
db.close()

# 导出默认配置供其他模块使用
CONFIG = DEFAULT_CONFIG
