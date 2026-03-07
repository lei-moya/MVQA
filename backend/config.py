import os
import json
from typing import Dict, Any
from sqlalchemy.orm import Session

# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认配置
DEFAULT_CONFIG = {
    "video_processing": {
        "num_clips": 5,
        "frames_per_clip": 5,
        "target_size": [224, 224]
    },
    "audio_processing": {
        "sample_rate": 16000
    },
    "model_paths": {
        "video_model": os.path.join(BASE_DIR, "models/deepspeech-0.9.3-models-zh-CN/deepspeech-0.9.3-models-zh-CN.pbmm"),
        "video_scorer": os.path.join(BASE_DIR, "models/deepspeech-0.9.3-models-zh-CN/deepspeech-0.9.3-models-zh-CN.scorer"),
        "audio_model": os.path.join(BASE_DIR, "models/ast-finetuned-audioset-10-10-0.4593"),
        "text_model": os.path.join(BASE_DIR, "models/roberta-wwm-ext"),
        "visual_model": os.path.join(BASE_DIR, "models/vit-base-patch16-224"),
        "traced_model_path": os.path.join(BASE_DIR, "models/mvqa.pt")
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
    """配置管理器"""
    
    def __init__(self, db: Session = None):
        # 初始化时不加载配置，而是在调用 get_config 时加载
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
            
            # 从敏感词表加载敏感词
            sensitive_words = db.query(SensitiveWord.word).all()
            config['sensitive_words'] = [word[0] for word in sensitive_words]
            
            # 如果数据库中没有配置（settings表为空），使用默认配置并保存到数据库
            if not settings:
                config = DEFAULT_CONFIG
                self.save_config(config, db, user_id)
            
            return config
        except Exception as e:
            print(f"加载配置失败: {e}")
            return DEFAULT_CONFIG
        finally:
            if db is not SessionLocal():
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
            # 添加新的敏感词
            for word in sensitive_words:
                if word:
                    sensitive_word = SensitiveWord(word=word)
                    db.add(sensitive_word)
            
            db.commit()
        except Exception as e:
            print(f"保存配置失败: {e}")
            db.rollback()
        finally:
            if db is not SessionLocal():
                db.close()

    def get_config(self, user_id: int = None) -> Dict[str, Any]:
        """获取当前配置"""
        from backend.database import SessionLocal
        from backend.models import Setting, SensitiveWord
        
        db = SessionLocal()
        try:
            # 从数据库获取所有配置
            settings = db.query(Setting).filter(Setting.user_id == user_id).all()
            config = {}
            
            for setting in settings:
                config[setting.key] = setting.value
            
            # 从敏感词表加载敏感词
            sensitive_words = db.query(SensitiveWord.word).all()
            config['sensitive_words'] = [word[0] for word in sensitive_words]
            
            # 如果数据库中没有配置（settings表为空），使用默认配置并保存到数据库
            if not settings:
                config = DEFAULT_CONFIG
                self.save_config(config, db, user_id)
            
            return config
        except Exception as e:
            print(f"加载配置失败: {e}")
            return DEFAULT_CONFIG
        finally:
            db.close()

    def update_config(self, new_config: Dict[str, Any], db: Session = None, user_id: int = None):
        """更新配置"""
        # 合并配置
        current_config = self.get_config(user_id)
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
