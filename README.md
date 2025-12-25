# MVQA: Multimodal Video Quality Assessment

## 项目详情请见[ProjectDescription](ProjectDescription.md)

## 代码使用说明

本项目使用ViT + PANNs + RoBERTa搭建多模态短视频质量评估模型。

### 安装依赖
```bash
pip install -r requirements.txt
```

### 数据准备
- 视频帧：预提取为JPEG图像序列。
- 音频：WAV格式。
- 文本：弹幕数据，按时间段分组。

### 训练模型
```bash
python train.py
```

### 推理
```bash
python inference.py
```

### 文件结构
- `model.py`: 多模态模型定义。
- `data_preprocessing.py`: 数据加载和预处理。
- `train.py`: 训练脚本。
- `inference.py`: 推理脚本。