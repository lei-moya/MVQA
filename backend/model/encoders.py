import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    ViTModel,
    ASTModel,  # 修改：使用 ASTModel 替代 ASTForAudioClassification，更适合特征提取
)
import numpy as np
from backend.config import CONFIG


class AudioEncoder(nn.Module):
    """音频编码器 - 使用AST模型"""

    def __init__(self, freeze=True):
        super().__init__()
        # 修改：使用 ASTModel，配合 torchscript=True
        self.ast = ASTModel.from_pretrained(CONFIG["model_paths"]["audio_model"], torchscript=True)

        # 获取AST的隐藏维度
        self.hidden_size = self.ast.config.hidden_size  # 通常是768

        # 是否冻结参数
        if freeze:
            for param in self.ast.parameters():
                param.requires_grad = False

    def forward(self, audio_features):
        """
        Args:
            audio_features: (batch, time_steps, mel_bins)
        Returns:
            audio_embedding: (batch, hidden_size)
        """
        # AST需要特定的输入格式
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features).float()

        # TorchScript 模式下，输出是一个元组
        # outputs[0] 是 last_hidden_state，形状为 (batch, sequence_length, hidden_size)
        outputs = self.ast(audio_features)

        # 获取 last_hidden_state
        last_hidden_state = outputs[0]

        # 提取 [CLS] token (第 0 个位置) 作为整段音频的特征
        audio_embedding = last_hidden_state[:, 0, :]

        return audio_embedding


class TextEncoder(nn.Module):
    """文本编码器 - 使用RoBERTa"""

    def __init__(self, freeze=True):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(CONFIG["model_paths"]["text_model"], torchscript=True)

        self.hidden_size = self.roberta.config.hidden_size  # 768

        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            text_embedding: (batch, hidden_size)
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # 【关键修正】
        # TorchScript 模式下 outputs 是元组，使用 outputs[0] 获取 last_hidden_state
        last_hidden_state = outputs[0]

        # 使用[CLS] token的输出 (第 0 个位置)
        text_embedding = last_hidden_state[:, 0, :]

        return text_embedding


class VisualEncoder(nn.Module):
    """视觉编码器 - 使用ViT"""

    def __init__(self, freeze=True):
        super().__init__()

        self.vit = ViTModel.from_pretrained(CONFIG["model_paths"]["visual_model"], torchscript=True)

        self.hidden_size = self.vit.config.hidden_size  # 768

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (batch, num_frames, C, H, W)
        Returns:
            visual_embedding: (batch, hidden_size)
        """
        batch_size, num_frames = pixel_values.shape[:2]
        # 展平成
        pixel_values_flat = pixel_values.view(-1, *pixel_values.shape[2:])

        outputs = self.vit(pixel_values_flat)

        # 【关键修正】
        # TorchScript 模式下 outputs 是元组，使用 outputs[0] 获取 last_hidden_state
        last_hidden_state = outputs[0]

        # 提取 [CLS] token
        embeddings = last_hidden_state[:, 0, :]  # (batch*num_frames, hidden_size)

        # 重塑回
        visual_embedding = embeddings.view(batch_size, num_frames, -1)

        # 时序聚合（平均池化）
        visual_embedding = visual_embedding.mean(dim=1)  # (batch, hidden_size)

        return visual_embedding