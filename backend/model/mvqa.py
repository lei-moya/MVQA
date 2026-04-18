import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.model.encoders import AudioEncoder, TextEncoder, VisualEncoder
from backend.model.fusion import LowRankMultimodalFusion, TemporalAggregation


class MVQA(nn.Module):
    """
    支持片段级和视频级双层预测的MVQA模型

    架构:
    1. 编码器: AudioEncoder + TextEncoder + VisualEncoder
    2. 融合层: 对每个片段独立融合
    3. 片段级预测头: 对每个片段输出 12 个回归值
    4. 时序聚合: 将片段特征聚合为视频级特征
    5. 视频级预测头: 输出整个视频的 12 个回归值

    支持的训练模式:
    - clip_only: 只训练片段级预测
    - video_only: 只训练视频级预测
    - joint: 同时训练两级预测
    - hierarchical: 层次化训练（先clip后video）
    """

    def __init__(
            self,
            fusion_dim=512,
            output_dim=12,
            freeze_pretrained=True,
            lmf_rank=4,
            temporal_aggregation='attention',  # 'attention', 'mean', 'max', 'weighted_mean'
            clip_weight=1.0,
            video_weight=1.0,
            dropout=0.3,
            max_clips=125  # 最大片段数，用于位置编码
    ):
        super().__init__()

        # 保存配置
        self.CLIP_WEIGHT = clip_weight
        self.VIDEO_WEIGHT = video_weight
        self.output_dim = output_dim
        self.max_clips = max_clips

        print("\n" + "=" * 70)
        print("初始化 MVQA 模型")
        print("=" * 70)

        # ==========================================
        # 1. 编码器
        # ==========================================
        self.audio_encoder = AudioEncoder(freeze=freeze_pretrained)
        self.text_encoder = TextEncoder(freeze=freeze_pretrained)
        self.visual_encoder = VisualEncoder(freeze=freeze_pretrained)

        # 获取各编码器的输出维度
        audio_dim = self.audio_encoder.hidden_size
        text_dim = self.text_encoder.hidden_size
        visual_dim = self.visual_encoder.hidden_size

        print(f"\n编码器维度:")
        print(f"  - 音频: {audio_dim}")
        print(f"  - 文本: {text_dim}")
        print(f"  - 视觉: {visual_dim}")

        # ==========================================
        # 片段位置编码
        # ==========================================
        # 为每个片段添加位置编码，增强片段间的差异
        self.clip_pos_embedding = nn.Embedding(max_clips, fusion_dim)

        # ==========================================
        # 片段级预测头
        # ==========================================
        self.clip_fusion = LowRankMultimodalFusion(
            audio_dim, text_dim, visual_dim, fusion_dim, rank=lmf_rank, dropout=dropout
        )

        # 增强片段级预测头，增加复杂度以捕捉片段特有的特征
        self.clip_regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, output_dim)
        )

        # ==========================================
        # 时序聚合模块
        # ==========================================
        self.audio_temporal_agg = TemporalAggregation(
            audio_dim, aggregation_type=temporal_aggregation, dropout=dropout
        )
        self.visual_temporal_agg = TemporalAggregation(
            visual_dim, aggregation_type=temporal_aggregation, dropout=dropout
        )

        # ==========================================
        # 视频级预测头
        # ==========================================
        self.video_fusion = LowRankMultimodalFusion(
            audio_dim, text_dim, visual_dim, fusion_dim, rank=lmf_rank, dropout=dropout
        )

        self.video_regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, output_dim)
        )

        self._print_model_info(fusion_dim, output_dim)

    def _print_model_info(self, fusion_dim, num_answers):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n融合层维度: {fusion_dim}")
        print(f"输出类别数: {num_answers}")
        print(f"\n模型参数统计:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练: {trainable_params:,}")
        print(f"  - 冻结参数: {frozen_params:,}")
        print("=" * 70 + "\n")

    def forward(
            self,
            audio_clips,
            aside_input_ids,
            aside_attention_mask,
            danmu_input_ids,
            danmu_attention_mask,
            visual_clips,
            return_clip_features=False
    ):
        """
        前向传播

        Args:
            audio_clips: (batch, num_clips, time_steps, mel_bins)
            aside_input_ids: (batch, seq_len) 或 (batch, num_clips, seq_len)
            aside_attention_mask: (batch, seq_len)
            danmu_input_ids: (batch, num_clips, seq_len)
            danmu_attention_mask: (batch, num_clips, seq_len)
            visual_clips: (batch, num_clips, num_frames, C, H, W)
            return_clip_features: 是否返回片段级特征（用于可视化/分析）

        Returns:
            outputs: dict
                - clip_outputs: (batch, num_clips, 12) 片段级回归预测
                - video_outputs: (batch, 12) 视频级回归预测
                - clip_features: (batch, num_clips, fusion_dim) [可选]
        """
        batch_size, num_clips = audio_clips.shape[:2]

        # ==========================================
        # 1. 文本编码
        # ==========================================
        # 旁白对所有片段共享，只编码一次
        aside_emb = self.text_encoder(aside_input_ids, aside_attention_mask)  # (batch, text_dim)

        # 每个片段有独立的弹幕，需要展平后统一编码
        # (batch, num_clips, seq_len) -> (batch * num_clips, seq_len)
        danmu_ids_flat = danmu_input_ids.view(batch_size * num_clips, -1)
        danmu_mask_flat = danmu_attention_mask.view(batch_size * num_clips, -1)

        # 编码
        danmu_emb_flat = self.text_encoder(danmu_ids_flat, danmu_mask_flat)  # (batch*num_clips, text_dim)

        # ==========================================
        # 2. 展平所有片段，统一编码
        # ==========================================
        # (batch, num_clips, ...) -> (batch * num_clips, ...)
        audio_flat = audio_clips.view(batch_size * num_clips, *audio_clips.shape[2:])
        visual_flat = visual_clips.view(batch_size * num_clips, *visual_clips.shape[2:])

        # 编码
        audio_emb = self.audio_encoder(audio_flat)  # (batch*num_clips, audio_dim)
        visual_emb = self.visual_encoder(visual_flat)  # (batch*num_clips, visual_dim)

        # ==========================================
        # 2. 片段级预测路径
        # ==========================================
        # 使用 LowRank Multimodal Fusion
        clip_features_flat = self.clip_fusion(audio_emb, danmu_emb_flat, visual_emb)

        # 添加片段位置编码，增强片段间的差异
        # 创建位置索引: (batch_size, num_clips) -> (batch_size * num_clips)
        clip_indices = torch.arange(num_clips, device=clip_features_flat.device)
        clip_indices = clip_indices.repeat(batch_size)
        pos_emb = self.clip_pos_embedding(clip_indices)  # (batch*num_clips, fusion_dim)

        # 融合位置编码
        clip_features_flat = clip_features_flat + pos_emb

        # 预测
        clip_outputs_flat = self.clip_regressor(clip_features_flat)  # (batch*num_clips, 12)
        clip_outputs = clip_outputs_flat.view(batch_size, num_clips, self.output_dim)

        # ==========================================
        # 3. 视频级预测路径
        # ==========================================
        # 3.1 分别聚合各模态特征
        # 将展平的特征恢复为
        audio_emb_3d = audio_emb.view(batch_size, num_clips, -1)
        visual_emb_3d = visual_emb.view(batch_size, num_clips, -1)

        # 对原始模态特征进行时序聚合
        global_audio = self.audio_temporal_agg(audio_emb_3d)  # (batch, audio_dim)
        global_visual = self.visual_temporal_agg(visual_emb_3d)  # (batch, visual_dim)
        # 文本使用全局旁白 aside_emb

        # 3.2 使用 Low-Rank Multimodal Fusion
        video_feature = self.video_fusion(global_audio, aside_emb, global_visual)  # (batch, fusion_dim)

        # 预测
        video_outputs = self.video_regressor(video_feature)

        outputs = {
            'clip_outputs': clip_outputs,  # (batch, num_clips, 12)
            'video_outputs': video_outputs  # (batch, 12)
        }

        if return_clip_features:
            outputs['clip_features'] = clip_features_flat.view(batch_size, num_clips, -1)

        return outputs

    def compute_loss(
            self,
            outputs,
            clip_targets=None,
            video_targets=None,
            loss_type='joint'
    ):
        """
        计算回归损失

        Args:
            outputs: forward() 的输出
            clip_targets: (batch, num_clips, 12) 片段级目标值
            video_targets: (batch, 12) 视频级目标值
            loss_type: 'clip_only', 'video_only', 'joint'
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        # ==========================================
        # 1. 片段级损失
        # ==========================================
        if clip_targets is not None:
            clip_preds = outputs['clip_outputs']  # (batch, num_clips, 12)

            # 使用均方误差
            clip_loss = F.mse_loss(clip_preds, clip_targets)

            loss_dict['clip_loss'] = clip_loss.item()

            if loss_type in ['clip_only', 'joint']:
                total_loss = total_loss + self.CLIP_WEIGHT * clip_loss

        # ==========================================
        # 2. 视频级损失
        # ==========================================
        if video_targets is not None:
            video_preds = outputs['video_outputs']  # (batch, 12)

            # 使用均方误差
            video_loss = F.mse_loss(video_preds, video_targets)

            loss_dict['video_loss'] = video_loss.item()

            if loss_type in ['video_only', 'joint']:
                total_loss = total_loss + self.VIDEO_WEIGHT * video_loss

        # ==========================================
        # 3. 片段间差异化损失
        # ==========================================
        if 'clip_outputs' in outputs and outputs['clip_outputs'].shape[1] > 1:
            clip_preds = outputs['clip_outputs']  # (batch, num_clips, 12)

            # 计算片段间的方差，鼓励片段预测之间的差异
            clip_var = clip_preds.var(dim=1).mean()  # (batch, 12) -> (batch) -> scalar
            diversity_loss = 1.0 / (clip_var + 1e-8)  # 方差越小，损失越大

            loss_dict['diversity_loss'] = diversity_loss.item()
            if loss_type in ['clip_only', 'joint']:
                total_loss = total_loss + 0.1 * diversity_loss

        # ==========================================
        # 4. 片段级和视频级差异化损失
        # ==========================================
        if 'clip_outputs' in outputs and 'video_outputs' in outputs:
            clip_preds = outputs['clip_outputs']  # (batch, num_clips, 12)
            video_preds = outputs['video_outputs']  # (batch, 12)

            # 计算片段均值与视频预测的差异，允许适度的差异
            clip_mean = clip_preds.mean(dim=1)  # (batch, 12)
            diff_loss = F.mse_loss(clip_mean, video_preds)

            loss_dict['diff_loss'] = diff_loss.item()
            if loss_type in ['joint']:
                total_loss = total_loss + 0.3 * diff_loss

        # ==========================================
        # 5. 一致性损失 (可选)
        # ==========================================
        if loss_type == 'joint_with_consistency':
            # 对于回归，一致性可以理解为：片段预测的平均值应接近视频预测
            clip_mean = outputs['clip_outputs'].mean(dim=1)  # (batch, 12)
            video_pred = outputs['video_outputs']  # (batch, 12)

            consistency_loss = F.mse_loss(clip_mean, video_pred)

            loss_dict['consistency_loss'] = consistency_loss.item()
            total_loss = total_loss + 0.5 * consistency_loss

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

    def predict(
            self,
            audio_clips,
            aside_input_ids,
            aside_attention_mask,
            danmu_input_ids,
            danmu_attention_mask,
            visual_clips
    ):
        """
        推理预测 (回归任务直接输出数值)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                audio_clips, aside_input_ids, aside_attention_mask,
                danmu_input_ids, danmu_attention_mask, visual_clips
            )

        results = {
            'video_predictions': outputs['video_outputs'],  # (batch, 12)
            'clip_predictions': outputs['clip_outputs']  # (batch, num_clips, 12)
        }

        return results

    def get_clip_representations(
            self,
            audio_clips,
            text_input_ids,
            text_attention_mask,
            visual_clips
    ):
        """
        获取片段级表示（用于分析/可视化）

        Returns:
            representations: dict
                - audio_embeddings: (batch, num_clips, audio_dim)
                - text_embedding: (batch, text_dim)
                - visual_embeddings: (batch, num_clips, visual_dim)
                - clip_features: (batch, num_clips, fusion_dim)
        """
        batch_size, num_clips = audio_clips.shape[:2]

        # 文本编码
        text_emb = self.text_encoder(text_input_ids, text_attention_mask)

        # 展平片段
        audio_flat = audio_clips.view(batch_size * num_clips, *audio_clips.shape[2:])
        visual_flat = visual_clips.view(batch_size * num_clips, *visual_clips.shape[2:])

        # 编码
        audio_emb = self.audio_encoder(audio_flat)
        visual_emb = self.visual_encoder(visual_flat)

        # 融合
        text_emb_expanded = text_emb.unsqueeze(1).expand(-1, num_clips, -1)
        text_emb_flat = text_emb_expanded.contiguous().view(batch_size * num_clips, -1)
        clip_features = self.clip_fusion(audio_emb, text_emb_flat, visual_emb)

        # 恢复形状
        audio_emb = audio_emb.view(batch_size, num_clips, -1)
        visual_emb = visual_emb.view(batch_size, num_clips, -1)
        clip_features = clip_features.view(batch_size, num_clips, -1)

        return {
            'audio_embeddings': audio_emb,
            'text_embedding': text_emb,
            'visual_embeddings': visual_emb,
            'clip_features': clip_features
        }