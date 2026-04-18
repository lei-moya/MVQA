import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LowRankMultimodalFusion(nn.Module):
    """
    低秩多模态融合 (Low-rank Multimodal Fusion, LMF)

    论文: Efficient Low-rank Multimodal Fusion with Modality-Specific Factors
    核心思想：将高维张量分解为低秩因子的乘积，降低计算复杂度
    """

    def __init__(self, audio_dim, text_dim, visual_dim,
                 fusion_dim=512, rank=4, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.fusion_dim = fusion_dim

        # 投影层：将各模态特征投影到统一维度
        self.proj_audio = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        self.proj_visual = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        # 低秩因子参数
        # 每个模态的因子矩阵：shape (rank, fusion_dim+1, output_dim)
        # +1 是为了添加偏置项
        self.audio_factor = Parameter(torch.Tensor(rank, fusion_dim + 1, fusion_dim))
        self.text_factor = Parameter(torch.Tensor(rank, fusion_dim + 1, fusion_dim))
        self.visual_factor = Parameter(torch.Tensor(rank, fusion_dim + 1, fusion_dim))

        # 融合权重和偏置
        self.fusion_weights = Parameter(torch.Tensor(1, rank))
        self.fusion_bias = Parameter(torch.Tensor(1, fusion_dim))

        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """使用Xavier初始化"""
        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.visual_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        nn.init.zeros_(self.fusion_bias)

    def forward(self, audio_emb, text_emb, visual_emb):
        """
        Args:
            audio_emb: (batch, audio_dim)
            text_emb: (batch, text_dim)
            visual_emb: (batch, visual_dim)

        Returns:
            fused: (batch, fusion_dim)
        """
        batch_size = audio_emb.size(0)

        # 1. 投影到统一维度
        a = self.proj_audio(audio_emb)  # (batch, fusion_dim)
        t = self.proj_text(text_emb)  # (batch, fusion_dim)
        v = self.proj_visual(visual_emb)  # (batch, fusion_dim)

        # 2. 添加偏置项（类似TFN的做法）
        # 添加一个常数1，用于建模模态间的独立性
        a_one = torch.ones(batch_size, 1, device=a.device)
        t_one = torch.ones(batch_size, 1, device=t.device)
        v_one = torch.ones(batch_size, 1, device=v.device)

        _a = torch.cat([a_one, a], dim=-1)  # (batch, fusion_dim+1)
        _t = torch.cat([t_one, t], dim=-1)
        _v = torch.cat([v_one, v], dim=-1)

        # 3. 低秩分解变换
        # 使用einsum高效计算：(batch, dim) x (rank, dim, out) -> (batch, rank, out)
        # 这相当于对每个rank做线性变换，然后堆叠
        fusion_a = torch.einsum('bi,rio->bro', _a, self.audio_factor)
        fusion_t = torch.einsum('bi,rio->bro', _t, self.text_factor)
        fusion_v = torch.einsum('bi,rio->bro', _v, self.visual_factor)

        # 4. 多模态交互：逐元素乘积
        # 这实现了三模态的高阶交互，但通过低秩分解降低了复杂度
        # (batch, rank, fusion_dim)
        fusion_zy = fusion_a * fusion_t * fusion_v

        # 5. 加权求和：将多个rank的结果融合
        # fusion_weights: (1, rank)
        # fusion_zy: (batch, rank, fusion_dim)
        # 输出: (batch, fusion_dim)
        # 使用 einsum 进行加权求和: (1, rank) * (batch, rank, dim) -> (batch, dim)
        output = torch.einsum('br,brd->bd', self.fusion_weights.expand(fusion_zy.shape[0], -1), fusion_zy)
        output = output + self.fusion_bias  # (batch, fusion_dim)

        # 6. Dropout
        output = self.dropout(output)

        return output


class TemporalAggregation(nn.Module):
    """
    时序聚合模块：将多个片段特征聚合为视频级特征
    支持: attention, mean, max, weighted_mean
    """

    def __init__(self, fusion_dim, aggregation_type='attention', num_heads=8, dropout=0.1):
        super().__init__()
        self.aggregation_type = aggregation_type
        self.fusion_dim = fusion_dim
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn(1, 1, fusion_dim))


    def forward(self, clip_features):
        """
        Args:
            clip_features: (batch, num_clips, fusion_dim)
        Returns:
            video_feature: (batch, fusion_dim)
        """
        batch_size, num_clips, _ = clip_features.shape

        # 扩展查询向量到批次大小
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, fusion_dim)

        # Cross-attention: query 关注所有 clip features
        attn_out, attn_weights = self.temporal_attention(
            query, clip_features, clip_features
        )  # attn_out: (batch, 1, fusion_dim)

        attn_out = self.dropout(attn_out)
        video_feature = self.norm(attn_out.squeeze(1))  # (batch, fusion_dim)

        return video_feature