# -*- coding: utf-8 -*-
"""
MPF-Net: Multi-source Pattern Fusion Network
============================================
面向配电网多源异构数据的端到端负荷预测模型。

创新点:
  1. 聚类注意力 (MC-ANN 启发) — 可学习软聚类，替代静态 K-means
  2. 模式导向融合 (PRformer 启发) — 三种显式模式查询的季节/趋势/空间感知
  3. 缺失感知 Transformer — attention 层融入 missing mask，无需单独填补
  4. 跨模态加性融合 — 负荷数值 + 气象数值 + BERT 文本嵌入统一到 d_model

架构:
  InputEmbedding → TransformerEncoder → ClusteringAttention
    → PatternFusion → MultiTaskHead → 24h 预测
"""

import math
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. BERT 嵌入查表（离线预计算，训练时只查表 + 可学习投影）
# ═══════════════════════════════════════════════════════════════

class BERTEmbeddingLookup(nn.Module):
    """加载离线预计算的 BERT 768-dim 嵌入，通过可学习投影映射到 d_model。

    冻结 BERT 权重（不参与反向传播），仅投影层可训练。
    """

    def __init__(self, bert_path, d_model):
        super().__init__()
        with open(bert_path, "rb") as f:
            emb_dict = pickle.load(f)

        self.vocab = sorted(emb_dict.keys())
        embedding = torch.stack([torch.from_numpy(emb_dict[k]) for k in self.vocab])
        self.register_buffer("embedding", embedding)  # [vocab, 768], 冻结
        self.proj = nn.Linear(768, d_model)

    def forward(self, tokens):
        """tokens: [B, L] → 返回 [B, L, d_model]"""
        emb = F.embedding(tokens, self.embedding)  # [B, L, 768]
        return self.proj(emb)                       # [B, L, d_model]


# ═══════════════════════════════════════════════════════════════
# 2. 特征嵌入：数值 Linear + 类别 Embedding + 文本 BERT → d_model
# ═══════════════════════════════════════════════════════════════

class FeatureEmbedding(nn.Module):
    """统一嵌入三种特征类型，在 d_model 空间做加性融合。"""

    def __init__(self, numerical_cfg, categorical_cfg, bert_path, d_model):
        super().__init__()
        self.num_proj = nn.Linear(numerical_cfg["dim"], d_model)

        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(cfg["vocab"], d_model)
            for name, cfg in categorical_cfg.items()
        })

        self.text_embed = BERTEmbeddingLookup(bert_path, d_model)

    def forward(self, num_feat, cat_feat, text_feat):
        """
        num_feat:  [B, L, num_dim]
        cat_feat:  dict {name: [B, L]}      — 类别索引
        text_feat: dict {name: [B, L]}      — 文本索引
        """
        x = self.num_proj(num_feat)  # [B, L, d_model]
        for name, emb in self.cat_embeddings.items():
            x = x + emb(cat_feat[name])
        for name, tokens in text_feat.items():
            x = x + self.text_embed(tokens)
        return x


# ═══════════════════════════════════════════════════════════════
# 3. 缺失感知注意力
# ═══════════════════════════════════════════════════════════════

class MissingAwareAttention(nn.Module):
    """标准缩放点积注意力 + 缺失位置掩码（mask=0 → softmax 前置 -inf）。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x:    [B, L, d_model]
        mask: [B, L] — 1=观测, 0=缺失
        """
        B, L, D = x.shape
        H = self.n_heads
        dh = D // H

        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)  # [B, H, L, dh]
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn = attn.masked_fill(key_mask == 0, float(-1e9))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = attn @ v                               # [B, H, L, dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════
# 4. Transformer 编码器
# ═══════════════════════════════════════════════════════════════

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MissingAwareAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ═══════════════════════════════════════════════════════════════
# 5. 聚类注意力模块（MC-ANN 启发）
# ═══════════════════════════════════════════════════════════════
#
# 将每个变压器的序列表示 soft-assign 到 K 个可学习聚类中心，
# 让模型在训练中自动发现相似的负荷模式，替代静态 K-means。

class ClusteringAttention(nn.Module):
    """可学习聚类注意力。

    对每个样本，计算其表示与 K 个聚类中心之间的软分配权重，
    返回聚类加权上下文向量和分配权重。
    """

    def __init__(self, d_model, n_clusters, temp=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.temp = temp

        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, d_model))
        nn.init.xavier_uniform_(self.cluster_centers)
        self.center_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, d_model] — 编码器输出

        Returns:
          context:     [B, d_model] — 聚类加权上下文向量
          assignments: [B, K]       — 软聚类分配权重
        """
        # 序列级池化：最后一个时间步 + 均值池化的平均
        repr = (x[:, -1, :] + x.mean(dim=1)) / 2  # [B, d_model]

        # 归一化聚类中心
        centers = self.center_norm(self.cluster_centers)  # [K, d_model]

        # 软分配: [B, K]
        logits = repr @ centers.t() / (self.temp * math.sqrt(repr.size(-1)))
        logits = torch.clamp(logits, -50, 50)  # 防止溢出
        assignments = F.softmax(logits, dim=-1)
        assignments = torch.nan_to_num(assignments, nan=1.0 / self.n_clusters)

        # 聚类加权上下文
        context = assignments @ self.cluster_centers  # [B, d_model]

        return context, assignments


# ═══════════════════════════════════════════════════════════════
# 6. 多尺度时序模式融合 (Multi-Scale Temporal Pattern Fusion)
# ═══════════════════════════════════════════════════════════════
#
# 设计思路：负荷曲线存在天然的嵌套周期性——小时级波动(k=3h)、
# 半天级模式(k=12h)、日周期(k=24h)、多日趋势(k=48h/2天)。
# 用4个不同感受野的一维深度可分离卷积在时序维度上提取各尺度模式，
# 通过可学习门控自适应融合四个分支，残差连接保留原始编码信息。
#
# 与 ClusteringAttention 的关系：
#   - ClusteringAttention 做空间分组（哪些变压器的用电模式相似）
#   - PatternFusion 做时间分解（同时提取日周期/短时波动/趋势）
#   - 空间 × 时间 = 完整的时空表征

class DepthwiseTemporalConv(nn.Module):
    """深度可分离时序卷积：先分组卷积（每通道独立），再逐点卷积（通道间交互）。"""

    def __init__(self, d_model, kernel_size, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # 保持序列长度不变
        # 深度卷积：每通道独立，groups=d_model
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=padding, groups=d_model)
        # 逐点卷积：通道间交互
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D] → [B, D, L] → Conv1d → [B, D, L] → [B, L, D]
        x_t = x.transpose(1, 2)
        out = self.pointwise(self.depthwise(x_t))
        return self.norm(out.transpose(1, 2))


class PatternFusion(nn.Module):
    """多尺度时序卷积 + 门控自适应融合。

    四个分支：
      k=3  → 捕捉 1-3 小时内的短时波动（峰谷转折点）
      k=12 → 捕捉半天级模式（上午/下午/夜间负荷形态）
      k=24 → 捕捉日周期（每天的负荷曲线轮廓）
      k=48 → 捕捉 2 日趋势（持续性上升或下降倾向）
    """

    def __init__(self, d_model, n_heads=None, d_ff=None, dropout=0.1):
        super().__init__()
        kernel_sizes = [3, 12, 24, 48]

        self.branches = nn.ModuleList([
            DepthwiseTemporalConv(d_model, k) for k in kernel_sizes
        ])

        # 门控融合：根据全局池化特征学会分配各尺度权重
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, len(kernel_sizes)),
            nn.Softmax(dim=-1),
        )

        # 融合后精炼
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, L, d_model] — 编码器输出（已含聚类上下文）

        Returns:
          fused: [B, 1, d_model] — 多尺度融合后的聚合表示
        """
        B, L, D = x.shape

        # 各尺度分支 [B, L, D] → 均值池化 → [B, D]
        branch_outs = []
        for branch in self.branches:
            out = branch(x)                    # [B, L, D]
            out = out + x                      # 残差：保留原始信息
            pooled = out.mean(dim=1)            # [B, D]
            branch_outs.append(pooled)

        # 门控权重
        global_feat = x.mean(dim=1)            # [B, D]  全局池化作为门控输入
        weights = self.gate(global_feat)        # [B, 4]  每个样本各尺度权重

        # 加权融合
        fused = sum(
            w[:, k].unsqueeze(1) * branch_outs[k]
            for k in range(len(self.branches))
        )                                      # [B, D]

        # 精炼
        fused = fused.unsqueeze(1)              # [B, 1, D]
        fused = fused + self.dropout(self.ffn(self.norm(fused)))
        return fused  # [B, 1, D]


# ═══════════════════════════════════════════════════════════════
# 7. 多任务预测头
# ═══════════════════════════════════════════════════════════════

class MultiTaskHead(nn.Module):
    """共享编码层 + 每组专属预测头的多任务输出。

    共享层提取通用预测模式，每城市组独立学习个性化偏差。
    结构消融实验验证：该设计优于 CityEmbedding 注入方案。
    """

    def __init__(self, d_model, n_groups, forecast_horizon=24):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, forecast_horizon),
        )
        self.group_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, forecast_horizon),
            )
            for _ in range(n_groups)
        ])

    def forward(self, x, group_ids):
        x = x.squeeze(1)                     # [B, d_model]
        shared_out = self.shared(x)          # [B, 24]

        group_out = torch.zeros_like(shared_out)
        for g in range(len(self.group_heads)):
            mask = (group_ids == g)
            if mask.any():
                group_out[mask] = self.group_heads[g](x[mask])

        return shared_out + group_out  # [B, 24]


# ═══════════════════════════════════════════════════════════════
# 8. 完整 MPF-Net
# ═══════════════════════════════════════════════════════════════

class MPFNet(nn.Module):
    """
    Multi-source Pattern Fusion Network

    端到端模型，输入多源异构特征，输出未来 24h 逐时负荷预测。

    Pipeline:
      FeatureEmbedding → TransformerEncoder → ClusteringAttention + Skip
        → MultiTaskHead

    结构消融支持:
      设置 use_clustering=False 跳过聚类注意力
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.max_seq_len = config.get("max_seq_len", 168)

        # 结构消融开关
        self.use_clustering = config.get("use_clustering", True)
        self.use_pattern_fusion = config.get("use_pattern_fusion", True)

        # 特征嵌入
        self.embed = FeatureEmbedding(
            config["numerical"], config["categorical"],
            config["bert_path"], config["d_model"],
        )

        # 嵌入归一化（稳定训练，防止多模态加性融合后的数值爆炸）
        self.embed_norm = nn.LayerNorm(config["d_model"])

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, config["d_model"]))

        # Transformer 编码器
        self.encoder = TransformerEncoder(
            config["n_layers"], config["d_model"],
            config["n_heads"], config["d_ff"], config.get("dropout", 0.1),
        )

        # 聚类注意力
        if self.use_clustering:
            self.clustering = ClusteringAttention(
                config["d_model"], config["n_clusters"],
            )

        # 多尺度时序模式融合
        if self.use_pattern_fusion:
            self.pattern_fusion = PatternFusion(
                config["d_model"], config["n_heads"], config.get("d_ff", 512),
                config.get("dropout", 0.1),
            )

        # 预测头
        self.head = MultiTaskHead(
            config["d_model"], config["n_groups"],
            config.get("forecast_horizon", 24),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def forward(self, num_feat, cat_feat, text_feat, missing_mask, group_ids):
        """
        Args:
          num_feat:     [B, L, num_dim]       — 数值特征
          cat_feat:     dict {str: [B, L]}    — 类别特征索引
          text_feat:    dict {str: [B, L]}    — 文本特征索引
          missing_mask: [B, L]                — 1=观测, 0=缺失
          group_ids:    [B]                   — 城市组编号

        Returns:
          pred:        [B, 24]   — 未来 24h 负荷预测
          assignments: [B, K]    — 聚类软分配权重（禁用时为全零张量）
        """
        L = num_feat.size(1)
        B = num_feat.size(0)
        device = num_feat.device

        x = self.embed(num_feat, cat_feat, text_feat)  # [B, L, d_model]
        x = self.embed_norm(x)                          # 稳定化
        x = x + self.pos_embed[:, :L, :]                # [B, L, d_model]
        x = self.encoder(x, missing_mask)                 # [B, L, d_model]

        # 结构消融: 聚类注意力
        if self.use_clustering:
            context, assignments = self.clustering(x)
            x = x + context.unsqueeze(1)                  # [B, L, d_model]
        else:
            K = self.clustering.n_clusters if hasattr(self, "clustering") else 0
            assignments = torch.zeros(B, max(K, 1), device=device)

        # 结构消融: 多尺度时序模式融合
        if self.use_pattern_fusion:
            pattern = self.pattern_fusion(x)               # [B, 1, d_model]
        else:
            pattern = x.mean(dim=1, keepdim=True)          # [B, 1, d_model] (fallback)

        pred = self.head(pattern, group_ids)              # [B, 24]
        return pred, assignments


# ═══════════════════════════════════════════════════════════════
# 9. 默认配置与工厂函数
# ═══════════════════════════════════════════════════════════════

def default_config(bert_path="./data/bert_embeddings.pkl"):
    """返回 MPF-Net 默认超参数配置。"""
    return {
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "n_clusters": 5,
        "n_groups": 10,
        "max_seq_len": 168,
        "forecast_horizon": 24,
        "use_clustering": True,
        "use_pattern_fusion": True,
        "bert_path": bert_path,
        "numerical": {"dim": 18},
        "categorical": {
            "hour":        {"vocab": 24},     # [0-23]
            "day_of_week": {"vocab": 7},      # [0-6]
            "month":       {"vocab": 13},     # [1-12], index=0 留空
            "is_weekend":  {"vocab": 2},      # [0-1]
            "is_holiday":  {"vocab": 2},      # [0-1]
            "is_extreme":  {"vocab": 2},      # [0-1]
        },
    }


def build_model(config=None, bert_path="./data/bert_embeddings.pkl"):
    """创建 MPFNet 实例。"""
    if config is None:
        config = default_config(bert_path)
    return MPFNet(config)


# ═══════════════════════════════════════════════════════════════
# 10. 前向测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = default_config()
    model = build_model(cfg)

    B, L = 4, 168  # batch=4, seq_len=168 (7 天)

    num = torch.randn(B, L, cfg["numerical"]["dim"])
    cat = {
        "hour":        torch.randint(0, 24,  (B, L)),
        "day_of_week": torch.randint(0, 7,   (B, L)),
        "month":       torch.randint(1, 13,  (B, L)),  # 1-based
        "is_weekend":  torch.randint(0, 2,   (B, L)),
        "is_holiday":  torch.randint(0, 2,   (B, L)),
        "is_extreme":  torch.randint(0, 2,   (B, L)),
    }
    text = {
        "holiday_name":     torch.randint(0, 9,  (B, L)),
        "extreme_weather":  torch.randint(0, 12, (B, L)),
    }
    mask = torch.ones(B, L)
    gids = torch.randint(0, 10, (B,))

    pred, assign = model(num, cat, text, mask, gids)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MPF-Net 参数量: {total:,}")
    print(f"输入:  batch={B}, seq_len={L}")
    print(f"输出:  预测 {list(pred.shape)}  聚类 {list(assign.shape)}")
    print("前向传播测试通过!")
