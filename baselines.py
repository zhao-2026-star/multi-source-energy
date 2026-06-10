# -*- coding: utf-8 -*-
"""
MPF-Net 对比实验框架
=====================
包含两类对比实验：

  A) 模型对比 — 5 种 SOTA 基线模型 (顶刊方法)
  B) 数据消融 — 6 种特征组合方案

参考文献:
  [1] LSTM — Hochreiter & Schmidhuber, 1997 (NeurComp)
  [2] LSTM Seq2Seq — Sutskever et al., 2014 (NeurIPS)
  [3] CNN-BiLSTM-Attention — IET GT&D, 2024
  [4] Temporal Fusion Transformer — Lim et al., 2021 (Int. J. Forecasting)
  [5] Informer — Zhou et al., 2021 (AAAI)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. 基础模块
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """正弦位置编码 (Informer 等 Transformer 模型共用)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════
# 2. LSTM — 经典循环神经网络基线 [1]
# ═══════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """双层 LSTM + 全连接预测头。

    拓扑: 输入 → LSTM(128) → LSTM(128) → FC → 24h
    参考: [1] Hochreiter & Schmidhuber, 1997
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, horizon),
        )

    def forward(self, x):
        """x: [B, L, D] → [B, 24]"""
        out, _ = self.lstm(x)           # [B, L, H]
        return self.fc(out[:, -1, :])    # [B, 24]


# ═══════════════════════════════════════════════════════════════
# 3. LSTM Seq2Seq — 编码器-解码器多步预测 [2]
# ═══════════════════════════════════════════════════════════════

class LSTMSeq2Seq(nn.Module):
    """编码器-解码器结构，编码器压缩输入序列，解码器逐时间步生成预测。

    参考: [2] Sutskever et al., Sequence to Sequence Learning, NeurIPS 2014
    """

    def __init__(self, input_dim, hidden_dim=128, horizon=24):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 1)
        self.horizon = horizon

    def forward(self, x):
        """x: [B, L, D] → [B, 24]"""
        _, (h, c) = self.encoder(x)          # h, c: [1, B, H]

        dec_input = torch.zeros(x.size(0), 1, 1, device=x.device)
        outputs = []
        for _ in range(self.horizon):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.proj(out)            # [B, 1, 1]
            outputs.append(pred)
            dec_input = pred

        return torch.cat(outputs, dim=1).squeeze(-1)  # [B, 24]


# ═══════════════════════════════════════════════════════════════
# 4. CNN-BiLSTM-Attention — 混合时空注意力基线 [3]
# ═══════════════════════════════════════════════════════════════
#
# 拓扑: 输入 → Conv1D(提取局部模式) → BiLSTM(双向时序) → Attention → FC
# 参考: IET Generation, Transmission & Distribution, 2024
#       中文: VMD-CNN-BiLSTM-CBAM 《电力大数据》2024

class CNNBiLSTMAttention(nn.Module):
    """CNN 提取局部特征 → BiLSTM 建模双向时序 → 注意力加权 → 预测头。"""

    def __init__(self, input_dim, d_model=64, horizon=24):
        super().__init__()
        # CNN 模块
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        # BiLSTM
        self.bilstm = nn.LSTM(
            d_model, d_model, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1,
        )
        # 注意力
        self.attn = nn.MultiheadAttention(d_model * 2, 4, batch_first=True)
        self.attn_proj = nn.Linear(d_model * 2, d_model)
        # 预测头
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, horizon),
        )

    def forward(self, x):
        """x: [B, L, D] → [B, 24]"""
        x = x.transpose(1, 2)                      # [B, D, L]
        x = self.cnn(x).transpose(1, 2)             # [B, L, D]

        out, _ = self.bilstm(x)                     # [B, L, 2D]
        attn_out, _ = self.attn(out, out, out)      # [B, L, 2D]
        attn_out = self.attn_proj(attn_out)         # [B, L, D]

        pooled = attn_out.mean(dim=1)               # [B, D] 全局平均池化
        return self.fc(pooled)                      # [B, 24]


# ═══════════════════════════════════════════════════════════════
# 5. Temporal Fusion Transformer (TFT) — 时序融合Transformer [4]
# ═══════════════════════════════════════════════════════════════
#
# TFT 是 2024-2025 多篇顶刊对比实验中的最佳模型之一:
#   - IEEE Access 2025: TFT 在低波动数据上 MAE 最优 (1.643)
#   - arXiv 2025: 单户负荷预测 TFT RMSE 最优 (481.94)
#
# 简化实现: 特征投影 → LSTM 编码 → 自注意力 → 门控残差 → FC
# 参考: [4] Lim et al., Temporal Fusion Transformers, Int. J. Forecasting, 2021

class TemporalFusionTransformer(nn.Module):
    """轻量 TFT: 变量选择 → LSTM_Encoder → MultiheadAttention → 门控融合 → 预测头"""

    def __init__(self, input_dim, d_model=64, n_heads=4, horizon=24, dropout=0.1):
        super().__init__()
        # 变量选择网络 (GRN)
        self.variable_selection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # LSTM 编码器
        self.lstm_enc = nn.LSTM(d_model, d_model, batch_first=True, dropout=dropout)
        # 位置编码 + Transformer
        self.pos_enc = PositionalEncoding(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        # 门控残差网络 (GRN)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.gate_norm = nn.LayerNorm(d_model)
        # 预测头
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon),
        )

    def forward(self, x):
        """x: [B, L, D] → [B, 24]"""
        x = self.variable_selection(x)             # [B, L, D]
        x, _ = self.lstm_enc(x)                    # [B, L, D]

        x = self.pos_enc(x)
        attn_out, _ = self.self_attn(x, x, x)      # [B, L, D]

        # 门控残差
        gate = self.gate(attn_out)
        x = self.gate_norm(x + gate * attn_out)    # [B, L, D]
        pooled = x.mean(dim=1)                      # [B, D]
        return self.fc(pooled)                      # [B, 24]


# ═══════════════════════════════════════════════════════════════
# 6. Informer — 长序列稀疏注意力 Transformer [5]
# ═══════════════════════════════════════════════════════════════
#
# ProbSparse 自注意力核心: 用 KL 散度选择最重要的 query-key 对
# 参考: [5] Zhou et al., Informer, AAAI 2021
# 简化: 保留 ProbSparse 注意力，去掉蒸馏和生成式解码器

class ProbSparseAttention(nn.Module):
    """ProbSparse 自注意力 (简化版)：随机采样 key 子集做注意力，降低 O(L²) 复杂度。

    参考: [5] Zhou et al., Informer, AAAI 2021
    """

    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        dh = D // H

        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)  # [B, H, L, dh]
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)

        # 随机采样 key/value 子集
        u = max(1, int(self.factor * math.log(L)))
        u = min(u, L)
        k_idx = torch.randperm(L, device=x.device)[:u]         # [u]
        k_sample = k[:, :, k_idx, :]                           # [B, H, u, dh]
        v_sample = v[:, :, k_idx, :]                           # [B, H, u, dh]

        attn = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)                         # [B, H, L, u]
        out = torch.matmul(attn, v_sample)                     # [B, H, L, dh]

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor=5, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Informer(nn.Module):
    """单层 ProbSparse Transformer + 可学习位置编码 + 预测头。"""

    def __init__(self, input_dim, d_model=64, n_heads=4, d_ff=256, horizon=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 168, d_model))
        self.block = InformerBlock(d_model, n_heads, d_ff)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        x = self.block(x)
        return self.fc(x.mean(dim=1))


# ═══════════════════════════════════════════════════════════════
# 7. 模型注册表 & 工厂
# ═══════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "LSTM":         lambda D, H: LSTMModel(D, horizon=H),
    "LSTM-Seq2Seq": lambda D, H: LSTMSeq2Seq(D, horizon=H),
    "CNN-BiLSTM-Attn": lambda D, H: CNNBiLSTMAttention(D, d_model=64, horizon=H),
    "TFT":          lambda D, H: TemporalFusionTransformer(D, d_model=64, horizon=H),
    "Informer":     lambda D, H: Informer(D, d_model=64, horizon=H),
}


# ═══════════════════════════════════════════════════════════════
# 8. 数据消融方案
# ═══════════════════════════════════════════════════════════════
#
# 6 种特征组合, 每种只需指定从完整 18 维数值中选择哪些列。

ALL_NUM_COLS = [
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_48h", "lag_168h",
    "roll_mean_24h", "roll_std_24h", "roll_max_24h", "roll_min_24h",
    "capacity_kva", "load_factor",
    "温度(°C)", "相对湿度(%)", "平均风速", "降水量(mm)",
    "最高温度(°C)", "最低温度(°C)",
]

_LOAD = slice(0, 10)           # lag + roll + capacity + load_factor
_WEATHER = slice(12, 18)       # 温度/湿度/风速/降水 + 最高/最低温度

ABLATION_CONFIGS = {
    "load_only": {
        "num_idx": list(range(0, 12)),             # lag + roll + capacity + load_factor
        "cat": False,
        "text": False,
        "desc": "仅负荷历史 + 台账容量",
    },
    "weather_only": {
        "num_idx": list(range(12, 18)),
        "cat": False,
        "text": False,
        "desc": "仅气象数据",
    },
    "holiday_only": {
        "num_idx": [],
        "cat": True,
        "text": True,
        "desc": "仅节假日+极端天气文本",
    },
    "load_weather": {
        "num_idx": list(range(0, 12)) + list(range(12, 18)),
        "cat": False,
        "text": False,
        "desc": "负荷 + 气象",
    },
    "load_holiday": {
        "num_idx": list(range(0, 12)),
        "cat": True,
        "text": True,
        "desc": "负荷 + 节假日+极端天气",
    },
    "load_weather_holiday": {
        "num_idx": list(range(0, 12)) + list(range(12, 18)),
        "cat": True,
        "text": True,
        "desc": "负荷 + 气象 + 文本（全量）",
    },
}


# ═══════════════════════════════════════════════════════════════
# 9. 结构消融实验 — MPF-Net 组件解耦验证
# ═══════════════════════════════════════════════════════════════
#
# 目的: 量化 ClusteringAttention 和 PatternFusion 各自的贡献
# 方式: 构造 MPF-Net 的 4 种变体，在相同数据上训练和评估
#
# 用法:
#   from mpf_net import build_model, default_config
#   for name, cfg in STRUCTURE_ABLATION.items():
#       model_cfg = default_config(bert_path)
#       model_cfg.update(cfg["overrides"])
#       model = build_model(model_cfg)

STRUCTURE_ABLATION = {
    "mpfnet_full": {
        "overrides": {"use_clustering": True},
        "desc": "完整 MPF-Net (ClusteringAttention + CityEmbedding)",
    },
    "mpfnet_no_clustering": {
        "overrides": {"use_clustering": False},
        "desc": "去掉 ClusteringAttention → 仅 Encoder + 预测头",
    },
}


# ═══════════════════════════════════════════════════════════════
# 10. 缺失值处理策略对比实验
# ═══════════════════════════════════════════════════════════════
#
# 目的: 证明 MissingAwareAttention 优于传统填充方法
# 方式:
#   1. 在验证集上以不同比率人工注入额外缺失 (0%, 5%, 10%, 25%, 50%)
#   2. 每种缺失率下用 5 种策略处理
#   3. 比较各策略的 RMSE/MAE vs 缺失率
#   4. 预期: 缺失率越高, MissingAware 优势越明显
#
# 用法:
#   results = run_missing_experiment(model, val_loader)
#   # results 是 DataFrame, 可直接传给 visualize.py 画图

import numpy as np
from scipy import interpolate
import os

MISSING_RATES = [0.0, 0.05, 0.10, 0.25, 0.50]

MISSING_STRATEGIES = {
    "missing_aware": {
        "desc": "MissingAwareAttention (mask→-∞)",
        "impute_fn": None,         # 不填充, 由模型 mask 处理
        "pass_mask": True,
    },
    "zero_fill": {
        "desc": "零值填充",
        "impute_fn": lambda x: np.nan_to_num(x, nan=0.0),
        "pass_mask": False,
    },
    "forward_fill": {
        "desc": "前向填充 (LOCF)",
        "impute_fn": lambda x: _ffill(x),
        "pass_mask": False,
    },
    "mean_fill": {
        "desc": "全局均值填充",
        "impute_fn": lambda x: _mean_fill(x),
        "pass_mask": False,
    },
    "linear_interp": {
        "desc": "线性插值",
        "impute_fn": lambda x: _linear_interp(x),
        "pass_mask": False,
    },
}


def _ffill(arr):
    """一维前向填充 (LOCF)。"""
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def _mean_fill(arr):
    """全局均值填充。"""
    m = np.nanmean(arr)
    return np.nan_to_num(arr, nan=m if not np.isnan(m) else 0.0)


def _linear_interp(arr):
    """一维线性插值。"""
    n = len(arr)
    x = np.arange(n)
    mask = np.isnan(arr)
    if mask.all() or (~mask).sum() < 2:
        return np.nan_to_num(arr, nan=0.0)
    f = interpolate.interp1d(x[~mask], arr[~mask], kind="linear",
                              bounds_error=False, fill_value="extrapolate")
    return f(x)


@torch.no_grad()
def run_missing_experiment(model, val_loader, missing_rates=None,
                           strategies=None, seed=42):
    """
    对验证集注入不同比率的人工缺失，对比各策略的预测误差。

    参数:
      model:       训练好的 MPFNet
      val_loader:  验证集 DataLoader（保持原样, 含自然缺失）
      missing_rates: 要测试的缺失率列表, 默认 [0, 0.05, 0.1, 0.25, 0.5]
      strategies:    缺失处理策略名称列表, 默认用全部 5 种

    返回:
      pd.DataFrame, columns=[missing_rate, strategy, rmse, mae, mape]
        可直接传给 visualize.py 画折线图
    """
    import pandas as pd

    if missing_rates is None:
        missing_rates = MISSING_RATES
    if strategies is None:
        strategies = list(MISSING_STRATEGIES.keys())

    model.eval()
    device = next(model.parameters()).device

    rng = np.random.default_rng(seed)
    results = []

    for rate in missing_rates:
        print(f"\n  ── 缺失率 {rate*100:.0f}% ──")

        for sname in strategies:
            strat = MISSING_STRATEGIES[sname]
            all_pred, all_target = [], []

            for batch in val_loader:
                num_feat = batch["num_feat"].clone()     # [B, L, D_num]
                mask = batch["mask"].clone()              # [B, L]
                b, l = num_feat.shape[:2]

                # 前两列是 lag_1h, lag_2h（原始负荷滞后量）
                # 人工注入额外缺失：只在负荷相关列上注入
                if rate > 0:
                    # 对 'lag_1h' 位置 (第0列) 注入缺失
                    inject = rng.random((b, l)) < rate
                    num_feat[inject] = float("nan")
                    mask[inject] = 0.0

                # 按策略处理
                if strat["impute_fn"] is not None:
                    for i in range(b):
                        for j in range(num_feat.size(-1)):
                            num_feat[i, :, j] = torch.from_numpy(
                                strat["impute_fn"](num_feat[i, :, j].cpu().numpy())
                            )

                # 送进模型
                pass_mask = mask if strat["pass_mask"] else None
                batch["num_feat"] = num_feat.to(device)
                batch["cat_feat"] = {k: v.to(device) for k, v in batch["cat_feat"].items()}
                batch["text_feat"] = {k: v.to(device) for k, v in batch["text_feat"].items()}
                batch["group_id"] = batch["group_id"].to(device)

                pred, _ = model(
                    num_feat.to(device),
                    batch["cat_feat"],
                    batch["text_feat"],
                    pass_mask.to(device) if pass_mask is not None else None,
                    batch["group_id"],
                )
                all_pred.append(pred.cpu())
                all_target.append(batch["target"].cpu())

            pred_cat = torch.cat(all_pred)
            targ_cat = torch.cat(all_target)
            metrics = compute_metrics(pred_cat.numpy(), targ_cat.numpy())

            results.append({
                "missing_rate": rate,
                "strategy": sname,
                "strategy_desc": strat["desc"],
                "rmse": float(metrics["RMSE (kW)"].replace(" kW", "")),
                "mae":  float(metrics["MAE (kW)"].replace(" kW", "")),
                "mape": float(metrics["MAPE (%)"].replace("%", "")),
            })
            print(f"    {strat['desc']:30s}  RMSE={results[-1]['rmse']:.2f}  "
                  f"MAE={results[-1]['mae']:.2f}  MAPE={results[-1]['mape']:.2f}%")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 辅助：兼容训练脚本中的 compute_metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(pred, target):
    mask = target > 1e-6
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mae = np.mean(np.abs(pred - target))
    wape = np.sum(np.abs(pred - target)) / np.sum(target + 1e-6) * 100
    if mask.any():
        mape = np.mean(np.abs(pred[mask] - target[mask]) / target[mask]) * 100
    else:
        mape = 0.0
    return {"RMSE (kW)": f"{rmse:.2f}",
            "MAE (kW)":  f"{mae:.2f}",
            "MAPE (%)":  f"{mape:.2f}",
            "WAPE (%)":  f"{wape:.2f}"}


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPF-Net 对比实验")
    parser.add_argument("--run_id", type=int, default=None, help="实验编号（默认1）")
    parser.add_argument("--mode", choices=["test", "baselines", "ablation", "missing", "all"],
                        default="test", help="实验模式")
    args = parser.parse_args()

    run_id = args.run_id or 1
    result_dir = run_utils.get_result_dir(run_id)
    weight_dir = run_utils.get_weight_dir(run_id)
    os.makedirs(result_dir, exist_ok=True)

    B, L, D = 4, 168, 18
    x = torch.randn(B, L, D)

    # ── 基线模型测试 ──
    print("=" * 60)
    print("A) 基线模型前向测试")
    print("=" * 60)
    for name, builder in BASELINE_REGISTRY.items():
        model = builder(D, 24)
        out = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        loss = F.mse_loss(out, torch.randn(B, 24))
        loss.backward()
        print(f"  {name:20s} 输出 {list(out.shape)}  参数量 {params:>8,}  backward OK")

    # ── 数据消融 ──
    print(f"\n{'=' * 60}")
    print("B) 数据消融方案")
    print("=" * 60)
    for k, v in ABLATION_CONFIGS.items():
        print(f"  {k:30s} → {v['desc']:20s}  num_idx={len(v['num_idx'])}")

    # ── 结构消融测试 ──
    print(f"\n{'=' * 60}")
    print("C) 结构消融 (MPF-Net 变体)")
    print("=" * 60)
    from mpf_net import build_model, default_config
    base_cfg = default_config()
    sim_num = torch.randn(B, L, base_cfg["numerical"]["dim"])
    sim_cat = {
        "hour":        torch.randint(0, 24,  (B, L)),
        "day_of_week": torch.randint(0, 7,   (B, L)),
        "month":       torch.randint(1, 13,  (B, L)),
        "is_weekend":  torch.randint(0, 2,   (B, L)),
        "is_holiday":  torch.randint(0, 2,   (B, L)),
        "is_extreme":  torch.randint(0, 2,   (B, L)),
    }
    sim_text = {"holiday_name": torch.randint(0, 9, (B, L)),
                "extreme_weather": torch.randint(0, 12, (B, L))}
    sim_mask = torch.ones(B, L)
    sim_gid = torch.randint(0, 10, (B,))

    for name, sa in STRUCTURE_ABLATION.items():
        cfg = default_config()
        cfg.update(sa["overrides"])
        model = build_model(cfg)
        pred, assign = model(sim_num, sim_cat, sim_text, sim_mask, sim_gid)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        loss = F.mse_loss(pred, torch.randn(B, 24))
        loss.backward()
        print(f"  {name:25s} 预测 {list(pred.shape)} 聚类 {list(assign.shape)}  "
              f"参数量 {params:>8,}  backward OK")
        print(f"      → {sa['desc']}")

    # ── 缺失策略 ──
    print(f"\n{'=' * 60}")
    print("D) 缺失值处理策略")
    print("=" * 60)
    for sname, sopt in MISSING_STRATEGIES.items():
        print(f"  {sname:20s} → {sopt['desc']}")
    print(f"\n缺失率测试: {MISSING_RATES}")
    print("\n基线模型 + 消融框架测试通过!")

    # 保存测试结果摘要
    with open(os.path.join(result_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"实验 run{run_id} 模型框架验证通过\n")
        f.write(f"注册基线模型: {list(BASELINE_REGISTRY.keys())}\n")
        f.write(f"数据消融方案: {len(ABLATION_CONFIGS)} 种\n")
        f.write(f"结构消融方案: {len(STRUCTURE_ABLATION)} 种\n")
        f.write(f"缺失值策略: {len(MISSING_STRATEGIES)} 种\n")
