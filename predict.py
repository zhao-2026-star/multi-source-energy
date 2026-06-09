# -*- coding: utf-8 -*-
"""
MPF-Net 预测脚本
=================
加载训练好的模型检查点，在验证集上做预测，输出结果文件。

用法:
  python predict.py                                          # 默认验证集
  python predict.py --split train                            # 训练集
  python predict.py --checkpoint checkpoints/mpf_net_best.pth --batch 512

输出:
  predictions/predictions_{split}.pkl    — 完整预测结果 (DataFrame)
  predictions/predictions_{split}.csv    — 预览 (前 10000 行)
  predictions/predictions_metrics.txt    — 评价指标
"""

import os
import sys
import argparse
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpf_net import build_model, default_config
from data_loader import build_dataloaders, MPFDataset, build_text_vocab
from data_loader import NUMERICAL_COLS, CATEGORICAL_COLS, TEXT_COLS, TARGET_COL


DATA_DIR = "./data/feature_engineered"
BERT_PATH = "./data/bert_embeddings.pkl"
META_PATH = "./data/raw/变压器台账.xlsx"
CKPT_DIR = "./checkpoints"
OUT_DIR = "./predictions"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 评估指标
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
# 逐变压器评估
# ═══════════════════════════════════════════════════════════════

def evaluate_per_transformer(pred_df, target_col="target", pred_col="prediction"):
    """输出每个变压器的独立指标。"""
    results = []
    for tid, grp in pred_df.groupby("变压器编号"):
        p = grp[pred_col].values
        t = grp[target_col].values
        m = compute_metrics(p, t)
        m["变压器编号"] = tid
        m["样本数"] = len(p)
        m["平均负荷"] = f"{t.mean():.1f}"
        results.append(m)
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 逐小时评估（分析一天中哪些时段预测最难）
# ═══════════════════════════════════════════════════════════════

def evaluate_by_hour(pred_df, horizon=24):
    """按预测步长（0-23 小时）分别计算指标。"""
    rows = []
    for h in range(horizon):
        mask = pred_df["step"] == h
        if mask.any():
            sub = pred_df[mask]
            m = compute_metrics(sub["prediction"].values, sub["target"].values)
            m["hour"] = h
            rows.append(m)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 主预测函数
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, loader, split="val", horizon=24):
    """
    遍历 DataLoader，收集预测值和目标值。

    Returns:
      pred_df: DataFrame 包含 预测值, 目标值, 步长
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_targets = []
    all_steps = []

    for batch in loader:
        batch["num_feat"] = batch["num_feat"].to(device)
        batch["cat_feat"] = {k: v.to(device) for k, v in batch["cat_feat"].items()}
        batch["text_feat"] = {k: v.to(device) for k, v in batch["text_feat"].items()}
        batch["mask"] = batch["mask"].to(device)
        batch["group_id"] = batch["group_id"].to(device)

        pred, _ = model(
            batch["num_feat"], batch["cat_feat"], batch["text_feat"],
            batch["mask"], batch["group_id"],
        )

        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch["target"].cpu().numpy())
        all_steps.append(np.tile(np.arange(horizon), (pred.size(0), 1)))

    # 构造 DataFrame
    df = pd.DataFrame({
        "prediction": np.concatenate(all_preds).ravel(),
        "target": np.concatenate(all_targets).ravel(),
        "step": np.concatenate(all_steps).ravel(),
    })

    return df


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MPF-Net 预测")
    p.add_argument("--checkpoint", default=os.path.join(CKPT_DIR, "mpf_net_best.pth"))
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"MPF-Net 预测   {args.split}集")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[1/4] 加载 DataLoader...")
    train_loader, val_loader, _ = build_dataloaders(
        DATA_DIR, BERT_PATH, META_PATH,
        batch_size=args.batch, num_workers=args.workers,
    )
    loader = train_loader if args.split == "train" else val_loader

    # ── 加载模型 ──
    print(f"\n[2/4] 加载模型检查点: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config") or default_config()
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"      设备: {device}  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 预测 ──
    print(f"\n[3/4] 开始预测 ({len(loader)} batches)...")
    t0 = time.time()
    df = predict(model, loader, args.split)
    dur = time.time() - t0
    print(f"      完成! 耗时 {dur:.0f}s ({len(df):,} 行)")

    # ── 评估 ──
    print(f"\n[4/4] 评估指标:")
    print(f"      {'=' * 50}")
    overall = compute_metrics(df["prediction"].values, df["target"].values)
    for k, v in overall.items():
        print(f"      {k:20s} {v}")

    # ── 逐小时评估 ──
    hour_metrics = evaluate_by_hour(df)
    worst_hour = hour_metrics.loc[hour_metrics["RMSE (kW)"].str.replace(" kW", "").astype(float).idxmax()]
    best_hour = hour_metrics.loc[hour_metrics["RMSE (kW)"].str.replace(" kW", "").astype(float).idxmin()]
    print(f"\n      最难预测时段: {worst_hour['hour']:.0f}:00  RMSE={worst_hour['RMSE (kW)']}")
    print(f"      最易预测时段: {best_hour['hour']:.0f}:00  RMSE={best_hour['RMSE (kW)']}")

    # ── 保存 ──
    timestamp = datetime.now().strftime("%m%d_%H%M")
    pkl_name = f"predictions_{args.split}_{timestamp}.pkl"
    csv_name = f"predictions_{args.split}_{timestamp}.csv"
    metrics_name = "predictions_metrics.txt"

    df.to_pickle(os.path.join(OUT_DIR, pkl_name))
    df.head(10000).to_csv(os.path.join(OUT_DIR, csv_name), index=False, encoding="utf-8-sig")

    with open(os.path.join(OUT_DIR, metrics_name), "w", encoding="utf-8") as f:
        f.write(f"MPF-Net 预测指标 ({args.split}集)\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"总样本: {len(df):,}\n")
        for k, v in overall.items():
            f.write(f"{k:20s} {v}\n")
        f.write(f"\n逐小时指标:\n")
        for _, row in hour_metrics.iterrows():
            f.write(f"  {int(row['hour']):02d}:00  "
                    f"RMSE={row['RMSE (kW)']:>8}  MAE={row['MAE (kW)']:>8}  MAPE={row['MAPE (%)']:>8}\n")

    print(f"\n      结果保存: {OUT_DIR}/{pkl_name}")
    print(f"      预览:     {OUT_DIR}/{csv_name}")
    print(f"      指标:     {OUT_DIR}/{metrics_name}")
    print(f"\n{'=' * 60}")
    print("预测完成!")


if __name__ == "__main__":
    main()
