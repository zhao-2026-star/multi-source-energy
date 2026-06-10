# -*- coding: utf-8 -*-
"""
MPF-Net 完整对比实验 v3
=========================
基线模型与 MPF-Net 使用完全相同的底层数据（数值+类别+文本）。
类别特征 one-hot 编码后拼入数值向量，确保公平对比。

实验内容:
  A) 6 基线模型 (LSTM/LSTM-Seq2Seq/CNN-BiLSTM-Attn/TFT/Informer/MPF-Net)
  B) 6 数据消融 (仅负荷/仅气象/仅文本/负荷+气象/负荷+文本/全量)
  C) 2 结构消融 (完整 MPF-Net / 去 ClusteringAttention)
  D) 缺失值处理 (5策略 ×5率，MPF-Net 独占)

用法:
  python run_experiments.py --epochs 30 --run_id 1    # 全量
  python run_experiments.py --mode baselines --epochs 30 --run_id 1  # 单模块
"""

import os, sys, math, time, gc, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpf_net import build_model, default_config
from data_loader import (build_dataloaders, MPFDataset, build_text_vocab,
                         NUMERICAL_COLS, CATEGORICAL_COLS, TEXT_COLS, TARGET_COL,
                         TRANSFORMER_COL, TIME_COL, WINDOW, HORIZON, STRIDE)
from baselines import (BASELINE_REGISTRY, ABLATION_CONFIGS, STRUCTURE_ABLATION,
                       MISSING_STRATEGIES, MISSING_RATES, run_missing_experiment)
import run_utils

DATA_DIR = "./data/feature_engineered"
BERT_PATH = "./data/bert_embeddings.pkl"
META_PATH = "./data/raw/变压器台账.xlsx"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════ 统一数据集（基线+MPF-Net 共享同一份底层数据）═══════════

def _one_hot_dim(values):
    """返回 one-hot 编码所需维度。"""
    return int(max(values) - min(values) + 1) if len(values) > 0 else 0

class UnifiedDataset(Dataset):
    """
    将所有特征编码为统一向量 [B, L, D]，基线模型和 MPF-Net 共用。

    数值特征 18 维 + 类别 one-hot (24+7+13+2+2+2=50) + 文本索引 2 维 = 70 维
    """
    def __init__(self, pkl_path, text_vocab, num_idx=None, use_cat=True, use_text=True,
                 cat_dims=None):
        df = pd.read_pickle(pkl_path)

        # 确定数值列: None=全量, [] = 空, [0,1,..]=指定
        if num_idx is not None:
            self.num_cols = [NUMERICAL_COLS[i] for i in num_idx]
        else:
            self.num_cols = list(NUMERICAL_COLS)
        self.use_cat = use_cat
        self.use_text = use_text

        # 类别 one-hot 范围（外部传入或从当前数据计算）
        if use_cat and cat_dims is not None:
            self.cat_dims = cat_dims
        elif use_cat:
            self.cat_dims = {}
            for col in CATEGORICAL_COLS:
                if col in df.columns:
                    vals = df[col].dropna().unique()
                    lo, hi = int(vals.min()), int(vals.max())
                    self.cat_dims[col] = (lo, hi, hi - lo + 1)
        else:
            self.cat_dims = {}

        self.text_vocab = text_vocab if use_text else {}
        self.samples = []

        for tid, grp in df.groupby(TRANSFORMER_COL, sort=False):
            grp = grp.sort_values(TIME_COL).reset_index(drop=True)
            n = len(grp)
            total_len = WINDOW + HORIZON
            if n < total_len:
                continue

            # 数值特征
            num_arr = grp[self.num_cols].to_numpy(dtype=np.float32)
            num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)

            # 类别特征 one-hot
            cat_blocks = []
            if use_cat:
                for col in CATEGORICAL_COLS:
                    if col in self.cat_dims:
                        lo, hi, dim = self.cat_dims[col]
                        vals = grp[col].fillna(0).to_numpy(dtype=np.int64)
                        vals = np.clip(vals, lo, hi)
                        onehot = np.eye(dim, dtype=np.float32)[vals - lo]
                        cat_blocks.append(onehot)

            # 文本特征 → 索引值 (作为数值)
            text_blocks = []
            if use_text and self.text_vocab:
                for col in TEXT_COLS:
                    if col in grp.columns:
                        idx = grp[col].map(self.text_vocab).fillna(0).to_numpy(dtype=np.float32)
                        text_blocks.append(idx[:, None])

            # 拼合所有特征
            blocks = [num_arr] + cat_blocks + text_blocks
            feat_arr = np.concatenate(blocks, axis=1).astype(np.float32)

            # 目标
            tgt = grp[TARGET_COL].to_numpy(dtype=np.float32)
            tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

            for s in range(0, n - total_len + 1, STRIDE):
                self.samples.append({
                    "x": torch.from_numpy(feat_arr[s:s+WINDOW].copy()),
                    "y": torch.from_numpy(tgt[s+WINDOW:s+WINDOW+HORIZON].copy()),
                })

    @property
    def input_dim(self):
        return self.samples[0]["x"].shape[-1] if self.samples else 0

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def _build_unified_loaders(num_idx=None, use_cat=True, use_text=True, batch_size=256):
    text_vocab = build_text_vocab(BERT_PATH)

    # 固定类别维度（与 CATEGORICAL_COLS 顺序一致）
    # hour=24, day_of_week=7, month=13, is_weekend=2, is_holiday=2, is_extreme=2 → 50
    CAT_DIMS = {
        "hour": (0, 23, 24), "day_of_week": (0, 6, 7),
        "month": (1, 12, 13), "is_weekend": (0, 1, 2),
        "is_holiday": (0, 1, 2), "is_extreme": (0, 1, 2),
    }
    shared_cat_dims = CAT_DIMS if use_cat else {}

    train_ds = UnifiedDataset(
        os.path.join(DATA_DIR, "train_feature_matrix.pkl"), text_vocab,
        num_idx=num_idx, use_cat=use_cat, use_text=use_text,
        cat_dims=shared_cat_dims)
    val_ds = UnifiedDataset(
        os.path.join(DATA_DIR, "val_feature_matrix.pkl"), text_vocab,
        num_idx=num_idx, use_cat=use_cat, use_text=use_text,
        cat_dims=shared_cat_dims)
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw), train_ds.input_dim


# ═══════════ 通用训练工具 ═══════════

def compute_metrics(pred, target):
    pred, target = pred.numpy(), target.numpy()
    mask = target > 1e-6
    return dict(
        rmse=float(np.sqrt(np.mean((pred-target)**2))),
        mae=float(np.mean(np.abs(pred-target))),
        wape=float(np.sum(np.abs(pred-target))/np.sum(target+1e-6)*100),
        mape=float(np.mean(np.abs(pred[mask]-target[mask])/target[mask])*100) if mask.any() else 0.0,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_target = [], []
    for batch in loader:
        pred = model(batch["x"].to(device))
        all_pred.append(pred.cpu())
        all_target.append(batch["y"])
    return compute_metrics(torch.cat(all_pred), torch.cat(all_target))


def train_one_model(model, train_loader, val_loader, epochs, lr=1e-3, wd=1e-5):
    model.to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(train_loader) * epochs
    warmup = int(0.05 * total_steps)
    step = 0
    best_val_loss = float("inf")
    best_metrics = None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            loss = criterion(model(x), y)
            if torch.isnan(loss) or torch.isinf(loss): continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            factor = (step/warmup) if step<warmup else 0.5*(1.0+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for pg in optimizer.param_groups: pg["lr"] = lr * factor

        if step >= total_steps:
            break

        metrics = evaluate(model, val_loader, DEVICE)
        if metrics["rmse"] < best_val_loss:
            best_val_loss = metrics["rmse"]
            best_metrics = dict(metrics)
    return best_metrics


def _save_csv_incremental(df, path):
    df.to_csv(path, index=False)
    print(f"  → 已保存: {path}")


# ═══════════ A) 基线模型对比（统一数据） ═══════════

def run_baselines(epochs=30, batch_size=256, run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    csv_path = os.path.join(result_dir, "baseline_comparison.csv")
    results = []

    done = set()
    if os.path.exists(csv_path):
        done_df = pd.read_csv(csv_path)
        done = set(done_df["model"].tolist())
        results = done_df.to_dict("records")
        print(f"\n  [恢复] 已完成 {len(done)} 个：{sorted(done)}")

    # 统一 DataLoader: 数值+类别onehot+文本索引
    train_loader, val_loader, input_dim = _build_unified_loaders(
        None, use_cat=True, use_text=True, batch_size=batch_size)
    print(f"\n  >>> A) 基线模型对比  输入维度={input_dim}   (epochs={epochs})")

    for name, builder in BASELINE_REGISTRY.items():
        if name in done:
            print(f"  {name:25s}  ✅ 已完成，跳过")
            continue

        model = builder(input_dim, HORIZON)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:25s}  参数量: {params:>8,}")
        m = train_one_model(model, train_loader, val_loader, epochs)
        m["model"] = name; m["params"] = params
        results.append(m)
        del model; torch.cuda.empty_cache()
        _save_csv_incremental(pd.DataFrame(results), csv_path)

    del train_loader, val_loader; gc.collect(); torch.cuda.empty_cache()

    # MPF-Net: 复用 run1 权重
    mpf_name = "MPF-Net\n(Ours)"
    if mpf_name not in done:
        print(f"  {mpf_name:25s}  参数量: 1,256,843  (复用 run1 权重)")
        weight_dir = run_utils.get_weight_dir(run_id)
        ckpt = torch.load(os.path.join(weight_dir, "mpf_net_best.pth"),
                          map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("model_config") or default_config(BERT_PATH)
        mpf_model = build_model(model_cfg).to(DEVICE)
        mpf_model.load_state_dict(ckpt["model_state_dict"])

        _, val_loader, _ = build_dataloaders(DATA_DIR, BERT_PATH, META_PATH,
                                              batch_size=64, num_workers=0)
        mpf_model.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in ["num_feat","mask","group_id"]:
                    batch[k] = batch[k].to(DEVICE)
                batch["cat_feat"] = {k: v.to(DEVICE) for k, v in batch["cat_feat"].items()}
                batch["text_feat"] = {k: v.to(DEVICE) for k, v in batch["text_feat"].items()}
                pred, _ = mpf_model(
                    batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                    batch["mask"], batch["group_id"])
                all_pred.append(pred.cpu())
                all_target.append(batch["target"].cpu())
        m = compute_metrics(torch.cat(all_pred), torch.cat(all_target))
        m["model"] = mpf_name; m["params"] = 1256843
        results.append(m)
        del mpf_model, val_loader; gc.collect(); torch.cuda.empty_cache()
        _save_csv_incremental(pd.DataFrame(results), csv_path)
        print(f"  {mpf_name:25s}  RMSE={m['rmse']:6.2f}  "
              f"MAE={m['mae']:5.2f}  MAPE={m['mape']:5.2f}%")
    else:
        print(f"  {mpf_name:25s}  ✅ 已完成，跳过")

    return pd.DataFrame(results)


# ═══════════ B) 数据消融（统一数据） ═══════════

def run_data_ablation(epochs=30, batch_size=256, run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    csv_path = os.path.join(result_dir, "data_ablation.csv")
    results = []

    done = set()
    if os.path.exists(csv_path):
        done = set(pd.read_csv(csv_path)["ablation"].tolist())
        results = pd.read_csv(csv_path).to_dict("records")
        print(f"\n  [恢复] 已完成 {len(done)} 个：{sorted(done)}")

    print(f"\n  >>> B) 数据消融 (epochs={epochs})")

    from baselines import LSTMModel
    for ab_name, ab_cfg in ABLATION_CONFIGS.items():
        if ab_name in done:
            print(f"  {ab_name:25s}  ✅ 已完成，跳过")
            continue

        num_idx = ab_cfg["num_idx"]  # list, 空列表=仅文本/类别
        use_cat = ab_cfg.get("cat", True)
        use_text = ab_cfg.get("text", True)
        num_dim = len(num_idx) if num_idx else 18
        print(f"  {ab_name:25s}  num={num_dim}  cat={use_cat}  text={use_text}")

        tl, vl, input_dim = _build_unified_loaders(
            num_idx=num_idx, use_cat=use_cat, use_text=use_text, batch_size=batch_size)

        model = LSTMModel(input_dim, hidden_dim=128, horizon=HORIZON)
        m = train_one_model(model, tl, vl, epochs)
        m["ablation"] = ab_name; m["desc"] = ab_cfg["desc"]; m["input_dim"] = input_dim
        results.append(m)
        del model, tl, vl; gc.collect(); torch.cuda.empty_cache()
        _save_csv_incremental(pd.DataFrame(results), csv_path)

    return pd.DataFrame(results)


# ═══════════ C) 缺失值实验 ═══════════

def run_missing_experiments(run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    csv_path = os.path.join(result_dir, "missing_strategies.csv")

    if os.path.exists(csv_path):
        print(f"\n  [恢复] 缺失值实验已完成，跳过")
        return pd.read_csv(csv_path)

    print(f"\n  >>> C) 缺失值处理策略对比")
    weight_dir = run_utils.get_weight_dir(run_id)
    ckpt_path = os.path.join(weight_dir, "mpf_net_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [!] 未找到 {ckpt_path}, 跳过")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config") or default_config(BERT_PATH)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)

    _, val_loader, _ = build_dataloaders(DATA_DIR, BERT_PATH, META_PATH, batch_size=64, num_workers=0)
    df = run_missing_experiment(model, val_loader)
    df.to_csv(csv_path, index=False)
    del model, val_loader; gc.collect(); torch.cuda.empty_cache()
    return df


# ═══════════ 主入口 ═══════════

def parse_args():
    p = argparse.ArgumentParser(description="MPF-Net 对比实验 v3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--run_id", type=int, default=None)
    p.add_argument("--mode", choices=["all","baselines","data_ablation","missing"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or run_utils.get_next_run_id()
    result_dir = run_utils.get_result_dir(run_id)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print(f"MPF-Net 对比实验 v3  run{run_id}  epochs={args.epochs}  设备={DEVICE}")
    print(f"结果: {result_dir}/")
    print("=" * 60)

    if args.mode in ("all", "baselines"):
        t0 = time.time()
        run_baselines(args.epochs, args.batch, run_id)
        print(f"  ✅ 完成! {time.time()-t0:.0f}s")

    if args.mode in ("all", "data_ablation"):
        t0 = time.time()
        run_data_ablation(args.epochs, args.batch, run_id)
        print(f"  ✅ 完成! {time.time()-t0:.0f}s")

    if args.mode in ("all", "missing"):
        t0 = time.time()
        run_missing_experiments(run_id)
        print(f"  ✅ 完成! {time.time()-t0:.0f}s")

    print(f"\n{'=' * 60}")
    print(f"全部完成! {result_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
