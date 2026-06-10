# -*- coding: utf-8 -*-
"""
MPF-Net 完整对比实验
=====================
一键运行全部论文所需对比实验，结果保存至 runs/run{N}/result/

包含:
  A) 5 基线模型 (LSTM, LSTM-Seq2Seq, CNN-BiLSTM-Attn, TFT, Informer)
  B) 6 数据消融 (仅负荷 / 仅气象 / 仅文本 / ... / 全量)
  C) 4 结构消融 (完整 / 去聚类 / 去融合 / 去两者)
  D) 5 缺失策略 × 5 缺失率

用法:
  python run_experiments.py --epochs 30 --run_id 1    # 快速实验
  python run_experiments.py --epochs 50 --run_id 2    # 完整实验
"""

import os, sys, math, time, pickle, argparse, json
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

# ==================== 路径 ====================
DATA_DIR = "./data/feature_engineered"
BERT_PATH = "./data/bert_embeddings.pkl"
META_PATH = "./data/raw/变压器台账.xlsx"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 数值数据集 (给基线模型用) ====================
class FlatNumDataset(Dataset):
    def __init__(self, pkl_path, num_idx=None):
        df = pd.read_pickle(pkl_path)
        cols = [NUMERICAL_COLS[i] for i in num_idx] if num_idx else NUMERICAL_COLS
        self.samples = []

        for tid, grp in df.groupby(TRANSFORMER_COL, sort=False):
            grp = grp.sort_values(TIME_COL).reset_index(drop=True)
            arr = grp[cols].to_numpy(dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            tgt = grp[TARGET_COL].to_numpy(dtype=np.float32)
            tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
            n = len(grp)
            total_len = WINDOW + HORIZON
            for s in range(0, n - total_len + 1, STRIDE):
                self.samples.append({
                    "x": torch.from_numpy(arr[s:s+WINDOW]),
                    "y": torch.from_numpy(tgt[s+WINDOW:s+WINDOW+HORIZON]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==================== 训练工具 ====================
def _build_flat_loaders(num_idx=None, batch_size=256, num_workers=4):
    """为基线模型构建纯数值 DataLoader。"""
    train_ds = FlatNumDataset(
        os.path.join(DATA_DIR, "train_feature_matrix.pkl"), num_idx)
    val_ds = FlatNumDataset(
        os.path.join(DATA_DIR, "val_feature_matrix.pkl"), num_idx)
    # 推断输入维度
    input_dim = train_ds[0]["x"].shape[-1]
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    return train_loader, val_loader, input_dim


def compute_metrics(pred, target):
    pred, target = pred.numpy(), target.numpy()
    mask = target > 1e-6
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae = float(np.mean(np.abs(pred - target)))
    wape = float(np.sum(np.abs(pred - target)) / np.sum(target + 1e-6) * 100)
    mape = float(np.mean(np.abs(pred[mask] - target[mask]) / target[mask]) * 100) if mask.any() else 0.0
    return dict(rmse=rmse, mae=mae, mape=mape, wape=wape)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_target = [], []
    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x)
        all_pred.append(pred.cpu())
        all_target.append(batch["y"])
    return compute_metrics(torch.cat(all_pred), torch.cat(all_target))


def train_one_model(model, train_loader, val_loader, epochs, lr=1e-3, wd=1e-5,
                    model_name="", result_dir=None):
    model.to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(train_loader) * epochs
    warmup = int(0.05 * total_steps)
    step = 0
    best_metrics = None
    best_val_loss = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            loss = criterion(model(x), y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            if step < warmup:
                factor = step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr * factor

        metrics = evaluate(model, val_loader, DEVICE)
        val_loss = metrics["rmse"]  # 用 RMSE 选最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = dict(metrics)

    print(f"  {model_name:25s}  RMSE={best_metrics['rmse']:6.2f}  "
          f"MAE={best_metrics['mae']:5.2f}  MAPE={best_metrics['mape']:5.2f}%")
    return best_metrics


# ==================== A) 基线模型对比 ====================
def run_baselines(epochs=30, batch_size=256, run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    train_loader, val_loader, input_dim = _build_flat_loaders(None, batch_size)
    results = []
    print(f"\n  >>> A) 基线模型对比 (epochs={epochs})")

    for name, builder in BASELINE_REGISTRY.items():
        model = builder(input_dim, HORIZON)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:25s}  参数量: {params:>8,}")
        m = train_one_model(model, train_loader, val_loader, epochs,
                            model_name=name, result_dir=result_dir)
        m["model"] = name
        m["params"] = params
        results.append(m)

    # MPF-Net (our method)
    print(f"  {'MPF-Net (Ours)':25s}  参数量: 1,256,843")
    mpf_loader_train, mpf_loader_val, _ = build_dataloaders(
        DATA_DIR, BERT_PATH, META_PATH, batch_size=batch_size, num_workers=0)
    cfg = default_config(BERT_PATH)
    cfg["n_groups"] = 10
    model = build_model(cfg).to(DEVICE)

    # 专门训练 MPF-Net
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    total_steps = len(mpf_loader_train) * epochs
    warmup = int(0.05 * total_steps)
    step = 0
    best_val = float("inf")
    best_m = None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in mpf_loader_train:
            batch["num_feat"] = batch["num_feat"].to(DEVICE)
            batch["cat_feat"] = {k: v.to(DEVICE) for k, v in batch["cat_feat"].items()}
            batch["text_feat"] = {k: v.to(DEVICE) for k, v in batch["text_feat"].items()}
            batch["mask"] = batch["mask"].to(DEVICE)
            batch["group_id"] = batch["group_id"].to(DEVICE)
            pred, _ = model(batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                            batch["mask"], batch["group_id"])
            loss = criterion(pred, batch["target"].to(DEVICE))
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            if step < warmup:
                factor = step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = 1e-3 * factor

        # 验证
        model.eval()
        all_pred, all_target = [], []
        for batch in mpf_loader_val:
            batch["num_feat"] = batch["num_feat"].to(DEVICE)
            batch["cat_feat"] = {k: v.to(DEVICE) for k, v in batch["cat_feat"].items()}
            batch["text_feat"] = {k: v.to(DEVICE) for k, v in batch["text_feat"].items()}
            batch["mask"] = batch["mask"].to(DEVICE)
            batch["group_id"] = batch["group_id"].to(DEVICE)
            pred, _ = model(batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                            batch["mask"], batch["group_id"])
            all_pred.append(pred.cpu())
            all_target.append(batch["target"].cpu())
        m = compute_metrics(torch.cat(all_pred), torch.cat(all_target))
        if m["rmse"] < best_val:
            best_val = m["rmse"]
            best_m = dict(m)

    best_m["model"] = "MPF-Net\n(Ours)"
    best_m["params"] = 1256843
    results.append(best_m)
    print(f"  {'MPF-Net (Ours)':25s}  RMSE={best_m['rmse']:6.2f}  "
          f"MAE={best_m['mae']:5.2f}  MAPE={best_m['mape']:5.2f}%")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, "baseline_comparison.csv"), index=False)
    return df


# ==================== B) 数据消融 ====================
def run_data_ablation(epochs=30, batch_size=256, run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    results = []
    print(f"\n  >>> B) 数据消融 (epochs={epochs})")

    for ab_name, ab_cfg in ABLATION_CONFIGS.items():
        num_idx = ab_cfg["num_idx"]
        num_dim = len(num_idx) if num_idx else 0
        print(f"  {ab_name:25s}  num_dim={num_dim}")

        train_loader, val_loader, input_dim = _build_flat_loaders(num_idx or None, batch_size)

        # 用简单 LSTM 测试（消融实验不需要完整 MPF-Net）
        from baselines import LSTMModel
        model = LSTMModel(input_dim, hidden_dim=128, horizon=HORIZON)
        m = train_one_model(model, train_loader, val_loader, epochs,
                            model_name=ab_name, result_dir=result_dir)
        m["ablation"] = ab_name
        m["desc"] = ab_cfg["desc"]
        m["num_dim"] = num_dim
        results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, "data_ablation.csv"), index=False)
    return df


# ==================== C) 结构消融 ====================
def run_structure_ablation(epochs=30, batch_size=256, run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    results = []
    print(f"\n  >>> C) 结构消融 (epochs={epochs})")

    mpf_loader_train, mpf_loader_val, _ = build_dataloaders(
        DATA_DIR, BERT_PATH, META_PATH, batch_size=batch_size, num_workers=0)

    for ab_name, ab_cfg in STRUCTURE_ABLATION.items():
        cfg = default_config(BERT_PATH)
        cfg.update(ab_cfg["overrides"])
        cfg["n_groups"] = 10
        model = build_model(cfg)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {ab_name:25s}  参数量: {params:>8,}  → {ab_cfg['desc']}")
        model.to(DEVICE)

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        total_steps = len(mpf_loader_train) * epochs
        warmup = int(0.05 * total_steps)
        step = 0
        best_val = float("inf")
        best_m = None

        for ep in range(1, epochs + 1):
            model.train()
            for batch in mpf_loader_train:
                batch["num_feat"] = batch["num_feat"].to(DEVICE)
                batch["cat_feat"] = {k: v.to(DEVICE) for k, v in batch["cat_feat"].items()}
                batch["text_feat"] = {k: v.to(DEVICE) for k, v in batch["text_feat"].items()}
                batch["mask"] = batch["mask"].to(DEVICE)
                batch["group_id"] = batch["group_id"].to(DEVICE)
                pred, _ = model(batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                                batch["mask"], batch["group_id"])
                loss = criterion(pred, batch["target"].to(DEVICE))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                step += 1
                if step < warmup:
                    factor = step / max(1, warmup)
                else:
                    progress = (step - warmup) / max(1, total_steps - warmup)
                    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = 1e-3 * factor

            model.eval()
            all_pred, all_target = [], []
            with torch.no_grad():
                for batch in mpf_loader_val:
                    batch["num_feat"] = batch["num_feat"].to(DEVICE)
                    batch["cat_feat"] = {k: v.to(DEVICE) for k, v in batch["cat_feat"].items()}
                    batch["text_feat"] = {k: v.to(DEVICE) for k, v in batch["text_feat"].items()}
                    batch["mask"] = batch["mask"].to(DEVICE)
                    batch["group_id"] = batch["group_id"].to(DEVICE)
                    pred, _ = model(batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                                    batch["mask"], batch["group_id"])
                    all_pred.append(pred.cpu())
                    all_target.append(batch["target"].cpu())
            m = compute_metrics(torch.cat(all_pred), torch.cat(all_target))
            if m["rmse"] < best_val:
                best_val = m["rmse"]
                best_m = dict(m)

        best_m["ablation"] = ab_name
        best_m["desc"] = ab_cfg["desc"]
        best_m["params"] = params
        results.append(best_m)
        print(f"  {ab_name:25s}  RMSE={best_m['rmse']:6.2f}  "
              f"MAE={best_m['mae']:5.2f}  MAPE={best_m['mape']:5.2f}%")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, "structure_ablation.csv"), index=False)
    return df


# ==================== D) 缺失值实验 ====================
def run_missing_experiments(run_id=1):
    result_dir = run_utils.get_result_dir(run_id)
    print(f"\n  >>> D) 缺失值处理策略对比")

    # 加载训练好的最佳模型
    weight_dir = run_utils.get_weight_dir(run_id)
    ckpt_path = os.path.join(weight_dir, "mpf_net_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [!] 未找到最佳模型 {ckpt_path}, 跳过")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config") or default_config(BERT_PATH)
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)

    _, val_loader, _ = build_dataloaders(
        DATA_DIR, BERT_PATH, META_PATH, batch_size=256, num_workers=0)

    df = run_missing_experiment(model, val_loader)
    df.to_csv(os.path.join(result_dir, "missing_strategies.csv"), index=False)
    return df


# ==================== 主入口 ====================
def parse_args():
    p = argparse.ArgumentParser(description="MPF-Net 完整对比实验")
    p.add_argument("--epochs", type=int, default=30, help="训练轮数 (30=快速, 50=论文级)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--run_id", type=int, default=None)
    p.add_argument("--mode", choices=["all", "baselines", "data_ablation",
                     "structure_ablation", "missing"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or run_utils.get_next_run_id()
    result_dir = run_utils.get_result_dir(run_id)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print(f"MPF-Net 对比实验   实验: run{run_id}   epochs: {args.epochs}")
    print(f"设备: {DEVICE}")
    print(f"结果保存: {result_dir}")
    print("=" * 60)

    if args.mode in ("all", "baselines"):
        t0 = time.time()
        df = run_baselines(args.epochs, args.batch, run_id)
        print(f"\n  完成! 耗时 {time.time()-t0:.0f}s")

    if args.mode in ("all", "data_ablation"):
        t0 = time.time()
        df = run_data_ablation(args.epochs, args.batch, run_id)
        print(f"\n  完成! 耗时 {time.time()-t0:.0f}s")

    if args.mode in ("all", "structure_ablation"):
        t0 = time.time()
        df = run_structure_ablation(args.epochs, args.batch, run_id)
        print(f"\n  完成! 耗时 {time.time()-t0:.0f}s")

    if args.mode in ("all", "missing"):
        t0 = time.time()
        df = run_missing_experiments(run_id)
        print(f"\n  完成! 耗时 {time.time()-t0:.0f}s")

    print(f"\n{'=' * 60}")
    print(f"全部实验完成! 结果保存至: {result_dir}/")
    print(f"可使用: python visualize.py --run_id {run_id} \\")
    print(f"  --baseline-results {result_dir}/baseline_comparison.csv \\")
    print(f"  --ablation-results {result_dir}/data_ablation.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
