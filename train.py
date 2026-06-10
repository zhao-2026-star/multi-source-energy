# -*- coding: utf-8 -*-
"""
MPF-Net 训练脚本
=================
端到端训练配电网负荷预测模型。

用法:
  python train.py                          # 默认配置训练
  python train.py --epochs 50 --batch 512  # 自定义超参

输出:
  checkpoints/  — 最佳模型检查点
  logs/         — 训练日志 (CSV)
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout.reconfigure(encoding="utf-8")

# 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpf_net import build_model, default_config
from data_loader import build_dataloaders
import run_utils


# ═══════════════════════════════════════════════════════════════
# 常量路径
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "./data/feature_engineered"
BERT_PATH = "./data/bert_embeddings.pkl"
META_PATH = "./data/raw/变压器台账.xlsx"


# ═══════════════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════════════

def compute_metrics(pred, target):
    """
    pred, target: [B, H] — H=24
    返回 dict: rmse, mae, mape
    """
    # 过滤 target=0 的位置避免 MAPE 除零
    mask = target > 1e-6
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    if mask.any():
        mape = (torch.abs(pred[mask] - target[mask]) / target[mask]).mean().item() * 100
    else:
        mape = 0.0
    return {"rmse": rmse, "mae": mae, "mape": mape}


# ═══════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, model_cfg=None,
                 ckpt_dir="./checkpoints", log_dir="./logs"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_cfg = model_cfg
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        total_steps = len(train_loader) * config["epochs"]

        # 学习率预热 + Cosine 衰减
        # Transformer 需要预热防止早期梯度爆炸
        self.warmup_steps = int(0.05 * total_steps)  # 前 5% 步数线性预热
        self.current_step = 0
        self.total_steps = total_steps

        self.best_val_loss = float("inf")
        self.history = []

    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor)
                else {kk: vv.to(self.device) for kk, vv in v.items()}
                for k, v in batch.items()}

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        start = time.time()

        for i, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            pred, assign = self.model(
                batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                batch["mask"], batch["group_id"],
            )
            loss = self.criterion(pred, batch["target"])

            # 跳过 NaN/Inf batch（防止单个异常样本导致训练崩溃）
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    [!] 跳过异常 batch {i + 1}  loss={loss.item()}")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

            # 自定义 LR 调度: warmup → cosine decay
            self.current_step += 1
            if self.current_step < self.warmup_steps:
                # 线性预热: 0 → lr
                factor = self.current_step / max(1, self.warmup_steps)
            else:
                # Cosine 衰减: lr → 0
                progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config["lr"] * factor

            total_loss += loss.item()

            if (i + 1) % max(1, num_batches // 10) == 0:
                pct = (i + 1) / num_batches * 100
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"    [{pct:3.0f}%] loss={loss.item():.4f}  lr={lr:.2e}")

        return total_loss / num_batches, time.time() - start

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_pred, all_target = [], []

        for batch in self.val_loader:
            batch = self._to_device(batch)
            pred, _ = self.model(
                batch["num_feat"], batch["cat_feat"], batch["text_feat"],
                batch["mask"], batch["group_id"],
            )
            loss = self.criterion(pred, batch["target"])
            total_loss += loss.item()
            all_pred.append(pred.cpu())
            all_target.append(batch["target"].cpu())

        avg_loss = total_loss / len(self.val_loader)
        pred_concat = torch.cat(all_pred)
        target_concat = torch.cat(all_target)
        metrics = compute_metrics(pred_concat, target_concat)
        metrics["loss"] = avg_loss
        return metrics

    def train(self, epochs=None):
        epochs = epochs or self.config["epochs"]
        print(f"\n{'=' * 60}")
        print(f"MPF-Net 训练   设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"训练样本: {len(self.train_loader.dataset):,}")
        print(f"验证样本: {len(self.val_loader.dataset):,}")
        print(f"批次大小: {self.train_loader.batch_size}")
        print(f"学习率:   {self.config['lr']}")
        print(f"Epochs:   {epochs}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, epochs + 1):
            train_loss, dur = self.train_epoch()
            val_metrics = self.validate()
            val_loss = val_metrics["loss"]

            save_str = "  ← BEST" if val_loss < self.best_val_loss else ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            log = (f"Epoch {epoch:2d}/{epochs}  "
                   f"train_loss={train_loss:.4f}  "
                   f"val_loss={val_loss:.4f}  "
                   f"rmse={val_metrics['rmse']:.2f}  "
                   f"mae={val_metrics['mae']:.2f}  "
                   f"mape={val_metrics['mape']:.2f}%  "
                   f"time={dur:.0f}s{save_str}")
            print(log)

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "rmse": val_metrics["rmse"],
                "mae": val_metrics["mae"],
                "mape": val_metrics["mape"],
            })

        # 保存训练历史
        hist_df = pd.DataFrame(self.history)
        fname = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        hist_df.to_csv(os.path.join(self.log_dir, fname), index=False)
        print(f"\n训练日志已保存: {self.log_dir}/{fname}")
        print(f"最佳 val_loss: {self.best_val_loss:.4f}")
        return self.history

    def _save_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.ckpt_dir, "mpf_net_best.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "model_config": self.model_cfg,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt["val_loss"]
        print(f"检查点已加载: {path}  (val_loss={self.best_val_loss:.4f})")
        return ckpt["epoch"]


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MPF-Net 训练")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader 线程数 (GPU时推荐 6-8)")
    p.add_argument("--stride", type=int, default=24,
                   help="滑动窗口步长 (24=每天一次预测)")
    p.add_argument("--resume", type=str, default=None,
                   help="从检查点恢复训练")
    p.add_argument("--run_id", type=int, default=None,
                   help="实验编号 (默认自动检测下一个)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 运行目录 ──
    run_id = args.run_id or run_utils.get_next_run_id()
    ckpt_dir = run_utils.get_weight_dir(run_id)
    log_dir = run_utils.get_log_dir(run_id)

    print("=" * 60)
    print(f"MPF-Net 训练   实验编号: run{run_id}")
    print("=" * 60)

    # ── DataLoader ──
    train_loader, val_loader, (n_train, n_val) = build_dataloaders(
        DATA_DIR, BERT_PATH, META_PATH,
        batch_size=args.batch,
        num_workers=args.workers,
        stride=args.stride,
    )

    # ── 模型配置 ──
    model_cfg = default_config(BERT_PATH)
    model_cfg["n_groups"] = 10
    model = build_model(model_cfg)

    # ── 训练配置 ──
    train_cfg = {
        "lr": args.lr,
        "weight_decay": 1e-5,
        "epochs": args.epochs,
    }

    # ── Trainer ──
    trainer = Trainer(model, train_loader, val_loader, train_cfg, model_cfg,
                      ckpt_dir=ckpt_dir, log_dir=log_dir)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(args.epochs)

    print(f"\n完成! 最佳模型: {ckpt_dir}/mpf_net_best.pth")
    print(f"训练日志: {log_dir}")
    print(f"使用: python predict.py --run_id {run_id}")


if __name__ == "__main__":
    main()
