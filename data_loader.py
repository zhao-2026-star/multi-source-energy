# -*- coding: utf-8 -*-
"""
MPF-Net 数据加载器
===================
将特征工程产出的 pickle 转换为 PyTorch DataLoader。
使用滑动窗口构造 (168h 输入 → 24h 预测) 样本。

设计要点：
  - 每个变压器独立分组，避免跨变压器混叠
  - 文本特征预转索引（查 BERT 词汇表）
  - 数值 NaN 统一填零，missing_mask 告知注意力层
  - 多线程 DataLoader 最大化 GPU 利用率
"""

import pickle
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
# 列名常量（与 feature_engineering.py 产出对齐）
# ═══════════════════════════════════════════════════════════════

NUMERICAL_COLS = [
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_48h", "lag_168h",
    "roll_mean_24h", "roll_std_24h", "roll_max_24h", "roll_min_24h",
    "capacity_kva", "load_factor",
    "温度(°C)", "相对湿度(%)", "平均风速", "降水量(mm)",
    "最高温度(°C)", "最低温度(°C)",
]

CATEGORICAL_COLS = [
    "hour", "day_of_week", "month",
    "is_weekend",
    "is_holiday", "is_extreme",
]

TEXT_COLS = ["holiday_name", "extreme_weather"]

TARGET_COL = "负荷(kW)"
TRANSFORMER_COL = "变压器编号"
TIME_COL = "时间"

# 滑动窗口参数
WINDOW = 168   # 7 天输入
HORIZON = 24   # 24h 预测
STRIDE = 24    # 每天一次预测（可调整）


# ═══════════════════════════════════════════════════════════════
# BERT 词汇表构建
# ═══════════════════════════════════════════════════════════════

def build_text_vocab(bert_path):
    """从 BERT 嵌入 pickle 中恢复 text → idx 映射（与模型侧排序一致）。"""
    with open(bert_path, "rb") as f:
        emb_dict = pickle.load(f)
    vocab = sorted(emb_dict.keys())
    return {v: i for i, v in enumerate(vocab)}


# ═══════════════════════════════════════════════════════════════
# MPF Dataset
# ═══════════════════════════════════════════════════════════════

class MPFDataset(Dataset):
    """滑动窗口数据集，每个样本 = 168h 输入 → 24h 预测目标（归一化后）。"""

    def __init__(self, df, text_vocab, city_map,
                 window=WINDOW, horizon=HORIZON, stride=STRIDE,
                 normalizer=None):
        """
        normalizer: dict {变压器编号: (mean, std)}，验证集沿用训练集统计量
        为 None 时自动从本数据集计算（训练集）。
        """
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.stride = stride
        total_len = window + horizon
        self.is_train = (normalizer is None)

        # 按变压器分组，每个变压器独立建滑动窗口索引
        self.data = []   # 每个元素: dict of tensors
        self.index = []  # [(group_idx, window_start), ...]
        self.normalizer = normalizer or {}  # {tid: (mean, std)}

        for tid, grp in df.groupby(TRANSFORMER_COL, sort=False):
            grp = grp.sort_values(TIME_COL).reset_index(drop=True)
            n = len(grp)
            if n < total_len:
                continue  # 数据太少，跳过

            city_id = int(city_map.get(tid, 0))

            # ── 数值特征 ──
            num_arr = grp[NUMERICAL_COLS].to_numpy(dtype=np.float32)
            num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)

            # ── 类别特征 ──
            cat_arr = grp[CATEGORICAL_COLS].to_numpy(dtype=np.int64)

            # ── 文本特征 → 索引 ──
            text_arr = np.empty((n, len(TEXT_COLS)), dtype=np.int64)
            for j, col in enumerate(TEXT_COLS):
                text_arr[:, j] = grp[col].map(text_vocab).fillna(0).to_numpy(dtype=np.int64)

            # ── 缺失掩码（1=观测到，0=缺失） ──
            mask_arr = np.ones(n, dtype=np.float32)
            load_vals = grp[TARGET_COL].to_numpy(dtype=np.float32)
            mask_arr[np.isnan(load_vals)] = 0.0

            # ── 目标负荷 ──
            target_arr = np.nan_to_num(load_vals, nan=0.0, posinf=0.0, neginf=0.0)

            # ── 目标归一化统计量（每台变压器独立 z-score） ──
            # 使各变压器的误差对 loss 贡献相等，解决 MAPE 偏高问题
            if tid in self.normalizer:
                t_mean, t_std = self.normalizer[tid]
            else:
                t_mean = float(np.mean(target_arr))
                t_std = float(np.std(target_arr)) or 1.0
                self.normalizer[tid] = (t_mean, t_std)

            # ── 滑动窗口索引 ──
            starts = range(0, n - total_len + 1, stride)
            for s in starts:
                self.index.append((len(self.data), s))

            self.data.append({
                "num": torch.from_numpy(num_arr),
                "cat": torch.from_numpy(cat_arr),
                "text": torch.from_numpy(text_arr),
                "mask": torch.from_numpy(mask_arr),
                "target": torch.from_numpy(target_arr),
                "target_mean": t_mean,
                "target_std": t_std,
                "group_id": torch.tensor(city_id, dtype=torch.long),
            })

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data_idx, start = self.index[idx]
        d = self.data[data_idx]
        end = start + self.window

        cat_slice = d["cat"][start:end]  # [W, 6]
        target_slice = d["target"][end:end + self.horizon]
        # z-score 归一化目标值
        target_norm = (target_slice - d["target_mean"]) / (d["target_std"] + 1e-8)

        return {
            "num_feat":    d["num"][start:end],                    # [W, 18]
            "cat_feat": {                                          # {name: [W]}
                name: cat_slice[:, i] for i, name in enumerate(CATEGORICAL_COLS)
            },
            "text_feat": {                                         # {name: [W]}
                "holiday_name":    d["text"][start:end, 0],
                "extreme_weather": d["text"][start:end, 1],
            },
            "mask":        d["mask"][start:end],                   # [W]
            "group_id":    d["group_id"],                          # scalar
            "target":      target_norm.float(),                    # [24] 归一化
            "target_mean": torch.tensor(d["target_mean"], dtype=torch.float32),
            "target_std":  torch.tensor(d["target_std"],  dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════
# DataLoader 工厂
# ═══════════════════════════════════════════════════════════════

def build_dataloaders(data_dir, bert_path, meta_path,
                      batch_size=256, num_workers=0,
                      window=WINDOW, horizon=HORIZON, stride=STRIDE):
    """构建训练和验证 DataLoader。

    返回:
      train_loader, val_loader, (train_windows, val_windows)
    """

    print("[DataLoader] 加载 BERT 词汇表...")
    text_vocab = build_text_vocab(bert_path)
    print(f"            {len(text_vocab)} 个文本词条")

    print("[DataLoader] 加载变压器台账...")
    meta = pd.read_excel(meta_path, dtype={TRANSFORMER_COL: str})
    city_map = dict(zip(meta[TRANSFORMER_COL], meta["城市编号"]))
    print(f"            {len(city_map)} 台变压器")

    loaders = []
    window_counts = []
    normalizer = None
    save_normalizer_path = os.path.join(data_dir, "target_normalizers.pkl")

    for name, pkl_name in [("训练", "train_feature_matrix.pkl"),
                            ("验证", "val_feature_matrix.pkl")]:
        pkl_path = os.path.join(data_dir, pkl_name)
        print(f"[DataLoader] 读取 {name}集: {pkl_name}...")
        df = pd.read_pickle(pkl_path)

        ds = MPFDataset(df, text_vocab, city_map,
                        window=window, horizon=horizon, stride=stride,
                        normalizer=normalizer)

        # 训练集统计量传递给验证集
        if normalizer is None:
            normalizer = ds.normalizer
            # 保存到文件，供 predict.py 使用
            with open(save_normalizer_path, "wb") as f:
                pickle.dump(normalizer, f)
            print(f"            目标归一化统计量已保存: {save_normalizer_path}")

        nw = len(ds)
        window_counts.append(nw)
        print(f"            {nw:,} 个样本 ({name})")

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": (name == "训练"),
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = True

        loader = DataLoader(ds, **loader_kwargs)
        loaders.append(loader)

    print(f"[DataLoader] 完成!")
    return loaders[0], loaders[1], tuple(window_counts)


# ═══════════════════════════════════════════════════════════════
# 独立测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIR = "./data/feature_engineered"
    BERT_PATH = "./data/bert_embeddings.pkl"
    META_PATH = "./data/raw/变压器台账.xlsx"

    train_loader, val_loader, (n_train, n_val) = build_dataloaders(
        DATA_DIR, BERT_PATH, META_PATH,
        batch_size=64, num_workers=0,
    )

    batch = next(iter(train_loader))
    print(f"\nBatch 结构:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:15s}  {list(v.shape)}  {v.dtype}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k:15s}[{kk:20s}]  {list(vv.shape)}  {vv.dtype}")
    print(f"\n  总参数量: {n_train:,} 训练 / {n_val:,} 验证")
    print("DataLoader 测试通过!")
