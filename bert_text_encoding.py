# -*- coding: utf-8 -*-
"""
文本特征 BERT 编码预处理
==========================
为 holiday_name 和 extreme_weather 生成 frozen BERT 嵌入向量。
离线运行一次，训练时直接查表，避免每 epoch 重复跑 BERT。

输入:  train/val_feature_matrix.pkl 中的文本列
输出:  data/bert_embeddings.pkl  (dict: text_str → 768-dim numpy array)
       data/bert_unique_texts.txt (唯一文本列表，供核查)

使用预训练 Chinese BERT (bert-base-chinese)，权重冻结。
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "./data"
FEATURE_DIR = "./data/feature_engineered"
OUTPUT_FILE = os.path.join(DATA_DIR, "bert_embeddings.pkl")
TEXT_LIST_FILE = os.path.join(DATA_DIR, "bert_unique_texts.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("BERT 文本编码预处理")
print(f"设备: {DEVICE}")
print("=" * 60)

# ==========================================================
# 1. 收集所有唯一文本值
# ==========================================================
print("\n[1/4] 收集唯一文本值...")

train_df = pd.read_pickle(os.path.join(FEATURE_DIR, "train_feature_matrix.pkl"))
val_df = pd.read_pickle(os.path.join(FEATURE_DIR, "val_feature_matrix.pkl"))

holiday_names = set(train_df["holiday_name"].unique()) | set(val_df["holiday_name"].unique())
extreme_weathers = set(train_df["extreme_weather"].unique()) | set(val_df["extreme_weather"].unique())

all_texts = sorted(holiday_names) + sorted(extreme_weathers)
print(f"  holiday_name 唯一值: {len(holiday_names)} 个")
print(f"  extreme_weather 唯一值: {len(extreme_weathers)} 个")
print(f"  合计: {len(all_texts)} 个")

# 保存唯一文本列表
with open(TEXT_LIST_FILE, "w", encoding="utf-8") as f:
    f.write("=== holiday_name ===\n")
    for t in sorted(holiday_names):
        f.write(f"  {t}\n")
    f.write("\n=== extreme_weather ===\n")
    for t in sorted(extreme_weathers):
        f.write(f"  {t}\n")
print(f"  [OK] 已保存: {TEXT_LIST_FILE}")

# ==========================================================
# 2. 英→中 映射（适配 Chinese BERT）
# ==========================================================
print("\n[2/4] 构建英→中映射...")

en_to_cn = {
    # --- holiday_name ---
    "New year":                "元旦",
    "Spring Festival":         "春节",
    "Qingming":                "清明节",
    "International Labor Day": "劳动节",
    "Dragon Boat Festival":    "端午节",
    "Mid-autumn festival":     "中秋节",
    "National Day":            "国庆节",
    "Weekday":                 "工作日",
    "Weekend":                 "周末",
    # --- extreme_weather ---
    "Normal Weather":          "正常天气",
    "Excessive flooding":      "暴雨洪涝",
    "Dragon-boat rain":        "龙舟水",
    "Typhoon Chaba":           "台风查帕卡",
    "Hot weather":             "高温",
    "Cold wave":               "寒潮",
    "Drought":                 "干旱",
    "Severe convective weather": "强对流天气",
    "Rainstorm":               "暴雨",
    "Extreme rainfall":        "极端降水",
    "Typhoon Haikui":          "台风海葵",
    "Tropical Storm Sanba":    "热带风暴三巴",
}

# 验证所有文本都有映射
missing = [t for t in all_texts if t not in en_to_cn]
if missing:
    print(f"  [!] 警告: 以下文本缺少中英映射: {missing}")
    for t in missing:
        en_to_cn[t] = t  # 回退：保留原文

cn_texts = [en_to_cn[t] for t in all_texts]
print(f"  映射完成, {len(cn_texts)} 个中文文本:")
for en, cn in zip(all_texts, cn_texts):
    print(f"    {en:35s} → {cn}")

# ==========================================================
# 3. BERT 编码
# ==========================================================
print(f"\n[3/4] 加载 BERT (bert-base-chinese)...")

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
model.eval()
model.to(DEVICE)

# 冻结参数
for param in model.parameters():
    param.requires_grad = False

print(f"  [OK] BERT 已加载，参数已冻结")

# 编码
embeddings = {}
with torch.no_grad():
    for en, cn in zip(all_texts, cn_texts):
        inputs = tokenizer(cn, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        # 取 [CLS] token 的 embedding (768-dim)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        embeddings[en] = cls_emb
        print(f"    [OK] {en:35s} → {cn}  ({cls_emb.shape[0]}维)")

# ==========================================================
# 4. 保存
# ==========================================================
print(f"\n[4/4] 保存 BERT 嵌入...")

import pickle
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print(f"  [OK] 已保存: {OUTPUT_FILE}")
print(f"  [OK] 共 {len(embeddings)} 个文本嵌入, 每个 {list(embeddings.values())[0].shape[0]} 维")
print(f"  [OK] 文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
print(f"\n{'=' * 60}")
print("BERT 编码预处理完成!")
print(f"提示: 训练时用 pickle.load(open('{OUTPUT_FILE}','rb')) 加载即可")
print("=" * 60)
