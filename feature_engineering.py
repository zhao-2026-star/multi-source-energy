# -*- coding: utf-8 -*-
"""
多源异构数据特征工程 - 为变压器负荷预测构造特征矩阵

输入：data/ 下所有数据源
输出：data/feature_engineered/ 下的训练集和验证集

流程：
  1. 读取所有数据源
  2. 按时间切分为训练集(70%)和验证集(30%)
  3. 构造负荷滞后特征、日历特征、气象特征、节假日特征、台账特征
  4. 合并为宽表，保存
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
import warnings
warnings.filterwarnings('ignore')

# 解决 Windows GBK 编码问题
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================================
# 0. 路径与参数
# ==========================================================
DATA_DIR = "./data"
RAW_DIR = "./data/raw"
OUTPUT_DIR = "./data/feature_engineered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 时间切分点（按时间顺序 7:3）
# 总时间: 2022-01-01 ~ 2023-11-10（约 679 天）
# 训练: 2022-01-01 ~ 2023-04-21（约 476 天, 70%）
# 验证: 2023-04-22 ~ 2023-11-10（约 203 天, 30%）
TRAIN_CUTOFF = "2023-04-21 23:00:00"

# 构造滞后特征需要丢弃的时间窗口（最前面 7 天，因滞后特征不完整）
LAG_WARMUP = "2022-01-08 00:00:00"

print("=" * 60)
print("特征工程 - 多源异构数据融合")
print("=" * 60)

# ==========================================================
# 1. 读取负荷数据（8 部分合并）
# ==========================================================
print("\n[1/7] 读取负荷数据...")
load_parts = []
for i in range(1, 9):
    f = os.path.join(RAW_DIR, f"变压器原始负荷_第{i}部分（整理版）.xlsx")
    if os.path.exists(f):
        df_chunk = pd.read_excel(f, dtype={"变压器编号": str, "负荷(kW)": np.float32})
        load_parts.append(df_chunk)
        print(f"  [OK] 第{i}部分: {len(df_chunk):,} 行")

load_df = pd.concat(load_parts, ignore_index=True)
# 删除 423 号以外的变压器（台账有 423 台，但可能 load 中多出异常变压器）
transformer_list = pd.read_excel(os.path.join(RAW_DIR, "变压器台账.xlsx"))
valid_ids = set(transformer_list["变压器编号"].astype(str).tolist())
before = len(load_df)
load_df = load_df[load_df["变压器编号"].isin(valid_ids)]
print(f"  -> 合并后: {before:,} 行, 过滤后: {len(load_df):,} 行")
print(f"  -> 变压器数: {load_df['变压器编号'].nunique()}")
print(f"  -> 时间范围: {load_df['时间'].min()} ~ {load_df['时间'].max()}")
del load_parts

# 排序（按变压器编号 + 时间，这是构造滞后特征的前提）
load_df = load_df.sort_values(["变压器编号", "时间"]).reset_index(drop=True)

# ==========================================================
# 2. 读取辅助数据
# ==========================================================
print("\n[2/7] 读取外部数据...")

# 2a 变压器台账
transformer_meta = pd.read_excel(
    os.path.join(RAW_DIR, "变压器台账.xlsx"),
    dtype={"变压器编号": str}
)
transformer_meta = transformer_meta[["变压器编号", "城市编号", "额定容量(kVA)", "最近气象站"]]
transformer_meta["最近气象站"] = transformer_meta["最近气象站"].astype(str)
print(f"  [OK] 变压器台账: {len(transformer_meta)} 台")

# 2b 气象观测数据（逐日，按站）
weather_df = pd.read_excel(
    os.path.join(RAW_DIR, "气象观测数据.xlsx")
)
weather_df["时间"] = pd.to_datetime(weather_df["时间"])
weather_df["气象站ID"] = weather_df["气象站ID"].astype(str)
# 筛选有用列
weather_cols = ["时间", "气象站ID", "温度(°C)", "相对湿度(%)",
                "平均风速", "降水量(mm)", "最高温度(°C)", "最低温度(°C)"]
existing_weather_cols = [c for c in weather_cols if c in weather_df.columns]
weather_df = weather_df[existing_weather_cols]
# 日均值展开为逐时（气象数据是日值，所有24小时共享同一个值）
weather_df["date"] = weather_df["时间"].dt.date
print(f"  [OK] 气象数据: {len(weather_df)} 行, {weather_df['气象站ID'].nunique()} 个站")

# 2c 节假日标注
holiday_df = pd.read_excel(
    os.path.join(DATA_DIR, "节假日标注（整理版）.xlsx")
)
holiday_df["时间"] = pd.to_datetime(holiday_df["时间"])
holiday_df["date"] = holiday_df["时间"].dt.date
print(f"  [OK] 节假日标注: {len(holiday_df)} 行")

# 2d 极端天气预警
extreme_df = pd.read_excel(
    os.path.join(DATA_DIR, "极端天气预警（整理版）.xlsx")
)
extreme_df["时间"] = pd.to_datetime(extreme_df["时间"])
extreme_df["date"] = extreme_df["时间"].dt.date
print(f"  [OK] 极端天气预警: {len(extreme_df)} 行")

# 2e 获取变压器-气象站-城市映射
city_station_map = transformer_meta[["城市编号", "最近气象站"]].drop_duplicates()
city_station_map = city_station_map.sort_values("城市编号")
print(f"  [OK] 城市-气象站映射:")
for _, row in city_station_map.iterrows():
    print(f"     城市 {int(row['城市编号'])} -> 气象站 {row['最近气象站']}")

# 预构造"城市编号 -> 气象站"的快速字典
city_to_station = dict(zip(city_station_map["城市编号"], city_station_map["最近气象站"]))

# ==========================================================
# 3. 按城市分组处理（减少 pandas 处理压力）
# ==========================================================
print("\n[3/7] 按城市分组处理负荷数据...")

train_parts = []
val_parts = []
city_group_info = []

for city_id in sorted(load_df.merge(transformer_meta[["变压器编号", "城市编号"]], on="变压器编号")["城市编号"].unique()):
    city_id = int(city_id)
    # 取该城市所有变压器
    city_transformers = transformer_meta[transformer_meta["城市编号"] == city_id]["变压器编号"].tolist()
    city_load = load_df[load_df["变压器编号"].isin(city_transformers)].copy()

    print(f"\n  城市 {city_id} ({len(city_transformers)} 台变压器, {len(city_load):,} 行)")

    # ------ 3a 构造时间特征 ------
    city_load["hour"] = city_load["时间"].dt.hour
    city_load["day_of_week"] = city_load["时间"].dt.dayofweek  # 0=周一
    city_load["month"] = city_load["时间"].dt.month
    city_load["is_weekend"] = (city_load["day_of_week"] >= 5).astype(np.int8)
    city_load["day_of_year"] = city_load["时间"].dt.dayofyear

    # ------ 3b 构造负荷滞后特征（按每台变压器分别做）------
    # 注意：shift 在每个变压器的分组内进行
    city_load = city_load.sort_values(["变压器编号", "时间"])

    # 滞后特征（标记 NaN 的行在后续会被丢弃）
    city_load["lag_1h"] = city_load.groupby("变压器编号")["负荷(kW)"].shift(1)
    city_load["lag_2h"] = city_load.groupby("变压器编号")["负荷(kW)"].shift(2)
    city_load["lag_3h"] = city_load.groupby("变压器编号")["负荷(kW)"].shift(3)
    city_load["lag_24h"] = city_load.groupby("变压器编号")["负荷(kW)"].shift(24)
    city_load["lag_48h"] = city_load.groupby("变压器编号")["负荷(kW)"].shift(48)
    city_load["lag_168h"] = city_load.groupby("变压器编号")["负荷(kW)"].shift(168)

    # 滚动统计特征
    city_load["roll_mean_24h"] = city_load.groupby("变压器编号")["负荷(kW)"].transform(
        lambda x: x.rolling(window=24, min_periods=12).mean()
    )
    city_load["roll_std_24h"] = city_load.groupby("变压器编号")["负荷(kW)"].transform(
        lambda x: x.rolling(window=24, min_periods=12).std()
    )
    city_load["roll_max_24h"] = city_load.groupby("变压器编号")["负荷(kW)"].transform(
        lambda x: x.rolling(window=24, min_periods=12).max()
    )
    city_load["roll_min_24h"] = city_load.groupby("变压器编号")["负荷(kW)"].transform(
        lambda x: x.rolling(window=24, min_periods=12).min()
    )

    # ------ 3c 合并台账数据（静态属性）------
    meta_subset = transformer_meta[transformer_meta["城市编号"] == city_id][
        ["变压器编号", "额定容量(kVA)"]
    ].copy()
    meta_subset.rename(columns={"额定容量(kVA)": "capacity_kva"}, inplace=True)
    city_load = city_load.merge(meta_subset, on="变压器编号", how="left")

    # 计算负荷率（当前负荷 / 额定容量），容量为 0 时填 0
    city_load["load_factor"] = np.where(
        city_load["capacity_kva"] > 0,
        city_load["负荷(kW)"] / city_load["capacity_kva"],
        0.0,
    )

    # ------ 3d 合并气象数据（按城市关联的气象站）------
    station_id = city_to_station.get(city_id, None)
    if station_id:
        city_weather = weather_df[weather_df["气象站ID"] == station_id].copy()
        # 展开为逐时（每日 24 小时对齐）
        hourly_idx = pd.date_range(
            start=city_load["时间"].min().strftime("%Y-%m-%d"),
            end=city_load["时间"].max().strftime("%Y-%m-%d 23:00"),
            freq="h"
        )
        hourly_weather = pd.DataFrame({"时间": hourly_idx})
        hourly_weather["date"] = hourly_weather["时间"].dt.date
        hourly_weather = hourly_weather.merge(
            city_weather.drop(columns=["时间", "气象站ID"]),
            on="date", how="left"
        )
        city_load = city_load.merge(
            hourly_weather.drop(columns=["date"]),
            on="时间", how="left"
        )

    # ------ 3e 合并节假日数据（按日期）------
    city_load["date"] = city_load["时间"].dt.date
    holiday_map = holiday_df.set_index("date")["节假日名称"].to_dict()

    # 是否节假日
    city_load["holiday_name"] = city_load["date"].map(holiday_map)
    city_load["is_holiday"] = (~city_load["holiday_name"].isin(
        ["Weekday", "Weekend", None, np.nan]
    )).astype(np.int8)

    # ------ 3f 合并极端天气预警数据 ------
    extreme_map = extreme_df.set_index("date")["极端天气类型"].to_dict()
    city_load["extreme_weather"] = city_load["date"].map(extreme_map)
    # 是否为极端天气（非 Normal Weather）
    city_load["is_extreme"] = (
        (city_load["extreme_weather"] != "Normal Weather") &
        (city_load["extreme_weather"].notna())
    ).astype(np.int8)

    # ------ 3g 删除 date 辅助列 ------
    city_load.drop(columns=["date"], inplace=True)

    # ------ 3h 统一浮点数精度 ------
    # 原始负荷 & 滞后特征: 保留 1 位小数（原始数据精度 0.5）
    for col in ["负荷(kW)", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
                "roll_max_24h", "roll_min_24h"]:
        city_load[col] = city_load[col].round(1)

    # 滚动均值和标准差: 保留 2 位小数
    city_load["roll_mean_24h"] = city_load["roll_mean_24h"].round(2)
    city_load["roll_std_24h"] = city_load["roll_std_24h"].round(2)

    # 负荷率: 保留 4 位小数（容量数千级别，4位对应0.01%）
    city_load["load_factor"] = city_load["load_factor"].round(4)

    # 气象数据: 统一精度
    city_load["温度(°C)"] = city_load["温度(°C)"].round(2)
    city_load["最高温度(°C)"] = city_load["最高温度(°C)"].round(2)
    city_load["最低温度(°C)"] = city_load["最低温度(°C)"].round(2)
    city_load["相对湿度(%)"] = city_load["相对湿度(%)"].round(1)
    city_load["平均风速"] = city_load["平均风速"].round(2)
    # 降水量保留不变（已在合理范围 ~3 位）

    # ------ 3i 按时间切分 ------
    train_mask = city_load["时间"] <= TRAIN_CUTOFF
    # 丢弃训练集最前面的热身期（滞后特征不完整）
    warmup_mask = city_load["时间"] >= LAG_WARMUP

    city_train = city_load[train_mask & warmup_mask].copy()
    city_val = city_load[~train_mask].copy()

    # 丢弃滞后特征的 NaN 行
    lag_cols = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h"]
    before_train = len(city_train)
    city_train = city_train.dropna(subset=lag_cols)
    after_train = len(city_train)
    dropped = before_train - after_train
    if dropped > 0:
        print(f"    训练集丢弃 {dropped:,} 行（滞后特征不完整）")

    train_parts.append(city_train)
    val_parts.append(city_val)

    city_group_info.append({
        "city": city_id,
        "transformers": len(city_transformers),
        "train_rows": len(city_train),
        "val_rows": len(city_val)
    })

    # 及时释放内存
    del city_load, city_train, city_val

del load_df

# ==========================================================
# 4. 合并所有城市数据
# ==========================================================
print("\n[4/7] 合并所有城市数据...")

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)

print(f"  训练集: {len(train_df):,} 行, {train_df.shape[1]} 列")
print(f"  验证集: {len(val_df):,} 行, {val_df.shape[1]} 列")

# ==========================================================
# 5. 保留文本特征（不编码为数值，给后续 BERT 类模型处理）
# ==========================================================
print("\n[5/7] 保留文本特征...")

# 确保文本列为 str 类型（NaN 填充为空字符串）
train_df["holiday_name"] = train_df["holiday_name"].fillna("").astype(str)
val_df["holiday_name"] = val_df["holiday_name"].fillna("").astype(str)

train_df["extreme_weather"] = train_df["extreme_weather"].fillna("").astype(str)
val_df["extreme_weather"] = val_df["extreme_weather"].fillna("").astype(str)

# 同时保留 is_holiday / is_extreme 作为快捷二值特征（不依赖文本模型也能用）
# 已在第 3 步中生成，此处不做变动

# ==========================================================
# 6. 确认特征列顺序并保存
# ==========================================================
print("\n[6/7] 最终特征列表：")

# 确定特征列（去掉 变压器编号 和 时间 为目标列）
feature_cols = [c for c in train_df.columns if c not in ["变压器编号", "时间"]]
target_col = "负荷(kW)"

print(f"\n  [OK] 目标变量: {target_col}")
print(f"  [OK] 特征数量: {len(feature_cols)}")
print(f"\n  特征列表:")
for i, col in enumerate(feature_cols):
    dtype_str = str(train_df[col].dtype)
    non_null = train_df[col].notna().sum()
    null_count = len(train_df) - non_null
    print(f"    {i+1:2d}. {col:20s}  [{dtype_str:10s}]  非空 {non_null:>8,} 空 {null_count:>6,}")

# ==========================================================
# 7. 保存
# ==========================================================
print(f"\n[7/7] 保存特征矩阵...")

# 保存为 pickle 格式（无需额外依赖，读写快）
train_df.to_pickle(os.path.join(OUTPUT_DIR, "train_feature_matrix.pkl"))
val_df.to_pickle(os.path.join(OUTPUT_DIR, "val_feature_matrix.pkl"))
print(f"\n  [OK] 训练集: {OUTPUT_DIR}/train_feature_matrix.pkl ({len(train_df):,} 行)")
print(f"  [OK] 验证集: {OUTPUT_DIR}/val_feature_matrix.pkl ({len(val_df):,} 行)")

# 也保存为 CSV（便于直接查看，但只保存前 10000 行做预览）
try:
    train_preview = train_df.head(10000)
    val_preview = val_df.head(10000)
    train_preview.to_csv(os.path.join(OUTPUT_DIR, "train_preview.csv"), index=False, encoding="utf-8-sig")
    val_preview.to_csv(os.path.join(OUTPUT_DIR, "val_preview.csv"), index=False, encoding="utf-8-sig")
    print(f"  [OK] 预览 CSV: {OUTPUT_DIR}/train_preview.csv (10000 行)")
except PermissionError:
    print(f"  [!] CSV 预览因文件占用未能写入（pickle 已正确保存）")

# 保存城市分组信息摘要
summary = pd.DataFrame(city_group_info)
summary.to_csv(os.path.join(OUTPUT_DIR, "city_group_summary.csv"), index=False, encoding="utf-8-sig")
print(f"  [OK] 城市分组摘要: {OUTPUT_DIR}/city_group_summary.csv")

# 打印最终汇总
total = len(train_df) + len(val_df)
print(f"\n{'=' * 60}")
print(f"  数据汇总")
print(f"  {'=' * 60}")
print(f"  总样本数:        {total:>12,}")
print(f"  训练集 (70%):    {len(train_df):>12,} ({len(train_df)/total*100:.1f}%)")
print(f"  验证集 (30%):    {len(val_df):>12,} ({len(val_df)/total*100:.1f}%)")
print(f"  特征数:           {len(feature_cols):>12}")
print(f"  变压器数(训练):  {train_df['变压器编号'].nunique():>12}")
print(f"  变压器数(验证):  {val_df['变压器编号'].nunique():>12}")
print(f"{'=' * 60}")
print("特征工程完成!")
