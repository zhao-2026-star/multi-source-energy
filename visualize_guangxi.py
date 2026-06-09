# -*- coding: utf-8 -*-
"""
广西配电网变压器负荷可视化
- 地图：广西行政区划 + 气象站位置
- 图表：各区域变压器负荷统计
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import os
from sqlalchemy import create_engine

# ===== 设置中文字体 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("./result", exist_ok=True)

# ===== 读取数据 =====
engine = create_engine('sqlite:///./data/Transformer_DB/Transformer_DB.db')
meta_df = pd.read_sql('SELECT * FROM transformer_meta', engine)
gdf = gpd.read_file('data/guangxi_administration/guangxi.shp', encoding='utf-8')
weather_meta_df = pd.read_sql('SELECT * FROM weather_meta', engine)

# City编号 → 最近气象站对应区域名称
city_station_map = {
    0: "百色", 1: "柳州", 2: "桂林", 3: "北海",
    4: "贵港", 5: "南宁", 6: "钦州", 7: "钦州",
    8: "河池", 9: "梧州"
}

# ===== 各城市变压器统计 =====
city_load = meta_df.groupby('CITY').agg(
    变压器数量=('TRANSFORMER_ID', 'count'),
    平均容量=('YXRL', 'mean'),
    总容量=('YXRL', 'sum')
).reset_index()
city_load['区域名称'] = city_load['CITY'].map(city_station_map)

print("数据加载完成，开始绘图...")

# ===== 创建图形1：地图 + 柱状图组合 =====
fig = plt.figure(figsize=(18, 10))
fig.suptitle("广西配电网变压器负荷分布总览", fontsize=20, fontweight='bold', y=0.98)

# ---- 子图1：广西地图 ----
ax1 = fig.add_subplot(1, 2, 1)
gdf.boundary.plot(ax=ax1, linewidth=0.8, color='#333333', alpha=0.7)
gdf.plot(ax=ax1, facecolor='#e8f4f8', edgecolor='#666666', linewidth=0.5, alpha=0.6)

# 投影转换后计算中心点（EPSG:2380 Xian_1980 适合广西）
gdf_proj = gdf.to_crs('EPSG:2380')
gdf_proj['centroid'] = gdf_proj.geometry.centroid
gdf['centroid'] = gdf_proj['centroid'].to_crs('EPSG:4326')

for _, row in gdf.iterrows():
    ax1.annotate(
        row['name'], xy=(row['centroid'].x, row['centroid'].y),
        fontsize=8, ha='center', va='center', color='#2c3e50',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7)
    )

# 标注气象站位置
for _, row in weather_meta_df.iterrows():
    station_id = row['STATION_ID']
    matches = meta_df[meta_df['CLOSEST_STATION'] == station_id]
    if len(matches) > 0:
        cities = matches['CITY'].unique()
        city_labels = [str(city_station_map.get(c, str(c))) for c in cities]
        label = f"{row['STATION NAME']}\n({','.join(city_labels)})"
    else:
        label = row['STATION NAME']

    ax1.scatter(row['LON'], row['LAT'], s=80, c='#e74c3c', edgecolors='white',
                linewidth=1.5, zorder=5, marker='D')
    ax1.annotate(
        label, xy=(row['LON'], row['LAT']),
        xytext=(8, 8), textcoords='offset points', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff9c4', edgecolor='#f0c040', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='#888888', lw=0.5)
    )

ax1.set_title("广西行政区划与气象站分布", fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel("经度", fontsize=10)
ax1.set_ylabel("纬度", fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# ---- 子图2：变压器数量 ----
ax2 = fig.add_subplot(2, 2, 2)
colors = plt.cm.Set2(np.linspace(0, 1, len(city_load)))
bars1 = ax2.bar(city_load['区域名称'], city_load['变压器数量'], color=colors, edgecolor='white', linewidth=1.2)
ax2.set_title("各区域变压器数量", fontsize=14, fontweight='bold', pad=10)
ax2.set_ylabel("变压器数量 (个)", fontsize=10)
ax2.set_xlabel("区域", fontsize=10)
for bar, val in zip(bars1, city_load['变压器数量']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.tick_params(axis='x', rotation=30)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# ---- 子图3：平均容量 ----
ax3 = fig.add_subplot(2, 2, 4)
bars2 = ax3.bar(city_load['区域名称'], city_load['平均容量'], color=colors, edgecolor='white', linewidth=1.2)
ax3.set_title("各区域变压器平均容量 (kVA)", fontsize=14, fontweight='bold', pad=10)
ax3.set_ylabel("平均容量 (kVA)", fontsize=10)
ax3.set_xlabel("区域", fontsize=10)
for bar, val in zip(bars2, city_load['平均容量']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val:.0f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.tick_params(axis='x', rotation=30)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./result/广西配电网负荷分布总览.png", dpi=200, bbox_inches='tight')
print("已保存: ./result/广西配电网负荷分布总览.png")

# ===== 图2：统计明细表 =====
fig2, ax4 = plt.subplots(figsize=(12, 4))
ax4.axis('off')

table_data = []
for _, row in city_load.iterrows():
    table_data.append([row['CITY'], row['区域名称'], row['变压器数量'],
                       f"{row['平均容量']:.0f}", f"{row['总容量']:.0f}"])
# 汇总行
total_row = ["合计", "", city_load['变压器数量'].sum(),
             f"{city_load['平均容量'].mean():.0f}", f"{city_load['总容量'].sum():.0f}"]
table_data.append(total_row)

table = ax4.table(
    cellText=table_data,
    colLabels=["城市编号", "区域名称", "变压器数量(个)", "平均容量(kVA)", "总容量(kVA)"],
    cellLoc='center',
    loc='center',
    colWidths=[0.1, 0.2, 0.2, 0.2, 0.2]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# 表头样式
for j in range(5):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
# 合计行样式
last_row_idx = len(table_data)
for j in range(5):
    table[last_row_idx, j].set_facecolor('#f0f0f0')

ax4.set_title("广西配电网变压器统计明细表", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("./result/广西配电网统计明细表.png", dpi=200, bbox_inches='tight')
print("已保存: ./result/广西配电网统计明细表.png")

plt.show()
print("可视化完成！结果文件位于 result/ 目录下。")
