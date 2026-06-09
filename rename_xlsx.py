import os

data_dir = "./data"
rename_map = {
    "extreme_weather_calculated.xlsx": "极端天气分级统计.xlsx",
    "extreme_weather_internet.xlsx": "极端天气预警.xlsx",
    "holiday.xlsx": "节假日标注.xlsx",
    "transformer_meta.xlsx": "变压器台账.xlsx",
    "transformer_raw_part1.xlsx": "变压器原始负荷_第1部分.xlsx",
    "transformer_raw_part2.xlsx": "变压器原始负荷_第2部分.xlsx",
    "transformer_raw_part3.xlsx": "变压器原始负荷_第3部分.xlsx",
    "transformer_raw_part4.xlsx": "变压器原始负荷_第4部分.xlsx",
    "transformer_raw_part5.xlsx": "变压器原始负荷_第5部分.xlsx",
    "transformer_raw_part6.xlsx": "变压器原始负荷_第6部分.xlsx",
    "transformer_raw_part7.xlsx": "变压器原始负荷_第7部分.xlsx",
    "transformer_raw_part8.xlsx": "变压器原始负荷_第8部分.xlsx",
    "weather.xlsx": "气象观测数据.xlsx",
    "weather_meta.xlsx": "气象站信息.xlsx",
}

for old, new in rename_map.items():
    old_path = os.path.join(data_dir, old)
    new_path = os.path.join(data_dir, new)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"  {old}  →  {new}")
    else:
        print(f"  [跳过] {old} 不存在")

print("\n重命名完成！")
