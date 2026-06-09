import pandas as pd
import os

src = "data/极端天气预警.xlsx"
dst = "data/极端天气预警（整理版）.xlsx"

df = pd.read_excel(src)

# Fill blank cells in 极端天气类型 column with "Normal Weather"
df['极端天气类型'] = df['极端天气类型'].fillna('Normal Weather')
# Also handle empty strings or whitespace-only strings
df['极端天气类型'] = df['极端天气类型'].replace(r'^\s*$', 'Normal Weather', regex=True)

# Save with xlsxwriter to keep datetime formatting readable
with pd.ExcelWriter(dst, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    df.to_excel(writer, index=False)
    ws = writer.sheets['Sheet1']
    ws.set_column('A:A', 22)
    ws.set_column('B:B', 30)

print(f"Done! Saved to: {dst}")
print(f"  Total rows: {len(df)}")
print(f"  Filled {df['极端天气类型'].value_counts().get('Normal Weather', 0)} cells with 'Normal Weather'")
