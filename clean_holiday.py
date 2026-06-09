import pandas as pd
import calendar

src = "data/节假日标注.xlsx"
dst = "data/节假日标注（整理版）.xlsx"

df = pd.read_excel(src)
df['时间'] = pd.to_datetime(df['时间'])

# For blank cells in 节假日名称, assign Weekday/Weekend based on the date
blank_mask = df['节假日名称'].isna()
weekday_names = {0: 'Weekday', 1: 'Weekday', 2: 'Weekday', 3: 'Weekday', 4: 'Weekday',
                 5: 'Weekend', 6: 'Weekend'}

df.loc[blank_mask, '节假日名称'] = df.loc[blank_mask, '时间'].apply(lambda x: weekday_names[x.weekday()])

# Save
with pd.ExcelWriter(dst, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    df.to_excel(writer, index=False)
    ws = writer.sheets['Sheet1']
    ws.set_column('A:A', 22)
    ws.set_column('B:B', 30)

print(f"Done! Saved to: {dst}")
print(f"  Total rows: {len(df)}")
print(f"  Holidays kept unchanged: {(~blank_mask).sum()}")
print(f"  Weekday fills: {(df.loc[blank_mask, '节假日名称'] == 'Weekday').sum()}")
print(f"  Weekend fills: {(df.loc[blank_mask, '节假日名称'] == 'Weekend').sum()}")
