import pandas as pd
import os
import math
from sqlalchemy import create_engine, inspect

db_path = "./data/Transformer_DB/Transformer_DB.db"
output_dir = "./data/Transformer_DB/excel_tables"
max_rows = 1_000_000  # Excel max ~1,048,576, use 1M as safety

engine = create_engine(f"sqlite:///{db_path}")
inspector = inspect(engine)
tables = inspector.get_table_names()
os.makedirs(output_dir, exist_ok=True)

for table in tables:
    print(f"Exporting: {table} ...", end=" ")
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    safe_name = table.replace("/", "_").replace("\\", "_")

    if len(df) <= max_rows:
        file_path = os.path.join(output_dir, f"{safe_name}.xlsx")
        df.to_excel(file_path, index=False)
        print(f"done ({len(df)} rows, {len(df.columns)} cols)")
    else:
        n_parts = math.ceil(len(df) / max_rows)
        print(f"large table, splitting into {n_parts} parts ...", end=" ")
        for i in range(n_parts):
            chunk = df.iloc[i * max_rows : (i + 1) * max_rows]
            file_path = os.path.join(output_dir, f"{safe_name}_part{i+1}.xlsx")
            chunk.to_excel(file_path, index=False)
        print(f"done ({len(df)} rows total, {n_parts} files)")

print(f"\nAll {len(tables)} tables exported to: {output_dir}")
