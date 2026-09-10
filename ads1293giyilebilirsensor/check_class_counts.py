import os
import ast
import pandas as pd

ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
csv_path = os.path.join(ptbxl_base_path, "ptbxl_database.csv")
df = pd.read_csv(csv_path)

scp_df = pd.read_csv(os.path.join(ptbxl_base_path, "scp_statements.csv"), index_col=0)
PTBXL_CLASSES = list(scp_df.index)

records_by_class = {cls: [] for cls in PTBXL_CLASSES}

for _, row in df.iterrows():
    try:
        codes = ast.literal_eval(row['scp_codes'])
    except:
        continue
        
    for code in codes:
        if code in PTBXL_CLASSES and len(records_by_class[code]) < 200:
            records_by_class[code].append(row['filename_lr'])
            break

for cls, records in records_by_class.items():
    if len(records) > 0:
        print(f"{cls}: {len(records)} patients")
