import os
import ast
import pandas as pd

ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
csv_path = os.path.join(ptbxl_base_path, "ptbxl_database.csv")
df = pd.read_csv(csv_path)

top_10_classes = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
records_by_class = {cls: 0 for cls in top_10_classes}

for _, row in df.iterrows():
    try:
        codes = ast.literal_eval(row['scp_codes'])
    except:
        continue
    for code in top_10_classes:
        if code in codes:
            records_by_class[code] += 1

for cls, count in records_by_class.items():
    print(f"{cls}: {count} total records in PTB-XL")
