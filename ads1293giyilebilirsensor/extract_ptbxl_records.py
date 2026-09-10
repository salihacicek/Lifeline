import os, ast
import pandas as pd

def get_ptbxl_patients(ptbxl_base_path: str, num_patients_per_class: int = 20):
    csv_path = os.path.join(ptbxl_base_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    af_records = []
    mi_records = []
    mi_scp_codes = ['IMI', 'ASMI', 'AMI', 'LMI', 'ALMI', 'INJAS', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
    
    for _, row in df.iterrows():
        if len(af_records) >= num_patients_per_class and len(mi_records) >= num_patients_per_class:
            break
        try:
            codes = ast.literal_eval(row['scp_codes'])
        except:
            continue
        if 'AFIB' in codes and len(af_records) < num_patients_per_class:
            af_records.append(f"'PTB-XL (AF): {row['filename_lr']}'")
        elif any(code in codes for code in mi_scp_codes) and len(mi_records) < num_patients_per_class:
            mi_records.append(f"'PTB-XL (MI): {row['filename_lr']}'")
    return af_records, mi_records

ptbxl_path = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
af, mi = get_ptbxl_patients(ptbxl_path, 20)
print(", ".join(af))
print(", ".join(mi))
