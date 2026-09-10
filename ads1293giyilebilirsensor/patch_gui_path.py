import os

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Fix the dropdown item
old_dropdown = """        self.record_selector.addItems([
            'PTB-XL: 00001_lr (Sağlıklı - NORM)',
            'PTB-XL: 00002_lr (Sağlıklı - NORM)',"""

new_dropdown = """        self.record_selector.addItems([
            'PTB-XL: records100/00000/00001_lr (Sağlıklı - NORM)',
            'PTB-XL: records100/00000/00002_lr (Sağlıklı - NORM)',"""
            
code = code.replace(old_dropdown, new_dropdown)

# Fix the ground truth extraction to use the basename for CSV lookup
old_extract = """                    import ast
                    try:
                        df_db = pd.read_csv(os.path.join(ptbxl_base_path, "ptbxl_database.csv"))
                        row = df_db[df_db['filename_lr'] == record_filename]"""

new_extract = """                    import ast
                    try:
                        df_db = pd.read_csv(os.path.join(ptbxl_base_path, "ptbxl_database.csv"))
                        # record_filename is records100/00000/00001_lr
                        row = df_db[df_db['filename_lr'] == record_filename]"""

code = code.replace(old_extract, new_extract)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
