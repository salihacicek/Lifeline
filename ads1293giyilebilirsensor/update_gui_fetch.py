import sys

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

if "import wfdb" not in code:
    code = code.replace("import time", "import time\nimport wfdb")

old_fetch_logic = """        def fetch_and_start():
            try:
                # İnternet üzerinden PhysioNet veri çekimi
                pipeline = RawDataPipeline(record_id=rec_id)
                data = pipeline.fetch_data(duration_seconds=300)
                self.sim_data = data["ecg_uv"]
                self.sim_idx = 0
                QMetaObject.invokeMethod(self, "on_sim_data_ready", Qt.QueuedConnection)
            except Exception as e:
                print(f"Veri çekme hatası: {e}")"""

new_fetch_logic = """        def fetch_and_start():
            try:
                if rec_id.startswith("PTB-XL"):
                    # ptbxl string looks like: PTB-XL (AF): records100/00000/00017_lr
                    record_filename = rec_id.split(": ")[1]
                    ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
                    record_path = os.path.join(ptbxl_base_path, record_filename)
                    
                    record = wfdb.rdrecord(record_path, channels=[0])
                    ecg_mv = record.p_signal[:, 0]
                    # Convert to microvolts as the GUI uses ecg_uv
                    self.sim_data = ecg_mv * 1000.0
                else:
                    # İnternet üzerinden PhysioNet veri çekimi (MIT-BIH)
                    pipeline = RawDataPipeline(record_id=rec_id)
                    data = pipeline.fetch_data(duration_seconds=300)
                    self.sim_data = data["ecg_uv"]
                    
                self.sim_idx = 0
                QMetaObject.invokeMethod(self, "on_sim_data_ready", Qt.QueuedConnection)
            except Exception as e:
                print(f"Veri çekme hatası: {e}")"""

code = code.replace(old_fetch_logic, new_fetch_logic)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
