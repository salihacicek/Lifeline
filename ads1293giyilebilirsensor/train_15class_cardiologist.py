import os
import ast
import pandas as pd
import numpy as np
import scipy.signal

import wfdb
import torch
import warnings

from ads1293_pipeline.module1_data_acquisition import RawDataPipeline, ADCScaler
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import FeatureExtractionPipeline
from ads1293_pipeline.module6_training import TrainingPipeline

warnings.filterwarnings("ignore")

# Define 15 specific classes
PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES
CLASS_MAPPING = {cls_name: idx for idx, cls_name in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)

def process_signal_and_extract_features(raw_signal, fs, target_class_idx):
    pipeline2 = SignalCleaningPipeline(fs=fs)
    cleaned_signal = pipeline2.run(raw_signal, normalize=True)["ecg_normalized"]
    
    pipeline3 = FeatureExtractionPipeline()
    features = pipeline3.run(cleaned_signal, fs=fs)
    
    cnn_X = []
    labels = []
    for segment in features.get("heartbeat_segments", []):
        if len(segment) != 250:
            segment = scipy.signal.resample(segment, 250)
        cnn_X.append(segment)
        labels.append(target_class_idx)
    return cnn_X, labels

def fetch_from_physionet(db: str, record_id: str, target_class_str: str, duration: int = 600):
    print(f"[MIT-BIH] {record_id} çekiliyor (Sınıf: {target_class_str})...")
    pipeline = RawDataPipeline(record_id=record_id, duration_seconds=duration, simulate_packet_loss=False)
    pipeline.loader.database = db
    try:
        data = pipeline.fetch_data(duration_seconds=duration)
    except Exception as e:
        print(f"Hata: {e}")
        return [], []
    return process_signal_and_extract_features(data['raw_adc_24bit'], data['fs'], CLASS_MAPPING[target_class_str])

def fetch_from_ptbxl_local(ptbxl_base_path: str, record_filename: str, target_class_str: str):
    record_path = os.path.join(ptbxl_base_path, record_filename)
    try:
        record = wfdb.rdrecord(record_path, channels=[0]) # Lead I
        ecg_mv = record.p_signal[:, 0]
        fs = record.fs
        
        ecg_uv = ecg_mv * 1000.0
        scaler = ADCScaler(v_ref=2.4)
        raw_adc_float = ecg_uv / scaler.v_lsb_uv
        raw_adc_24bit = np.clip(
            np.round(raw_adc_float).astype(np.int32),
            -scaler.adc_max,
            scaler.adc_max - 1,
        )
        return process_signal_and_extract_features(raw_adc_24bit, fs, CLASS_MAPPING[target_class_str])
    except Exception as e:
        return [], []

def get_ptbxl_patients(ptbxl_base_path: str, num_patients_per_class: int = 500):
    csv_path = os.path.join(ptbxl_base_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    
    records_by_class = {cls: [] for cls in PTBXL_CLASSES}
    
    # Shuffle dataframe to ensure random patient selection
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for _, row in df.iterrows():
        try:
            codes = ast.literal_eval(row['scp_codes'])
        except:
            continue
            
        for code in PTBXL_CLASSES:
            if code in codes and len(records_by_class[code]) < num_patients_per_class:
                records_by_class[code].append(row['filename_lr'])
                break # Ensure this patient is only counted for one major class to avoid leakage
                
    return records_by_class

def main():
    print("==========================================================")
    print(f"ADS1293 - GERÇEK YAPAY ZEKA ({NUM_CLASSES} SINIFLI) EĞİTİMİ")
    print("==========================================================\n")
    
    all_cnn_x = []
    all_labels = []
    
    # MIT-BIH (Aritmi ve Ritim Sınıfları)
    # 5 classes * 3 records * 600 seconds (10 mins) = 150 mins of data
    physionet_records = [
        ("mitdb", "100", "NORM_MIT"), ("mitdb", "101", "NORM_MIT"), ("mitdb", "103", "NORM_MIT"),
        ("mitdb", "109", "L_MIT"), ("mitdb", "111", "L_MIT"), ("mitdb", "207", "L_MIT"),
        ("mitdb", "118", "R_MIT"), ("mitdb", "124", "R_MIT"), ("mitdb", "212", "R_MIT"),
        ("mitdb", "106", "V_MIT"), ("mitdb", "119", "V_MIT"), ("mitdb", "200", "V_MIT"),
        ("mitdb", "209", "A_MIT"), ("mitdb", "220", "A_MIT"), ("mitdb", "223", "A_MIT")
    ]
    for db, rec, cls in physionet_records:
        x, y = fetch_from_physionet(db, rec, cls, duration=600)
        all_cnn_x.extend(x)
        all_labels.extend(y)
        
    # PTB-XL (10 Sınıf)
    print("\nPTB-XL dosyaları dengeli olarak çekiliyor...")
    ptbxl_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    records_dict = get_ptbxl_patients(ptbxl_path, num_patients_per_class=400) # 400 records * 10 = 4,000 records
    
    for cls, patients in records_dict.items():
        print(f"[PTB-XL] Sınıf: {cls} -> {len(patients)} hasta okunuyor...")
        for rec in patients:
            x, y = fetch_from_ptbxl_local(ptbxl_path, rec, cls)
            all_cnn_x.extend(x)
            all_labels.extend(y)
            
    if not all_cnn_x:
        print("Hiçbir veri toplanamadı!")
        return
        
    X_train = np.array(all_cnn_x)
    y_train = np.array(all_labels)
    
    indices = np.random.permutation(len(X_train))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    X_t, y_t = X_train[train_idx], y_train[train_idx]
    X_v, y_v = X_train[val_idx], y_train[val_idx]
    
    print(f"\n[BİLGİ] GERÇEK EĞİTİM İÇİN {len(X_train)} EKG vuruşu hazır!")
    
    print(f"\n--- 1D-CNN + BiLSTM Modeli Eğitiliyor ({NUM_CLASSES} SINIF) ---")
    trainer = TrainingPipeline()
    trainer.num_classes = NUM_CLASSES
    result = trainer.run(X_t, y_t, X_v, y_v, epochs=40, batch_size=128) # Train more thoroughly
    
    best_model = result['best_model']
    torch.save(best_model.state_dict(), "ecg_model_weights_15class.pth")
    
    print("\nEĞİTİM BAŞARIYLA TAMAMLANDI! Model dosyaları kaydedildi (Kopya yok).")

if __name__ == "__main__":
    main()
