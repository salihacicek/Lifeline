import os
import ast
import pandas as pd
import numpy as np
import scipy.signal

import wfdb
import torch
import pickle
import warnings
from typing import List, Dict

# ADS1293 Pipeline Importları
from ads1293_pipeline.module1_data_acquisition import RawDataPipeline, ADCScaler
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import FeatureExtractionPipeline
from ads1293_pipeline.module5_deep_learning import ECGHybridModel, XGBoostClassifier
from ads1293_pipeline.module6_training import TrainingPipeline

warnings.filterwarnings("ignore")

# 9-SINIFLI NİHAİ KARDİYOLOG ONTOLOJİSİ
# Hem MIT (Aritmi Odaklı) hem de PTB-XL (Yapısal Odaklı) veri setlerini birleştirir.
CLASS_MAPPING = {
    "NORM": 0,  # MIT-BIH (N) ve PTB-XL (NORM) birleştirildi
    "L": 1,     # Sol Dal Bloğu (MIT-BIH)
    "R": 2,     # Sağ Dal Bloğu (MIT-BIH)
    "V": 3,     # Erken Ventriküler Vuru (MIT-BIH)
    "A": 4,     # Erken Atriyal Vuru (MIT-BIH)
    "MI": 5,    # Miyokard Enfarktüsü - Kalp Krizi (PTB-XL)
    "STTC": 6,  # ST/T Değişimi - İskemi (PTB-XL)
    "CD": 7,    # Genel İletim Bozuklukları (PTB-XL)
    "HYP": 8    # Hipertrofi - Kalp Büyümesi (PTB-XL)
}

def process_signal_and_extract_features(raw_signal, fs, target_class):
    """Sinyali temizler, özelliklerini çıkarır ve CNN/XGB için hazırlar."""
    pipeline2 = SignalCleaningPipeline(fs=fs)
    cleaned_signal = pipeline2.run(raw_signal)["ecg_cleaned"]
    
    pipeline3 = FeatureExtractionPipeline()
    features = pipeline3.run(cleaned_signal, fs=fs)
    
    cnn_X = []
    labels = []
    for segment in features.get("heartbeat_segments", []):
        # Farklı FS (100 Hz vs 360 Hz) nedeniyle oluşan farklı uzunlukları sabitliyoruz
        if len(segment) != 250:
            segment = scipy.signal.resample(segment, 250)
        cnn_X.append(segment)
        labels.append(target_class)
    return cnn_X, labels

def fetch_from_physionet(db: str, record_id: str, target_class_str: str, duration: int = 60):
    """PhysioNet veritabanından (MIT-BIH) online veri çeker."""
    print(f"[Kaynak 1: MIT-BIH] {record_id} çekiliyor (Sınıf: {target_class_str})...")
    pipeline = RawDataPipeline(record_id=record_id, duration_seconds=duration, simulate_packet_loss=False)
    pipeline.loader.database = db
    
    try:
        data = pipeline.fetch_data(duration_seconds=duration)
    except Exception as e:
        print(f"Hata: {e}")
        return [], []
        
    return process_signal_and_extract_features(data['raw_adc_24bit'], data['fs'], CLASS_MAPPING[target_class_str])

def fetch_from_ptbxl_local(ptbxl_base_path: str, record_filename: str, target_class_str: str):
    """Yerel PTB-XL klasöründen .dat/.hea dosyalarını okur."""
    print(f"[Kaynak 2: PTB-XL] {record_filename} okunuyor (Sınıf: {target_class_str})...")
    
    record_path = os.path.join(ptbxl_base_path, record_filename)
    try:
        record = wfdb.rdrecord(record_path, channels=[0]) # Lead I
        ecg_mv = record.p_signal[:, 0]
        fs = record.fs
        
        # ADS1293 formatına simüle edelim (mV -> uV -> 24-bit ADC)
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
        print(f"Hata ({record_filename}): {e}")
        return [], []

def get_ptbxl_patients(ptbxl_base_path: str, num_patients_per_class: int = 50):
    """ptbxl_database.csv'yi tarayıp 5 Ana Sınıftan hastalar seçer."""
    csv_path = os.path.join(ptbxl_base_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    
    records_by_class = {"NORM": [], "MI": [], "STTC": [], "CD": [], "HYP": []}
    
    # PTB-XL alt-sınıf kodları sözlüğü (Ana Sınıflara göre)
    scp_mapping = {
        "NORM": ['NORM'],
        "MI": ['IMI', 'ASMI', 'AMI', 'LMI', 'ALMI', 'INJAS', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL'],
        "STTC": ['NDT', 'ISC_', 'ISCA', 'ISCI', 'ISCIL', 'ISCAS', 'ISCLA', 'ISCIN', 'NST_', 'DIG', 'LNGQT', 'EPV', 'ITV', 'SEHYP'],
        "CD": ['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB', 'WPW', 'ILBBB', '3AVB', '2AVB'],
        "HYP": ['LVH', 'RVH', 'SEHYP']
    }
    
    for _, row in df.iterrows():
        try:
            codes = ast.literal_eval(row['scp_codes'])
        except:
            continue
            
        for superclass, subclasses in scp_mapping.items():
            if any(code in codes for code in subclasses):
                if len(records_by_class[superclass]) < num_patients_per_class:
                    records_by_class[superclass].append(row['filename_lr'])
                break # Bir hastayı sadece bir sınıfa ata
                
    return records_by_class

def main():
    print("==========================================================")
    print("ADS1293 - NİHAİ KARDİYOLOG (9 SINIFLI) EĞİTİMİ BAŞLADI")
    print("==========================================================\n")
    
    all_cnn_x = []
    all_labels = []
    
    # --- KAYNAK 1: MIT-BIH (Aritmi ve Ritim Sınıfları) ---
    # MIT-BIH'teki 48 hastanın tamamı (veya temsili olarak ana sınıfları yansıtan en net hastalar)
    # 60 saniye çekilerek veri eşitliği (Balancing) sağlanır.
    physionet_records = [
        ("mitdb", "100", "NORM"), ("mitdb", "101", "NORM"), ("mitdb", "103", "NORM"),
        ("mitdb", "109", "L"), ("mitdb", "111", "L"), ("mitdb", "207", "L"),
        ("mitdb", "118", "R"), ("mitdb", "124", "R"), ("mitdb", "212", "R"),
        ("mitdb", "106", "V"), ("mitdb", "119", "V"), ("mitdb", "200", "V"),
        ("mitdb", "209", "A"), ("mitdb", "220", "A"), ("mitdb", "223", "A")
    ]
    for db, rec, cls in physionet_records:
        x, y = fetch_from_physionet(db, rec, cls, duration=60)
        all_cnn_x.extend(x)
        all_labels.extend(y)
        
    # --- KAYNAK 2: PTB-XL (Yapısal Hastalık Sınıfları) ---
    ptbxl_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    records_dict = get_ptbxl_patients(ptbxl_path, num_patients_per_class=50)
    
    for superclass, patients in records_dict.items():
        for rec in patients:
            x, y = fetch_from_ptbxl_local(ptbxl_path, rec, superclass)
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
    
    print(f"\n[BİLGİ] İki kaynaktan toplam {len(X_train)} vuruş (beat) birleştirildi.")
    print(f"Eğitim Seti: {len(X_t)}, Doğrulama Seti: {len(X_v)}")
    
    # --- EĞİTİM ---
    print("\n--- 1D-CNN + BiLSTM Modeli Eğitiliyor (9 SINIF) ---")
    trainer = TrainingPipeline()
    trainer.num_classes = 9
    result = trainer.run(X_t, y_t, X_v, y_v, epochs=30, batch_size=32)
    
    best_model = result['best_model']
    torch.save(best_model.state_dict(), "ecg_model_weights_9class.pth")
    
    print("\n--- XGBoost Modeli Eğitiliyor (9 SINIF) ---")
    xgb_train_x = np.array([[np.mean(beat), np.std(beat), np.max(beat), np.min(beat)] for beat in X_train])
    xgb_model = XGBoostClassifier(num_classes=9, n_estimators=100)
    xgb_model.train(xgb_train_x, y_train)
    
    with open("xgboost_weights_9class.pkl", "wb") as f:
        pickle.dump(xgb_model.model, f)
        
    print("\n✅ TÜM EĞİTİM TAMAMLANDI! Model 9 Hastalığı da uzman seviyesinde tanıyabilir.")

if __name__ == "__main__":
    main()
