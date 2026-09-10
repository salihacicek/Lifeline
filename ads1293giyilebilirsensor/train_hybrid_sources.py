import os
import ast
import pandas as pd
import numpy as np
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

CLASS_MAPPING = {
    "N": 0,    # Normal (PhysioNet - MIT-BIH)
    "L": 1,    # LBBB (PhysioNet - MIT-BIH)
    "R": 2,    # RBBB (PhysioNet - MIT-BIH)
    "V": 3,    # PVC (PhysioNet - MIT-BIH)
    "A": 4,    # APC (PhysioNet - MIT-BIH)
    "AF": 5,   # Atriyal Fibrilasyon (Yerel PTB-XL)
    "MI": 6    # Miyokard Enfarktüsü (Yerel PTB-XL)
}

def process_signal_and_extract_features(raw_signal, fs, target_class):
    """Sinyali temizler, özelliklerini çıkarır ve CNN/XGB için hazırlar."""
    pipeline2 = SignalCleaningPipeline(fs=fs)
    cleaned_signal = pipeline2.run(raw_signal)["ecg_cleaned"]
    
    pipeline3 = FeatureExtractionPipeline()
    features = pipeline3.run(cleaned_signal, fs=fs)
    
    cnn_X = []
    labels = []
    # Segments listesinden beat'leri çek (segmentler dict'teki "heartbeat_segments" anahtarındadır)
    for segment in features.get("heartbeat_segments", []):
        cnn_X.append(segment)
        labels.append(target_class)
    return cnn_X, labels

def fetch_from_physionet(db: str, record_id: str, target_class_str: str, duration: int = 120):
    """PhysioNet veritabanından (MIT-BIH) online veri çeker."""
    print(f"[Kaynak 1: PhysioNet] {db}/{record_id} çekiliyor (Sınıf: {target_class_str})...")
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
    print(f"[Kaynak 2: Yerel PTB-XL] {record_filename} okunuyor (Sınıf: {target_class_str})...")
    
    record_path = os.path.join(ptbxl_base_path, record_filename)
    
    try:
        # Lead I (0. kanal) 100 Hz EKG verisini çekiyoruz
        record = wfdb.rdrecord(record_path, channels=[0])
        ecg_mv = record.p_signal[:, 0]
        fs = record.fs
        
        # ADS1293 formatına simüle edelim (mV -> uV -> 24-bit ADC)
        ecg_uv = ecg_mv * 1000.0
        scaler = ADCScaler(v_ref=2.4)
        raw_adc_float = ecg_uv / scaler.v_lsb_uv
        raw_adc_24bit = np.clip(
            np.round(raw_adc_float).astype(np.int32),
            -scaler.ADC_MAX,
            scaler.ADC_MAX - 1,
        )
        
        return process_signal_and_extract_features(raw_adc_24bit, fs, CLASS_MAPPING[target_class_str])
    except Exception as e:
        print(f"Hata ({record_filename}): {e}")
        return [], []

def get_ptbxl_patients(ptbxl_base_path: str, num_patients_per_class: int = 20):
    """ptbxl_database.csv'yi tarayıp AF ve MI hastalarından n'er adet seçer."""
    csv_path = os.path.join(ptbxl_base_path, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return [], []
        
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
            af_records.append(row['filename_lr'])
        elif any(code in codes for code in mi_scp_codes) and len(mi_records) < num_patients_per_class:
            mi_records.append(row['filename_lr'])
            
    return af_records, mi_records

def main():
    print("==========================================================")
    print("ADS1293 - ÇOKLU KAYNAK HİBRİT EĞİTİM BAŞLATILIYOR")
    print("Kaynak 1: MIT PhysioNet (Hastalıklar: A, B, C, D, E)")
    print("Kaynak 2: Yerel PTB-XL (Hastalıklar: F, G)")
    print("==========================================================\n")
    
    all_cnn_x = []
    all_labels = []
    
    # ---------------------------------------------------------
    # KAYNAK 1: MIT PhysioNet (Hastalık A, B, C, D, E)
    # ---------------------------------------------------------
    physionet_records = [
        ("mitdb", "100", "N"), # Hastalık A (Normal)
        ("mitdb", "111", "L"), # Hastalık B
        ("mitdb", "118", "R"), # Hastalık C
        ("mitdb", "119", "V"), # Hastalık D
        ("mitdb", "209", "A")  # Hastalık E
    ]
    for db, rec, cls in physionet_records:
        x, y = fetch_from_physionet(db, rec, cls, duration=30)
        all_cnn_x.extend(x)
        all_labels.extend(y)
        
    # ---------------------------------------------------------
    # KAYNAK 2: Yerel PTB-XL (Hastalık F, G)
    # ---------------------------------------------------------
    ptbxl_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    
    print(f"\n[BİLGİ] PTB-XL içinden rastgele 20 AF, 20 MI hastası seçiliyor...")
    af_patients, mi_patients = get_ptbxl_patients(ptbxl_path, num_patients_per_class=20)
    
    for rec in af_patients:
        x, y = fetch_from_ptbxl_local(ptbxl_path, rec, "AF")
        all_cnn_x.extend(x)
        all_labels.extend(y)
        
    for rec in mi_patients:
        x, y = fetch_from_ptbxl_local(ptbxl_path, rec, "MI")
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
    
    # ---------------------------------------------------------
    # MODELLERİN EĞİTİMİ (A, B, C, D, E, F, G Birleştirilmiş Halde)
    # ---------------------------------------------------------
    print("\n--- 1D-CNN + BiLSTM Modeli Eğitiliyor (Kaynak 1 + Kaynak 2 Birleşimi) ---")
    trainer = TrainingPipeline()
    result = trainer.run(X_t, y_t, X_v, y_v, epochs=30, batch_size=32)
    
    best_model = result['best_model']
    torch.save(best_model.state_dict(), "ecg_model_weights_7class.pth")
    
    print("\n--- XGBoost Modeli Eğitiliyor ---")
    xgb_train_x = np.array([[np.mean(beat), np.std(beat), np.max(beat), np.min(beat)] for beat in X_train])
    xgb_model = XGBoostClassifier(num_classes=7, n_estimators=50)
    xgb_model.train(xgb_train_x, y_train)
    
    with open("xgboost_weights.pkl", "wb") as f:
        pickle.dump(xgb_model.model, f)
        
    print("\n✅ TÜM EĞİTİM TAMAMLANDI! Yapay zeka artık iki farklı sitenin verilerini tek havuzda birleştirerek 7 hastalığı aynı anda tahmin edebilir.")

if __name__ == "__main__":
    main()
