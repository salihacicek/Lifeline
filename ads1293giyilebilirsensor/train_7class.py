import os
import numpy as np
import torch
import pickle
import warnings
from typing import List, Dict

# ADS1293 Pipeline Importları
from ads1293_pipeline.module1_data_acquisition import RawDataPipeline
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import FeatureExtractionPipeline
from ads1293_pipeline.module5_deep_learning import ECGHybridModel, XGBoostClassifier, get_device
from ads1293_pipeline.module6_training import TrainingPipeline

# Hataları gizle
warnings.filterwarnings("ignore")

# 7 Hastalık Sınıfı Eşleşmesi
CLASS_MAPPING = {
    "N": 0,    # Normal
    "L": 1,    # LBBB
    "R": 2,    # RBBB
    "V": 3,    # PVC
    "A": 4,    # APC
    "AF": 5,   # Atriyal Fibrilasyon
    "MI": 6    # Miyokard Enfarktüsü (ST+)
}

def fetch_and_process_record(db: str, record_id: str, target_class_str: str, duration: int = 120):
    """Veriyi çeker, temizler, özellikleri çıkarır ve beat'leri döner."""
    print(f"\n[{db}/{record_id}] Veri çekiliyor (Sınıf: {target_class_str})...")
    
    pipeline1 = RawDataPipeline(record_id=record_id, duration_seconds=duration, simulate_packet_loss=False)
    # PhysioNetLoader'dan veri çek
    pipeline1.loader.database = db
    
    try:
        data1 = pipeline1.fetch_data(duration_seconds=duration)
    except Exception as e:
        print(f"Hata: {e}")
        return [], [], []

    raw_signal = data1['raw_adc_24bit']
    fs = data1['fs']
    
    print(f"[{db}/{record_id}] Temizleniyor...")
    pipeline2 = SignalCleaningPipeline(fs=fs)
    cleaned_signal = pipeline2.run(raw_signal)['ecg_cleaned']
    
    print(f"[{db}/{record_id}] Özellikler çıkarılıyor...")
    pipeline3 = FeatureExtractionPipeline()
    features = pipeline3.run(cleaned_signal, fs=fs)
    
    # Tüm beat'leri hedef sınıfa ata (basitleştirilmiş etiketleme)
    target_class = CLASS_MAPPING[target_class_str]
    
    cnn_X = []
    xgb_X = []
    labels = []
    
    # Segmentleri topla
    for segment in features.get("heartbeat_segments", []):
        cnn_X.append(segment)
        labels.append(target_class)
        
    return cnn_X, labels

def main():
    print("ADS1293 - 7 Sınıflı HİBRİT EĞİTİM (CNN + BiLSTM + XGBoost) BAŞLATILIYOR\n")
    print("Veritabanları: MIT-BIH (mitdb), Atrial Fibrillation (afdb), PTB Diagnostic (ptbdb)")
    
    # 1. Eğitim Verisini Topla
    all_cnn_x = []
    all_labels = []
    
    # Veritabanı ve kayıt örnekleri (Demo için kısa süreler)
    records_to_fetch = [
        ("mitdb", "100", "N"),
        ("mitdb", "111", "L"),
        ("mitdb", "118", "R"),
        ("mitdb", "119", "V"),
        ("mitdb", "209", "A"),
        ("afdb", "04015", "AF"),
        ("ptbdb", "patient001/s0010_re", "MI")
    ]
    
    for db, rec, cls in records_to_fetch:
        x, y = fetch_and_process_record(db, rec, cls, duration=30) # Her kayıttan 30 saniye
        if x:
            all_cnn_x.extend(x)
            all_labels.extend(y)
            
    if not all_cnn_x:
        print("HİÇ VERİ ÇEKİLEMEDİ! Lütfen internet bağlantınızı kontrol edin.")
        return
        
    X_train = np.array(all_cnn_x)
    y_train = np.array(all_labels)
    
    # Veriyi Train/Val olarak ikiye böl (80/20)
    indices = np.random.permutation(len(X_train))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_t, y_t = X_train[train_idx], y_train[train_idx]
    X_v, y_v = X_train[val_idx], y_train[val_idx]
    
    print(f"\nToplam Beat Sayısı: {len(X_train)}")
    print(f"Eğitim Seti: {len(X_t)}, Doğrulama Seti: {len(X_v)}")
    
    # 2. CNN + BiLSTM Modeli Eğitimi
    print("\n--- 1D-CNN + BiLSTM + Attention Modeli Eğitiliyor ---")
    trainer = TrainingPipeline()
    # module6_training.py içindeki TrainingPipeline modelinde num_classes=7 olarak güncellendi.
    result = trainer.run(X_t, y_t, X_v, y_v, epochs=3, batch_size=32)
    
    # Modeli kaydet
    best_model = result['best_model']
    torch.save(best_model.state_dict(), "ecg_model_weights_7class.pth")
    print("Derin Öğrenme Modeli Kaydedildi: ecg_model_weights_7class.pth")
    
    # 3. XGBoost Modeli Eğitimi (Demo için Rastgele Özniteliklerle Eğitilir)
    print("\n--- XGBoost Modeli Eğitiliyor ---")
    # Not: XGBoost için normalde PQRST özellik matrisi (örn: HRV, QT süresi) gerekir.
    # Burada demo amaçlı olarak CNN'e giren ham sinyalin basit istatistiklerini veriyoruz.
    xgb_train_x = np.array([[np.mean(beat), np.std(beat), np.max(beat), np.min(beat)] for beat in X_train])
    
    xgb_model = XGBoostClassifier(num_classes=7, n_estimators=50)
    xgb_model.train(xgb_train_x, y_train)
    
    # XGBoost modelini kaydet
    with open("xgboost_weights.pkl", "wb") as f:
        pickle.dump(xgb_model.model, f)
    print("XGBoost Modeli Kaydedildi: xgboost_weights.pkl")
    
    print("\n✅ TÜM EĞİTİM TAMAMLANDI! `gui_app.py` artık 7 sınıfı destekliyor.")

if __name__ == "__main__":
    main()
