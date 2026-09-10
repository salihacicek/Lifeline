# -*- coding: utf-8 -*-
"""
=============================================================================
HİBRİT YAPAY ZEKA EĞİTİMİ (DEEP LEARNING + MACHINE LEARNING)
=============================================================================
Bu script, MIT-BIH veritabanındaki uzman doktor etiketli verileri kullanarak:
1. 1D-CNN + BiLSTM modelini (Derin Öğrenme) ham EKG sinyalleri üzerinde eğitir.
2. XGBoost modelini (Makine Öğrenmesi) kardiyolojik özellikler (QRS, PR, vb.) üzerinde eğitir.

Lifeline verilerinde (Osman Bey'in verilerinde) doktor etiketi olmadığı için, 
önce bu algoritmalar MIT-BIH'teki yüz binlerce atım ile eğitilip "Doktor" seviyesine
çıkarılacak, ardından Lifeline verilerinde test edilecektir.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wfdb
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module1_data_acquisition import PhysioNetLoader
from module2_signal_processing import SignalCleaningPipeline
from module3_feature_extraction import PanTompkinsDetector, PQRSTExtractor
from module5_deep_learning import get_device, ECGHybridModel, XGBoostClassifier
from module6_training import FocalLoss

def extract_features():
    print("🚀 Uzman Doktor Etiketli Veriler (MIT-BIH) Toplanıyor...")

    all_records = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
        "213", "214", "215", "217", "219", "220", "221", "222", "223", "230", "231", "232", "233", "234"
    ]
    
    fs = 360
    target_len = 288
    
    X_cnn_all = []
    X_xgb_all = []
    y_all = []
    
    symbol_map = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}

    cleaner = SignalCleaningPipeline(fs=fs)
    pqrst = PQRSTExtractor(fs=fs)

    for rec in all_records:
        try:
            loader = PhysioNetLoader(record_id=rec)
            # Daha fazla veri için 300 saniye (5 dakika) alıyoruz
            data = loader.fetch_record(duration_seconds=300) 
            ecg = data["ecg_uv"]
            clean_ecg = cleaner.run(ecg, normalize=True)["ecg_normalized"]
            
            annotation = wfdb.rdann(rec, 'atr', sampfrom=0, sampto=108000, pn_dir='mitdb')
            true_peaks = annotation.sample
            true_symbols = annotation.symbol
            
            # RR aralıklarını hesapla (XGBoost için gerekli)
            rr_intervals = np.diff(true_peaks) / fs * 1000.0 # ms
            
            for i in range(1, len(true_peaks) - 1):
                peak = true_peaks[i]
                symbol = true_symbols[i]
                
                if symbol in symbol_map: 
                    label = symbol_map[symbol]
                    start = peak - int(fs * 0.3)
                    end = peak + int(fs * 0.5)
                    
                    if start >= 0 and end <= len(clean_ecg):
                        # CNN İÇİN: Ham Segment
                        beat = clean_ecg[start:end]
                        if len(beat) < target_len:
                            beat = np.pad(beat, (0, target_len - len(beat)))
                        else:
                            beat = beat[:target_len]
                        
                        # XGBoost İÇİN: PQRST Morfolojisi
                        waves = pqrst.extract_waves(clean_ecg, peak)
                        intervals = pqrst.compute_intervals(waves)
                        
                        qrs_width = intervals["QRS_Width_ms"] if not np.isnan(intervals["QRS_Width_ms"]) else 100.0
                        pr_interval = intervals["PR_Interval_ms"] if not np.isnan(intervals["PR_Interval_ms"]) else 160.0
                        qt_interval = intervals["QT_Interval_ms"] if not np.isnan(intervals["QT_Interval_ms"]) else 400.0
                        pre_rr = rr_intervals[i-1]
                        post_rr = rr_intervals[i]
                        
                        xgb_features = [qrs_width, pr_interval, qt_interval, pre_rr, post_rr]
                        
                        X_cnn_all.append(beat)
                        X_xgb_all.append(xgb_features)
                        y_all.append(label)
                        
            print(f"[{rec}] Tamamlandı.")
        except Exception as e:
            print(f"[{rec}] Atlandı: {e}")

    return np.array(X_cnn_all), np.array(X_xgb_all), np.array(y_all)

def train_hybrid_models():
    # 1. Veri Hazırlığı
    X_cnn, X_xgb, y = extract_features()
    print(f"\n✅ Toplam {len(y)} adet atım (beat) çıkarıldı.")
    
    device = get_device()
    print(f"🖥️  Donanım Hızlandırma Aktif: {device}")
    
    # 2. XG-BOOST EĞİTİMİ (Makine Öğrenmesi)
    print("\n🌲 XGBoost Modeli Eğitiliyor (Klinik PQRST Verileri Üzerinde)...")
    xgb_model = XGBoostClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
    xgb_model.train(X_xgb, y)
    
    xgb_path = os.path.join(os.path.dirname(__file__), 'xgboost_weights.pkl')
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model.model, f)
    print("✅ XGBoost eğitimi tamamlandı ve kaydedildi!")

    # 3. 1D-CNN + BiLSTM EĞİTİMİ (Derin Öğrenme)
    print("\n🧠 1D-CNN + BiLSTM Modeli Eğitiliyor (Ham EKG Sinyali Üzerinde)...")
    X_tensor = torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    cnn_model = ECGHybridModel(in_channels=1, num_classes=5).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=weights)
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 20 # İstenilen şekilde 20-30 tur
    for epoch in range(epochs):
        cnn_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            out = cnn_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Kayıp: {total_loss/len(dataloader):.4f} | İsabet: %{acc:.2f}")

    cnn_path = os.path.join(os.path.dirname(__file__), 'ecg_model_weights.pth')
    torch.save(cnn_model.state_dict(), cnn_path)
    
    print("\n🎯 EĞİTİM BAŞARIYLA BİTTİ! HER İKİ MODEL DE KAYDEDİLDİ.")
    print("Artık bu uzman modelleri 'gui_app.py' içinde Lifeline verilerinde test edebiliriz!")

if __name__ == "__main__":
    train_hybrid_models()
