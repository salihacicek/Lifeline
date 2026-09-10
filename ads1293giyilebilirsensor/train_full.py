# -*- coding: utf-8 -*-
"""
Kapsamlı (Tam) Yapay Zeka Eğitim Scripti
MIT-BIH veritabanındaki tüm hastaları okuyup 30 Epoch ile profesyonelce eğitir.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wfdb
from collections import Counter

# Proje dizinini yola ekle ki diğer modülleri bulabilsin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module1_data_acquisition import PhysioNetLoader
from module2_signal_processing import SignalCleaningPipeline
from module5_deep_learning import get_device, ECGHybridModel
from module6_training import FocalLoss

def run_full_training():
    print("🚀 Veriler Toplanıyor (Sadece 1 Kere İndirilecek)...")

    all_records = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234"
    ]
    
    device = get_device()
    fs = 360
    target_len = 288
    
    X_all = []
    y_all = []
    
    symbol_map = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}

    # 1. ADIM: VERİLERİ RAM'E TOPLA
    for rec in all_records:
        try:
            loader = PhysioNetLoader(record_id=rec)
            data = loader.fetch_record(duration_seconds=300) 
            ecg = data["ecg_uv"]
            
            cleaner = SignalCleaningPipeline(fs=fs)
            clean_ecg = cleaner.run(ecg, normalize=True)["ecg_normalized"]
            
            annotation = wfdb.rdann(rec, 'atr', sampfrom=0, sampto=108000, pn_dir='mitdb')
            true_peaks = annotation.sample
            true_symbols = annotation.symbol
            
            for peak, symbol in zip(true_peaks, true_symbols):
                if symbol in symbol_map: 
                    label = symbol_map[symbol]
                    start = peak - int(fs*0.3)
                    end = peak + int(fs*0.5)
                    if start >= 0 and end <= len(clean_ecg):
                        beat = clean_ecg[start:end]
                        if len(beat) < target_len:
                            beat = np.pad(beat, (0, target_len - len(beat)))
                        else:
                            beat = beat[:target_len]
                        X_all.append(beat)
                        y_all.append(label)
            print(f"[{rec}] İndirildi ve RAM'e eklendi.")
        except Exception as e:
            print(f"[{rec}] atlandı: {e}")

    # 2. ADIM: EĞİTİM (30 EPOCH)
    print("\n🧠 Tüm Hastalar Toplandı! Şimdi 30 Turluk (Epoch) Derin Öğrenme Başlıyor...")
    
    X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(np.array(y_all), dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Sınıf Dengesizliği için Ağırlıklar (Weights): Normal (0) çok var, cezası az olsun. RBBB (2) az var, cezası ağır olsun.
    # Örnek Ağırlıklar: Normal: 0.1, LBBB: 1.0, RBBB: 1.0, PVC: 1.0, APC: 1.0
    weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    
    model = ECGHybridModel(in_channels=1, num_classes=5).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=weights) # Sınıf ağırlıklı Focal Loss!
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        print(f"Tur (Epoch) [{epoch+1}/{epochs}] | Ortalama Hata Payı: {total_loss/len(dataloader):.4f} | İsabet Oranı: %{acc:.2f}")

    weights_path = os.path.join(os.path.dirname(__file__), 'ecg_model_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    print(f"\n✅ 30 TURLUK KUSURSUZ EĞİTİM BAŞARIYLA TAMAMLANDI!")
    print(f"🧠 Yapay zeka ağırlıkları (Beyin) güncellendi:\n👉 {weights_path}")
    print("\nArtık 'python gui_app.py' komutuyla 231 numaralı hastayı seçip o RBBB'yi anında bulmasını izleyebilirsiniz!")

if __name__ == "__main__":
    run_full_training()
