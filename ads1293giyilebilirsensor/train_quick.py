# -*- coding: utf-8 -*-
"""
Hızlı Yapay Zeka Eğitim Scripti
Modül 5'te kurduğumuz devasa mimariyi, sistemi test edebilmeniz 
için ufak bir veri setiyle çok hızlı bir şekilde eğitir.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Proje dizinini yola ekle ki diğer modülleri bulabilsin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module1_data_acquisition import PhysioNetLoader
from module2_signal_processing import SignalCleaningPipeline
from module3_feature_extraction import PanTompkinsDetector
from module5_deep_learning import get_device, ECGHybridModel

def run_quick_training():
    print("🚀 Hızlı Yapay Zeka Eğitimi Başlıyor...")
    print("Veriler PhysioNet sunucusundan indiriliyor (Lütfen bekleyin)...\n")

    # Temsili 5 hasta ve hastalık etiketleri
    # 0: Normal, 1: LBBB, 2: RBBB, 3: PVC, 4: APC
    records = {"100": 0, "109": 1, "118": 2, "119": 3, "232": 4}
    X_all = []
    y_all = []
    fs = 360
    target_len = 288

    for rec, label in records.items():
        print(f"[{rec}] numaralı hastanın EKG verisi indiriliyor ve parçalanıyor...")
        try:
            loader = PhysioNetLoader(record_id=rec)
            # Sadece 60 saniyelik veri çekiyoruz (İşlem hızlı sürsün diye)
            data = loader.fetch_record(duration_seconds=60)
            ecg = data["ecg_uv"]
            
            # Sinyali parazitlerden temizle (Modül 2)
            cleaner = SignalCleaningPipeline(fs=fs)
            res = cleaner.run(ecg, normalize=True)
            clean_ecg = res["ecg_normalized"]
            
            # Kalp atışlarını tespit et (Modül 3 Pan-Tompkins)
            pt = PanTompkinsDetector(fs=fs)
            r_peaks = pt.detect(clean_ecg)
            
            # Her bir R tepesini merkeze alıp sinyali dilimle
            for r in r_peaks:
                start = r - int(fs*0.3)
                end = r + int(fs*0.5)
                if start >= 0 and end <= len(clean_ecg):
                    beat = clean_ecg[start:end]
                    if len(beat) < target_len:
                        beat = np.pad(beat, (0, target_len - len(beat)))
                    else:
                        beat = beat[:target_len]
                    X_all.append(beat)
                    y_all.append(label)
        except Exception as e:
            print(f"[{rec}] Hasta indirilirken hata: {e}")

    # Verileri PyTorch Tensor formuna (Matrislere) çevir
    X = torch.tensor(np.array(X_all), dtype=torch.float32).unsqueeze(1) # Boyut: (N, 1, 288)
    y = torch.tensor(np.array(y_all), dtype=torch.long)

    # Mac M2 Çipini (MPS) veya İşlemciyi bul
    device = get_device()
    print(f"\n💻 Toplam {len(X)} kalp atışı çıkartıldı.")
    print(f"🛠️ Donanım Hızlandırma: Eğitim {device} (İşlemci/GPU) üzerinde yapılıyor...")
    
    # Yeni ve boş beyni oluştur (Modül 5)
    model = ECGHybridModel(in_channels=1, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss() # Hata ölçüm fonksiyonu
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Öğrenme hızı

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("\n🧠 Derin Öğrenme Döngüsü (Training) Başlıyor (10 Tur)...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()  # Hataları geriye yay (Backpropagation)
            optimizer.step() # Beyindeki nöron ağırlıklarını güncelle
            
            total_loss += loss.item()
            
        print(f"Tur (Epoch) {epoch+1}/10 - Hata Payı (Loss): {total_loss/len(loader):.4f}")

    # Eğitilmiş (Öğrenmiş) beyni bilgisayara sadece 1 MB'lık bir dosya olarak kaydet
    weights_path = os.path.join(os.path.dirname(__file__), 'ecg_model_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    print(f"\n✅ Eğitim mükemmel şekilde tamamlandı!")
    print(f"🧠 Yapay zeka ağırlıkları (Beyin) şu dosyaya kaydedildi:\n👉 {weights_path}")
    print("\nArtık 'python gui_app.py' komutuyla arayüzü başlatabilirsiniz. %100 GERÇEK ZEKA DEVREDE!")

if __name__ == "__main__":
    run_quick_training()
