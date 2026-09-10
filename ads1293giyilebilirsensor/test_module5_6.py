import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Proje dizinini ekleyelim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module5_deep_learning import ECGHybridModel, XGBoostClassifier, get_device
from module6_training import FocalLoss, TrainingPipeline, ECGTrainer

def main():
    print("=" * 70)
    print("=== Modül 5-6 Entegre Testi Başlatılıyor ===")
    print("=" * 70)
    
    device = get_device()
    print(f"Tespit Edilen Cihaz: {device}")
    
    # 1. Sentetik Veri Oluşturma
    print("\n1. Sentetik Veri Oluşturuluyor...")
    # Deep Learning için (Batch, 1, Seq_len)
    num_samples = 100
    seq_len = 288 # Örnek ECG segment uzunluğu
    num_classes = 5
    
    # DL Verisi
    X_dl = np.random.randn(num_samples, seq_len)
    y_dl = np.random.randint(0, num_classes, num_samples)
    
    # XGBoost için tablo verisi
    num_features = 15
    X_xgb = np.random.randn(num_samples, num_features)
    y_xgb = np.random.randint(0, num_classes, num_samples)
    
    print(f"DL Veri Boyutu: {X_dl.shape}")
    print(f"XGB Veri Boyutu: {X_xgb.shape}")
    
    # 2. Mimari Testi: ECGHybridModel
    print("\n2. ECGHybridModel Test Ediliyor...")
    model = ECGHybridModel(in_channels=1, num_classes=num_classes)
    model.to(device)
    
    # Forward pass testi
    sample_input = torch.tensor(X_dl[:4], dtype=torch.float32).unsqueeze(1).to(device) # Batch size: 4
    with torch.no_grad():
        outputs = model(sample_input)
    
    print(f"Model Çıktı Boyutu (Forward Pass): {outputs.shape} (Beklenen: [4, 5])")
    
    # 3. XGBoostClassifier Testi
    print("\n3. XGBoostClassifier Test Ediliyor...")
    xgb_model = XGBoostClassifier()
    xgb_model.train(X_xgb[:80], y_xgb[:80]) # Train
    xgb_preds = xgb_model.predict(X_xgb[80:]) # Eval
    print(f"XGBoost Tahmin Boyutu: {xgb_preds.shape} (Beklenen: [20,])")
    
    # 4. Focal Loss Testi
    print("\n4. Focal Loss Test Ediliyor...")
    criterion = FocalLoss(gamma=2.0)
    sample_targets = torch.tensor(y_dl[:4], dtype=torch.long).to(device)
    loss_val = criterion(outputs, sample_targets)
    print(f"Focal Loss Değeri: {loss_val.item():.4f}")
    
    # 5. Eğitim Döngüsü (Training Pipeline) Testi
    print("\n5. Eğitim Döngüsü (Training Pipeline) Başlatılıyor (3 Epoch)...")
    pipeline = TrainingPipeline()
    # Veriyi Train/Val olarak ikiye böl
    X_train, y_train = X_dl[:80], y_dl[:80]
    X_val, y_val = X_dl[80:], y_dl[80:]
    
    results = pipeline.run(X_train, y_train, X_val, y_val, epochs=3, batch_size=16)
    history = results['history']
    
    print("Eğitim Tamamlandı!")
    
    # 6. Görselleştirme
    print("\n6. Eğitim Kayıp Eğrisi Kaydediliyor...")
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
    plt.title('Kayıp (Loss) Eğrisi')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o', color='green')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', marker='s', color='orange')
    plt.title('Doğruluk (Accuracy) Eğrisi')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('module5_6_pipeline_output.png', dpi=150)
    print("Kayıt tamamlandı: module5_6_pipeline_output.png")

if __name__ == '__main__':
    main()
