import os
import sys
import numpy as np

# Modülleri görebilmesi için pipeline klasörünü ekliyoruz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module1_data_acquisition import RawDataPipeline
from module2_signal_processing import SignalCleaningPipeline
from module3_feature_extraction import PanTompkinsDetector, HRVAnalyzer
from module5_deep_learning import get_device, ECGHybridModel
import torch

def main():
    print("\n" + "="*60)
    print("🏥 ADS1293 EKG TELEMETRİ SİSTEMİ - UÇTAN UCA ANALİZ")
    print("="*60)

    # ---------------------------------------------------------
    # ADIM 1: Veri Alma (Sensör veya İnternet)
    # ---------------------------------------------------------
    print("\n[1/4] Fizyolojik EKG Verisi Çekiliyor...")
    raw_pipeline = RawDataPipeline(record_id="100", duration_seconds=10.0, simulate_packet_loss=False)
    raw_result = raw_pipeline.run()
    ecg_uv = raw_result["ecg_uv"]
    fs = raw_result["fs"]
    print(f"      -> Veri alındı. Örnekleme Frekansı: {fs} Hz, Uzunluk: {len(ecg_uv)} örnek.")

    # ---------------------------------------------------------
    # ADIM 2: Sinyal Temizleme
    # ---------------------------------------------------------
    print("\n[2/4] Sinyal Gürültüden Arındırılıyor...")
    clean_pipeline = SignalCleaningPipeline(fs=fs)
    clean_result = clean_pipeline.run(ecg_uv, normalize=True)
    cleaned_signal = clean_result["ecg_normalized"]
    print("      -> Dalgacık (Wavelet) dönüşümü ve Z-Score normalizasyonu tamamlandı.")

    # ---------------------------------------------------------
    # ADIM 3: Özellik Çıkarımı (R-Tepe ve Kalp Atışı Tespiti)
    # ---------------------------------------------------------
    print("\n[3/4] R-Tepeleri Tespit Ediliyor ve Segmentlere Ayrılıyor...")
    pt_detector = PanTompkinsDetector(fs=fs)
    r_peaks = pt_detector.detect(cleaned_signal)
    
    # Sadece ilk kalp atışını alalım (Tahmin için)
    rp = r_peaks[2] # Stabil olması için ortadan bir tepe alıyoruz
    beat_segment = cleaned_signal[rp - int(fs*0.3) : rp + int(fs*0.5)]
    print(f"      -> Toplam {len(r_peaks)} R-tepesi bulundu. Örnek bir kalp atımı kesildi.")

    # ---------------------------------------------------------
    # ADIM 4: Yapay Zeka Teşhisi (Deep Learning)
    # ---------------------------------------------------------
    print("\n[4/4] Yapay Zeka Teşhis Modeli (1D-CNN + BiLSTM) Devrede...")
    device = get_device()
    model = ECGHybridModel(in_channels=1, num_classes=5).to(device)
    model.eval() # Test (Tahmin) Modu
    
    # Sinyali PyTorch tensörüne çevirme (Batch, Channel, Length)
    # Uzunluğu 288 olacak şekilde padding veya kesme işlemi
    target_len = 288
    if len(beat_segment) < target_len:
        beat_segment = np.pad(beat_segment, (0, target_len - len(beat_segment)))
    else:
        beat_segment = beat_segment[:target_len]

    input_tensor = torch.tensor(beat_segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Model Tahmini
    with torch.no_grad():
        output_logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output_logits, dim=1)[0].cpu().numpy()
        prediction = np.argmax(probabilities)

    siniflar = {0: "Normal Sinüs Ritmi", 1: "LBBB (Sol Dal Bloğu)", 2: "RBBB (Sağ Dal Bloğu)", 3: "PVC (Erken Karıncıksal Vuru)", 4: "APC (Erken Kulakçıksal Vuru)"}
    
    print("\n" + "="*60)
    print(f"✅ YAPAY ZEKA TAHMİNİ: {siniflar.get(prediction, 'Bilinmiyor')}")
    print(f"   Güven Oranı: %{np.max(probabilities)*100:.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
