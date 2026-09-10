import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Proje dizinini ekleyelim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module1_data_acquisition import RawDataPipeline
from module2_signal_processing import SignalCleaningPipeline
from module3_feature_extraction import PanTompkinsDetector, HRVAnalyzer, SampleEntropyCalculator, TWAAnalyzer
from module4_data_augmentation import TimeSeriesAugmentor, BorderlineSMOTEBalancer

def main():
    print("=" * 70)
    print("=== Modül 3-4 Entegre Testi Başlatılıyor ===")
    print("=" * 70)
    
    # 1. Modül 1 ve 2: Veri Çekme ve Temizleme
    print("\n1. Veri alınıyor ve temizleniyor...")
    raw_pipeline = RawDataPipeline(
        record_id="100",
        channel=0,
        duration_seconds=15.0, # 15 saniyelik daha kısa bir test
        ring_buffer_capacity=4096,
        v_ref=2.4,
        gain=3.5,
        simulate_packet_loss=False
    )
    raw_result = raw_pipeline.run()
    ecg_uv = raw_result["ecg_uv"]
    fs = raw_result["fs"]
    
    clean_pipeline = SignalCleaningPipeline(fs=fs)
    clean_result = clean_pipeline.run(ecg_uv, normalize=True)
    # Modül 2, normalize parametresi true olduğunda ecg_normalized döner
    cleaned_signal = clean_result["ecg_normalized"]
    
    # 2. Modül 3: Özellik Çıkarımı
    print("\n2. Özellikler Çıkarılıyor (Modül 3)...")
    pt_detector = PanTompkinsDetector(fs=fs)
    r_peaks = pt_detector.detect(cleaned_signal)
    rr_intervals = pt_detector.get_rr_intervals(r_peaks)
    
    hrv_analyzer = HRVAnalyzer()
    time_domain = hrv_analyzer.compute_time_domain(rr_intervals)
    freq_domain = hrv_analyzer.compute_frequency_domain(rr_intervals)
    
    sampen_calc = SampleEntropyCalculator()
    entropy = sampen_calc.compute(rr_intervals)
    
    twa_analyzer = TWAAnalyzer()
    twa_ratio = twa_analyzer.compute_twa(cleaned_signal, r_peaks, fs)
    
    print(f"R-tepe sayısı: {len(r_peaks)}")
    print("[HRV Time Domain]:", {k: f"{v:.2f}" for k, v in time_domain.items() if isinstance(v, (int, float))})
    print("[Sample Entropy]:", f"{entropy:.4f}")
    print("[TWA Ratio]:", f"{twa_ratio:.4f}")
    
    # 3. Modül 4: Veri Artırma
    print("\n3. Veri Artırma Uygulanıyor (Modül 4)...")
    augmentor = TimeSeriesAugmentor(random_state=42)
    
    # Kalp atımı segmentasyon (örnekleme)
    beat_segments = []
    labels = []
    
    for rp in r_peaks[1:-1]: # ilk ve son tepeyi atla sınırlar için
        if rp - int(fs*0.3) > 0 and rp + int(fs*0.5) < len(cleaned_signal):
            seg = cleaned_signal[rp - int(fs*0.3) : rp + int(fs*0.5)]
            beat_segments.append(seg)
            labels.append(0) # Hepsi N (0) sınıfında
            
    beat_segments = np.array(beat_segments)
    labels = np.array(labels)
    
    if len(beat_segments) > 0:
        aug_seg_warp = augmentor.temporal_warp(beat_segments[0])
        aug_seg_drop = augmentor.sensor_dropout(beat_segments[0])
        aug_seg_noise = augmentor.gaussian_noise(beat_segments[0])
        print(f"Orijinal atım uzunluğu: {len(beat_segments[0])}")
        print(f"Çarpıtılmış (Warped) atım uzunluğu: {len(aug_seg_warp)}")
        
        # SMOTE Test (Mock data)
        print("\n4. SMOTE Testi (Sentetik Tablo Verisi ile)...")
        # Sentetik X_features: 50 normal (0), 5 LBBB (1)
        X_mock = np.random.randn(55, 10)
        y_mock = np.array([0]*50 + [1]*5)
        balancer = BorderlineSMOTEBalancer()
        try:
            X_bal, y_bal = balancer.balance(X_mock, y_mock)
            print(f"Orijinal Sınıf Dağılımı: 0: 50, 1: 5")
            print(f"SMOTE Sonrası Dağılım : 0: {np.sum(y_bal==0)}, 1: {np.sum(y_bal==1)}")
        except Exception as e:
            print("SMOTE hatası:", e)

        # 5. Görselleştirme
        print("\n5. Görselleştirme Kaydediliyor...")
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
        
        # A. Orijinal sinyal ve R-tepeleri
        t_sec = np.arange(len(cleaned_signal)) / fs
        axes[0].plot(t_sec, cleaned_signal, label='Temizlenmiş EKG')
        axes[0].scatter(r_peaks/fs, cleaned_signal[r_peaks], color='red', label='R-Tepeleri', zorder=5)
        axes[0].set_title('Pan-Tompkins R-Tepe Tespiti', fontweight="bold")
        axes[0].set_ylabel('Genlik / Z-Score')
        axes[0].legend()
        
        # B. HRV dağılımı (Zaman domen)
        axes[1].plot(rr_intervals, marker='o', color='green')
        axes[1].set_title(f'R-R Aralıkları (Ortalama HR: {time_domain.get("mean_hr_bpm", 0):.1f} BPM)', fontweight="bold")
        axes[1].set_ylabel('R-R Süresi (ms)')
        axes[1].set_xlabel('Atım İndeksi')
        
        # C. Tek bir kalp atımı segmenti ve Noise Augmentation
        t_beat = np.arange(len(beat_segments[0])) / fs
        axes[2].plot(t_beat, beat_segments[0], label='Orijinal Atım', color='black', linewidth=2)
        axes[2].plot(t_beat, aug_seg_noise, alpha=0.7, label='Gauss Gürültüsü (Noise)', color='red')
        axes[2].plot(t_beat, aug_seg_drop, alpha=0.7, label='Sensör Temassızlığı (Dropout)', color='purple', linestyle='--')
        axes[2].set_title('Data Augmentation: Noise & Dropout', fontweight="bold")
        axes[2].legend()
        
        # D. Temporal Warping
        t_warp = np.arange(len(aug_seg_warp)) / fs
        axes[3].plot(t_beat, beat_segments[0], label='Orijinal Atım', color='black', linewidth=2)
        axes[3].plot(t_warp, aug_seg_warp, alpha=0.7, label='Temporal Warping (Hız/Yavaşlama)', color='orange')
        axes[3].set_title('Data Augmentation: Temporal Warping', fontweight="bold")
        axes[3].set_xlabel('Zaman (s)')
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig('module3_4_pipeline_output.png', dpi=150)
        print("Kayıt tamamlandı: module3_4_pipeline_output.png")
    else:
        print("Yeterli kalp atımı bulunamadı!")

if __name__ == '__main__':
    main()
