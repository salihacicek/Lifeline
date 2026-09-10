# -*- coding: utf-8 -*-
"""
=============================================================================
ADS1293 Giyilebilir Telemetri Sistemi — Modül 1 & 2 Test Script'i
=============================================================================
Bu script, Modül 1 (Veri Çekme + Ölçekleme) ve Modül 2 (Gürültü İzolasyonu)
pipeline'larını uçtan uca çalıştırır ve sonuçları görselleştirir.

Çalıştırma:
    python test_module1_2.py
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan kaydetme modu
import matplotlib.pyplot as plt

from ads1293_pipeline.module1_data_acquisition import RawDataPipeline
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline


def main():
    print("=" * 70)
    print("ADS1293 Giyilebilir Telemetri Sistemi")
    print("MODÜL 1 & 2 — Uçtan Uca Test")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    #  MODÜL 1: Ham Veri Akışı Pipeline
    # ══════════════════════════════════════════════════════════════════════
    print("\n[MODÜL 1] PhysioNet MIT-BIH kaydı çekiliyor...")
    print("  → Kayıt: 100, Kanal: 0, Süre: 30 saniye")
    print("  → ADS1293 simülasyonu: 24-bit ADC formatına dönüştürme")
    print("  → Ring Buffer kapasitesi: 4096 örnek")
    print("  → Paket kaybı simülasyonu: Aktif (%0.5)")

    raw_pipeline = RawDataPipeline(
        record_id="100",
        channel=0,
        duration_seconds=30.0,
        ring_buffer_capacity=4096,
        v_ref=2.4,
        gain=3.5,
        simulate_packet_loss=True,
        packet_loss_rate=0.005,
    )

    raw_result = raw_pipeline.run()

    # İstatistikleri yazdır
    stats = raw_result["pipeline_stats"]
    print(f"\n  ✓ Toplam ham örnek      : {stats['total_samples_raw']}")
    print(f"  ✓ Paket kaybı sonrası   : {stats['samples_after_loss']}")
    print(f"  ✓ İnterpolasyon sonrası : {stats['samples_after_interpolation']}")
    print(f"  ✓ Tespit edilen boşluk  : {stats['detected_gaps']}")
    print(f"  ✓ Satürasyon bölgeleri  : {stats['saturation_regions']}")
    print(f"  ✓ Buffer taşması        : {stats['ring_buffer_overflows']}")
    print(f"  ✓ V_LSB                 : {stats['v_lsb_uv']:.6f} μV/LSB")
    print(f"  ✓ ADC Scaler            : {stats['adc_scaler']}")
    print(f"  ✓ Örnekleme frekansı    : {raw_result['fs']} Hz")

    ecg_uv = raw_result["ecg_uv"]
    timestamps = raw_result["timestamps"]
    fs = raw_result["fs"]

    # ══════════════════════════════════════════════════════════════════════
    #  MODÜL 2: Gürültü İzolasyonu Pipeline
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[MODÜL 2] Gürültü izolasyonu başlatılıyor...")
    print("  → FiltFilt: Butterworth Bandpass [0.5 - 40.0] Hz, Derece: 4")
    print("  → Wavelet: Daubechies 4 (db4), 5 seviye, Soft Thresholding")
    print("  → Z-Score: (X - μ) / σ standardizasyon")

    clean_pipeline = SignalCleaningPipeline(
        fs=fs,
        lowcut=0.5,
        highcut=40.0,
        butter_order=4,
        wavelet="db4",
        wavelet_level=5,
        threshold_mode="soft",
    )

    clean_result = clean_pipeline.run(ecg_uv, normalize=True)

    # Bilgileri yazdır
    filt_info = clean_result["filter_info"]
    print(f"\n  ✓ Filtre tipi    : {filt_info['type']}")
    print(f"  ✓ Faz bozulumu   : {filt_info['phase_distortion']}")
    print(f"  ✓ Efektif derece : {filt_info['effective_order']}")

    z_params = clean_result["zscore_params"]
    print(f"\n  ✓ Z-Score μ      : {z_params['mean']:.4f} μV")
    print(f"  ✓ Z-Score σ      : {z_params['std']:.4f} μV")

    wv_info = clean_result["wavelet_info"]
    print(f"\n  ✓ Dalgacık       : {wv_info['wavelet']}")
    print(f"  ✓ Seviye         : {wv_info['decomposition_level']}")
    print(f"  ✓ Eşikleme       : {wv_info['threshold_mode']}")
    print("  ✓ Frekans bantları:")
    for level_name, level_data in wv_info["levels"].items():
        print(
            f"    - {level_name}: {level_data['frequency_band']}, "
            f"{level_data['n_coefficients']} katsayı, "
            f"Enerji={level_data['energy']:.2f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  GÖRSELLEŞTİRME
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[GÖRSELLEŞTİRME] Grafik oluşturuluyor...")

    ecg_bandpass = clean_result["ecg_bandpass"]
    ecg_cleaned = clean_result["ecg_cleaned"]
    ecg_normalized = clean_result["ecg_normalized"]

    # Zaman eksenini kısalt (görselleştirme için ilk 5 saniye)
    display_seconds = 5.0
    display_samples = int(display_seconds * fs)
    t = timestamps[:display_samples]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        "ADS1293 Giyilebilir Telemetri — Modül 1 & 2 Pipeline Çıktıları\n"
        f"Kayıt: MIT-BIH 100 | fs={fs} Hz | İlk {display_seconds}s",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel 1: Ham veri (Modül 1 çıkışı, μV) ──
    axes[0].plot(t, ecg_uv[:display_samples], color="#e63946", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("Genlik (μV)", fontsize=10)
    axes[0].set_title(
        "① Ham EKG Sinyali (24-bit ADC → μV Ölçekleme Sonrası)",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(["Ham sinyal (paket kaybı + interpolasyon)"], loc="upper right", fontsize=8)

    # ── Panel 2: FiltFilt sonrası ──
    axes[1].plot(t, ecg_bandpass[:display_samples], color="#457b9d", linewidth=0.7)
    axes[1].set_ylabel("Genlik (μV)", fontsize=10)
    axes[1].set_title(
        "② FiltFilt Sıfır-Fazlı Bandpass [0.5-40 Hz] (Baseline Wander Temizliği)",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Wavelet denoising sonrası ──
    axes[2].plot(t, ecg_cleaned[:display_samples], color="#2a9d8f", linewidth=0.7)
    axes[2].set_ylabel("Genlik (μV)", fontsize=10)
    axes[2].set_title(
        "③ Dalgacık Gürültü İzolasyonu (db4, L=5, Soft Thresholding)",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[2].grid(True, alpha=0.3)

    # ── Panel 4: Z-Score normalizasyon sonrası ──
    axes[3].plot(t, ecg_normalized[:display_samples], color="#e9c46a", linewidth=0.7)
    axes[3].set_ylabel("Z-Score (σ)", fontsize=10)
    axes[3].set_xlabel("Zaman (saniye)", fontsize=11)
    axes[3].set_title(
        "④ Z-Score Standardizasyon (X - μ) / σ — Model Girişi",
        fontsize=11, fontweight="bold", loc="left",
    )
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = "module1_2_pipeline_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Grafik kaydedildi: {output_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  SNR (Sinyal-Gürültü Oranı) Karşılaştırması
    # ══════════════════════════════════════════════════════════════════════
    noise_raw = ecg_uv - ecg_cleaned
    snr_before = 10 * np.log10(
        np.sum(ecg_cleaned**2) / (np.sum(noise_raw**2) + 1e-10)
    )
    print(f"\n  ✓ Tahmini SNR iyileşme: {snr_before:.2f} dB")

    print("\n" + "=" * 70)
    print("✅ Modül 1 & 2 başarıyla tamamlandı.")
    print("   Sonraki adım: Modül 3 (Morfolojik Analiz & Pan-Tompkins)")
    print("   Onayınızı bekliyorum.")
    print("=" * 70)


if __name__ == "__main__":
    main()
