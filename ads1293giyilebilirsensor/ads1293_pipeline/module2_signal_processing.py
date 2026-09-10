# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 2: Giyilebilir Sistemlerde Ham Veri Temizleme ve Gürültü İzolasyonu
=============================================================================
Bu modül, ADS1293'ten gelen ölçeklenmiş EKG sinyalinden gürültüyü izole eder:

  1. FiltFilt (Sıfır-Fazlı) Butterworth Bandpass Filtre
     → Baseline Wander (0.1-0.5 Hz kayma) temizleme
     → Forward-Backward geçişle faz kayması = 0
     → EKG morfolojisi %100 korunur

  2. Dalgacık Dönüşümü (DWT) Tabanlı Gürültü İzolasyonu
     → Daubechies 4 (db4) ana dalgacığı (EKG altın standardı)
     → 5 seviye ayrıştırma (decomposition)
     → VisuShrink evrensel eşik (Donoho & Johnstone)
     → Soft Thresholding (yumuşak eşikleme, süreksizlik yok)
     → IDWT ile geri sentez

  3. Z-Score Standardizasyon
     → Min-Max yerine Z-Score: outlier'lara dayanıklı
     → Derin sinir ağlarında patlayan gradyan problemi önlenir
     → Model genliğe değil, sinyalin şekline odaklanır

Referans: Donoho, D. L. & Johnstone, I. M. (1994). "Ideal spatial
adaptation by wavelet shrinkage." Biometrika, 81(3), 425-455.
=============================================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt
import pywt
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
#  1) FiltFilt: Sıfır-Fazlı (Zero-Phase) Butterworth Bandpass Filtre
# ---------------------------------------------------------------------------
class ZeroPhaseFilter:
    """
    Sıfır-fazlı (Forward-Backward) Butterworth bandpass filtre.

    Baseline Wander (yürüme + nefes artefaktı, 0.1-0.5 Hz) ve
    yüksek frekanslı gürültüyü (EMG, powerline) temizler.

    Forward-Backward geçiş sayesinde:
      - Faz kayması NET olarak sıfırlanır
      - QRS morfolojisi ve dalga konumları %100 korunur
      - QT süresi bozulmaz

    Parameters
    ----------
    lowcut : float
        Alt kesim frekansı (Hz). Varsayılan: 0.5 Hz
        (baseline wander'ı kaldırır).
    highcut : float
        Üst kesim frekansı (Hz). Varsayılan: 40.0 Hz
        (EMG ve yüksek frekanslı gürültüyü kaldırır).
    fs : float
        Örnekleme frekansı (Hz).
    order : int
        Butterworth filtre derecesi. Varsayılan: 4
        (keskin geçiş bandı, düşük ripple).
    """

    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        fs: float = 360.0,
        order: int = 4,
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # Butterworth katsayıları (normalize frekanslar)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype="band")

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Sıfır-fazlı filtreleme uygular.

        Sinyal tasarlanan süzgeçten önce ileri yönlü (forward),
        ardından ters çevrilerek geri yönlü (backward) geçirilir.
        Bu matematiksel işlem sayesinde faz kayması net olarak sıfırlanır.

        Parameters
        ----------
        signal : np.ndarray
            Ham veya ölçeklenmiş EKG sinyali (1D, μV veya mV).

        Returns
        -------
        np.ndarray
            Sıfır-fazlı filtrelenmiş sinyal. Zamansal konumlar korunur.
        """
        # Minimum sinyal uzunluğu kontrolü
        # filtfilt, padlen = 3 * max(len(a), len(b)) gerektirir
        min_length = 3 * max(len(self.a), len(self.b))
        if len(signal) < min_length:
            raise ValueError(
                f"Sinyal uzunluğu ({len(signal)}) filtfilt için çok kısa. "
                f"Minimum {min_length} örnek gerekli."
            )

        return filtfilt(self.b, self.a, signal)

    def get_filter_info(self) -> Dict:
        """Filtre parametrelerini döndürür."""
        return {
            "type": "Butterworth Bandpass (Zero-Phase / FiltFilt)",
            "lowcut_hz": self.lowcut,
            "highcut_hz": self.highcut,
            "order": self.order,
            "effective_order": self.order * 2,  # forward + backward
            "fs_hz": self.fs,
            "phase_distortion": "0 (zero-phase)",
        }

    def __repr__(self) -> str:
        return (
            f"ZeroPhaseFilter(lowcut={self.lowcut}Hz, highcut={self.highcut}Hz, "
            f"order={self.order}, fs={self.fs}Hz)"
        )


# ---------------------------------------------------------------------------
#  2) Dalgacık Dönüşümü (DWT) Tabanlı Gürültü İzolasyonu
# ---------------------------------------------------------------------------
class WaveletDenoiser:
    """
    Ayrık Dalgacık Dönüşümü (DWT) tabanlı çoklu çözünürlük analiz mimarisi
    ile EMG gürültüsü izolasyonu.

    İşlem Adımları:
    1. Decomposition: db4 ana dalgacığı ile 5 seviye ayrıştırma
    2. VisuShrink Universal Threshold hesabı
    3. Soft Thresholding: Süreksizliksiz gürültü baskılama
    4. Reconstruction: IDWT ile geri sentez

    Neden db4?
    ----------
    Daubechies 4, EKG dalga formuna (QRS kompleksi) benzerliği nedeniyle
    literatürde altın standart kabul edilir. Asimetrik yapısı, P-QRS-T
    morfolojisini optimum şekilde yakalar.

    Neden Soft Thresholding?
    -------------------------
    Hard thresholding, matematiksel süreksizliklere ve keskin sinyal
    kırılmalarına yol açar. Soft thresholding, geçiş yumuşaklığını
    koruyarak sinyalde süreksizlik yaratmaz.

    Parameters
    ----------
    wavelet : str
        Ana dalgacık türü. Varsayılan: 'db4' (Daubechies 4).
    level : int
        Ayrıştırma seviyesi. Varsayılan: 5.
    threshold_mode : str
        Eşikleme modu: 'soft' (yumuşak) veya 'hard' (sert).
        Varsayılan: 'soft'.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 5,
        threshold_mode: str = "soft",
    ):
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode

        # Dalgacık nesnesini doğrula
        if wavelet not in pywt.wavelist():
            raise ValueError(
                f"Geçersiz dalgacık: '{wavelet}'. "
                f"Kullanılabilir dalgacıklar: {pywt.wavelist()[:10]}..."
            )

    def _compute_universal_threshold(self, detail_coeffs: np.ndarray) -> float:
        """
        VisuShrink (Donoho & Johnstone) evrensel eşik değerini hesaplar.

        Formül:
            σ = MAD(d1) / 0.6745
            threshold = σ × √(2 × ln(N))

        Burada:
            - MAD: Median Absolute Deviation (en yüksek frekanslı detay katmanı)
            - N: Sinyal uzunluğu
            - 0.6745: Gaussian dağılım düzeltme faktörü

        Parameters
        ----------
        detail_coeffs : np.ndarray
            En yüksek frekanslı detay katsayıları (seviye 1 detayı).

        Returns
        -------
        float
            Hesaplanan evrensel eşik değeri.
        """
        # Anlık gürültü standart sapması (robust tahmin)
        # MAD / 0.6745 → Gaussian σ tahmini
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745

        # Evrensel eşik: σ × √(2 × ln(N))
        n = len(detail_coeffs)
        threshold = sigma * np.sqrt(2.0 * np.log(n))

        return threshold

    def _soft_threshold(
        self, coeffs: np.ndarray, threshold: float
    ) -> np.ndarray:
        """
        Yumuşak Eşikleme (Soft Thresholding) uygular.

        Matematiksel kural:
            y(x) = sign(x) × max(|x| - threshold, 0)

        - Eşik altındaki gürültü bileşenleri → tamamen sıfırlanır
        - Eşik üstündeki gerçek tepe katsayıları → süreksizlik yaratmayacak
          şekilde daraltılarak korunur

        Parameters
        ----------
        coeffs : np.ndarray
            Dalgacık katsayıları.
        threshold : float
            Eşik değeri.

        Returns
        -------
        np.ndarray
            Eşiklenmiş katsayılar.
        """
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        DWT tabanlı gürültü izolasyonu tam pipeline.

        Adımlar:
        1. db4 ile 5 seviye ayrıştırma
        2. VisuShrink evrensel eşik hesabı (en yüksek frekanslı detay katmanından)
        3. Soft Thresholding ile detay katsayılarının gürültüden arındırılması
        4. IDWT ile geri sentez

        Parameters
        ----------
        signal : np.ndarray
            FiltFilt çıkışından gelen bandpass filtrelenmiş 1D EKG sinyali.

        Returns
        -------
        np.ndarray
            EMG gürültüsü izole edilmiş, QRS morfolojisi korunmuş sinyal.
        """
        # ── Adım 1: Çoklu Çözünürlük Ayrıştırması (Decomposition) ──
        # wavedec → [cA5, cD5, cD4, cD3, cD2, cD1]
        # cA5: 5. seviye yaklaşım (approximation) katsayıları
        # cD1-cD5: Detay katsayıları (yüksek → düşük frekans)
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # ── Adım 2: Dinamik Evrensel Eşik Hesabı ──
        # En yüksek frekanslı detay katmanı (cD1) üzerinden σ tahmini
        detail_level1 = coeffs[-1]  # cD1: EMG gürültüsü yoğunlaşır
        threshold = self._compute_universal_threshold(detail_level1)

        # ── Adım 3: Yumuşak Eşikleme (Soft Thresholding) ──
        # Yaklaşım katsayısı (cA5) dokunulmaz → genel şekil korunur
        # Detay katsayıları (cD1-cD5) eşiklenir
        denoised_coeffs = [coeffs[0]]  # cA5 orijinal kalır

        for i in range(1, len(coeffs)):
            if self.threshold_mode == "soft":
                denoised_coeffs.append(
                    self._soft_threshold(coeffs[i], threshold)
                )
            else:
                # Hard thresholding (raporda tercih edilmese de destek)
                thresholded = coeffs[i].copy()
                thresholded[np.abs(thresholded) < threshold] = 0.0
                denoised_coeffs.append(thresholded)

        # ── Adım 4: Geri İnşa (Reconstruction) ──
        # IDWT: Ters Ayrık Dalgacık Dönüşümü ile zaman düzlemine geri sentez
        reconstructed = pywt.waverec(denoised_coeffs, self.wavelet)

        # waverec, padding nedeniyle 1 örnek fazla üretebilir → düzelt
        reconstructed = reconstructed[: len(signal)]

        return reconstructed

    def get_decomposition_info(
        self, signal: np.ndarray
    ) -> Dict:
        """
        Ayrıştırma seviyelerinin frekans bantlarını ve katsayı sayılarını döndürür.
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        info = {
            "wavelet": self.wavelet,
            "decomposition_level": self.level,
            "threshold_mode": self.threshold_mode,
            "levels": {},
        }

        fs = 360.0  # varsayılan; gerçek değer dışarıdan verilebilir
        for i, c in enumerate(coeffs):
            if i == 0:
                name = f"cA{self.level} (Approximation)"
                freq_band = f"0-{fs / (2 ** (self.level + 1)):.1f} Hz"
            else:
                level_num = self.level - i + 1
                name = f"cD{level_num} (Detail)"
                low = fs / (2 ** (level_num + 1))
                high = fs / (2**level_num)
                freq_band = f"{low:.1f}-{high:.1f} Hz"

            info["levels"][name] = {
                "n_coefficients": len(c),
                "frequency_band": freq_band,
                "energy": float(np.sum(c**2)),
            }

        return info

    def __repr__(self) -> str:
        return (
            f"WaveletDenoiser(wavelet='{self.wavelet}', "
            f"level={self.level}, mode='{self.threshold_mode}')"
        )


# ---------------------------------------------------------------------------
#  3) Z-Score Standardizasyon
# ---------------------------------------------------------------------------
class ZScoreNormalizer:
    """
    Z-Score Standardizasyon: (X - μ) / σ

    Neden Min-Max [0, 1] DEĞİL?
    ----------------------------
    Min-Max, elektrotun cilde bir anlık sürtünmesinde (Outlier) oluşan
    spike, tüm sağlıklı veriyi sıfıra sıkıştırır.

    Z-Score avantajları:
      - Outlier'lara dayanıklı (robust) normalizasyon
      - Modelin genliğe değil, sinyalin ŞEKLİNE odaklanmasını mecbur kılar
      - Derin sinir ağlarında patlayan gradyan problemini önler
      - Dağılım parametreleri (μ, σ) eğitim setinden kaydedilir,
        test/inference'ta aynı dönüşüm uygulanır (data leakage önleme)

    Parameters
    ----------
    epsilon : float
        Sıfıra bölme hatası önleme sabiti. Varsayılan: 1e-8.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self._is_fitted: bool = False

    def fit(self, signal: np.ndarray) -> "ZScoreNormalizer":
        """
        Eğitim verisinden μ ve σ parametrelerini öğrenir.

        Parameters
        ----------
        signal : np.ndarray
            Eğitim EKG sinyali (1D).

        Returns
        -------
        self
        """
        self.mean_ = float(np.mean(signal))
        self.std_ = float(np.std(signal))
        self._is_fitted = True
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Öğrenilmiş parametrelerle Z-Score dönüşümü uygular.

        Formül: Z = (X - μ) / (σ + ε)

        Parameters
        ----------
        signal : np.ndarray
            Dönüştürülecek EKG sinyali.

        Returns
        -------
        np.ndarray
            Z-Score normalize edilmiş sinyal.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ZScoreNormalizer henüz eğitilmedi. Önce fit() çağırın."
            )

        return (signal - self.mean_) / (self.std_ + self.epsilon)

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """fit() + transform() tek adımda."""
        return self.fit(signal).transform(signal)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Z-Score'u orijinal ölçeğe geri dönüştürür."""
        if not self._is_fitted:
            raise RuntimeError(
                "ZScoreNormalizer henüz eğitilmedi. Önce fit() çağırın."
            )
        return normalized * (self.std_ + self.epsilon) + self.mean_

    def get_params(self) -> Dict:
        """Normalizasyon parametrelerini döndürür."""
        return {
            "mean": self.mean_,
            "std": self.std_,
            "epsilon": self.epsilon,
            "is_fitted": self._is_fitted,
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"ZScoreNormalizer(μ={self.mean_:.4f}, σ={self.std_:.4f}, "
                f"ε={self.epsilon})"
            )
        return f"ZScoreNormalizer(not fitted, ε={self.epsilon})"


# ---------------------------------------------------------------------------
#  4) Modül 2 Entegre Pipeline
# ---------------------------------------------------------------------------
class SignalCleaningPipeline:
    """
    Modül 2 ana orkestratör sınıfı.
    FiltFilt → Wavelet Denoising → Z-Score normalizasyon.

    Kullanım
    --------
    >>> from ads1293_pipeline.module1_data_acquisition import RawDataPipeline
    >>> raw_pipeline = RawDataPipeline(record_id='100', duration_seconds=30)
    >>> raw_result = raw_pipeline.run()
    >>>
    >>> clean_pipeline = SignalCleaningPipeline(fs=raw_result['fs'])
    >>> clean_result = clean_pipeline.run(raw_result['ecg_uv'])
    >>> print(clean_result['ecg_cleaned'].shape)
    >>> print(clean_result['ecg_normalized'].shape)
    """

    def __init__(
        self,
        fs: float = 360.0,
        # FiltFilt parametreleri
        lowcut: float = 0.5,
        highcut: float = 40.0,
        butter_order: int = 4,
        # Wavelet parametreleri
        wavelet: str = "db4",
        wavelet_level: int = 5,
        threshold_mode: str = "soft",
        # Z-Score parametreleri
        zscore_epsilon: float = 1e-8,
    ):
        self.fs = fs

        # Alt modülleri oluştur
        self.bandpass_filter = ZeroPhaseFilter(
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
            order=butter_order,
        )

        self.wavelet_denoiser = WaveletDenoiser(
            wavelet=wavelet,
            level=wavelet_level,
            threshold_mode=threshold_mode,
        )

        self.normalizer = ZScoreNormalizer(epsilon=zscore_epsilon)

    def run(
        self,
        ecg_uv: np.ndarray,
        normalize: bool = True,
    ) -> Dict:
        """
        Modül 2 tam pipeline'ını çalıştırır.

        Sıralı işlem:
        1. FiltFilt → Baseline Wander ve yüksek frekans gürültüsü temizliği
        2. DWT db4 → EMG / kas gürültüsü izolasyonu
        3. Z-Score → Gradyan stabilizasyonu için normalizasyon

        Parameters
        ----------
        ecg_uv : np.ndarray
            Modül 1 çıkışından gelen μV cinsinden EKG sinyali.
        normalize : bool
            Z-Score normalizasyonu uygulansın mı? Varsayılan: True.

        Returns
        -------
        dict
            'ecg_bandpass'   : FiltFilt sonrası sinyal (μV)
            'ecg_cleaned'    : Wavelet denoising sonrası sinyal (μV)
            'ecg_normalized' : Z-Score normalize sinyal (boyutsuz)
            'zscore_params'  : Normalizasyon parametreleri (μ, σ)
            'wavelet_info'   : Dalgacık ayrıştırma detayları
            'filter_info'    : Filtre parametreleri
        """
        # ── Adım 1: FiltFilt (Sıfır-Fazlı Bandpass Filtre) ──
        ecg_bandpass = self.bandpass_filter.apply(ecg_uv)

        # ── Adım 2: DWT Wavelet Denoising (db4, 5 seviye, Soft Threshold) ──
        ecg_cleaned = self.wavelet_denoiser.apply(ecg_bandpass)

        # ── Adım 3: Z-Score Standardizasyon ──
        ecg_normalized = None
        zscore_params = None

        if normalize:
            ecg_normalized = self.normalizer.fit_transform(ecg_cleaned)
            zscore_params = self.normalizer.get_params()

        # ── Bilgi toplama ──
        wavelet_info = self.wavelet_denoiser.get_decomposition_info(ecg_bandpass)
        filter_info = self.bandpass_filter.get_filter_info()

        return {
            "ecg_bandpass": ecg_bandpass,
            "ecg_cleaned": ecg_cleaned,
            "ecg_normalized": ecg_normalized,
            "zscore_params": zscore_params,
            "wavelet_info": wavelet_info,
            "filter_info": filter_info,
        }

    def __repr__(self) -> str:
        return (
            f"SignalCleaningPipeline(\n"
            f"  {self.bandpass_filter},\n"
            f"  {self.wavelet_denoiser},\n"
            f"  {self.normalizer}\n"
            f")"
        )
