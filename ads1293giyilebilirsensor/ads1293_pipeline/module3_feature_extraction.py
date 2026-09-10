# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 3: Morfolojik Analiz ve Özellik Çıkarımı (Feature Engineering)
=============================================================================
Bu modül, temizlenmiş EKG sinyalinden klinik özellik vektörü üretir:

  1. Pan-Tompkins Algoritması ile R-Tepe Noktası Tespiti
     → Bandpass (5-15 Hz) → Türev → Kare alma → MWI → Adaptif eşik
     → Search-back ile kaçırılan tepe kurtarma
     → RR aralık doğrulama

  2. Kalp Hızı Değişkenliği (HRV) Analizi
     → Zaman Düzlemi: SDNN, RMSSD, pNN50, Ortalama HR (BPM)
     → Frekans Düzlemi: VLF, LF, HF güç, LF/HF oranı (Welch PSD)

  3. Örnek Entropisi (Sample Entropy, SampEn)
     → Kardiyak düzensizliğin bilgi-kuramsal ölçütü
     → Düşük entropi = daha düzenli = potansiyel kalp yetmezliği riski

  4. T-Dalga Alternansı (TWA)
     → Atım-atım T-dalgası genlik farklılıkları
     → Ani kardiyak ölüm (SCD) risk göstergesi

  5. Özellik Çıkarım Pipeline (Orkestratör)
     → Tüm alt modülleri koordine eder
     → Bireysel kalp atımı segmentasyonu (CNN girişi için)
     → R-peak ± pencere ile segment çıkarma

Referans:
  Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm."
  IEEE Trans. Biomed. Eng., BME-32(3), 230-236.
=============================================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from typing import Optional, Dict, List, Tuple
import warnings


# ---------------------------------------------------------------------------
#  1) Pan-Tompkins R-Tepe Tespiti (QRS Detektörü)
# ---------------------------------------------------------------------------
class PanTompkinsDetector:
    """
    Pan-Tompkins QRS Tespit Algoritması — Tam Uygulama.

    Beş aşamalı sinyal işleme zinciri ve adaptif eşikleme ile
    gerçek zamanlı R-tepe noktası tespiti yapar.

    İşlem Adımları
    --------------
    1. Bandpass Filtre (5–15 Hz)
       → QRS kompleksinin enerji yoğunluğu bu bantta maksimum
       → P/T dalgaları ve EMG gürültüsü bastırılır

    2. Diferansiyel (Türev)
       → QRS'in dik yamaçları vurgulanır
       → 5 noktalı türev filtresi: [-1, -2, 0, 2, 1] / 8T

    3. Kare Alma (Squaring)
       → Tüm değerler pozitif yapılır
       → Büyük QRS tepeleri küçük gürültüye karşı vurgulanır

    4. Hareketli Pencere Entegrasyonu (MWI)
       → QRS süresine uygun pencere (~150ms)
       → Dalga zarfı (envelope) çıkarılır

    5. Adaptif Çift Eşikleme
       → Sinyal tepesi (SPKI) ve gürültü tepesi (NPKI) ayrı izlenir
       → THRESHOLD_I1 = NPKI + 0.25 × (SPKI − NPKI)
       → THRESHOLD_I2 = 0.5 × THRESHOLD_I1 (search-back için)
       → RR aralık doğrulama ile fizyolojik olmayan tepeler elenir

    Parameters
    ----------
    fs : float
        Örnekleme frekansı (Hz). Varsayılan: 360.0
    """

    def __init__(self, fs: float = 360.0):
        self.fs = fs

    # ── Adım 1: Bandpass Filtre (5–15 Hz) ──
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        5–15 Hz bandpass filtre uygular (Butterworth, 2. derece).

        Bu bant, QRS kompleksinin enerji yoğunluğunun maksimum olduğu
        bölgedir. P ve T dalgaları ile yüksek frekanslı EMG gürültüsü
        bu filtre tarafından bastırılır.

        Parameters
        ----------
        signal : np.ndarray
            Giriş EKG sinyali.

        Returns
        -------
        np.ndarray
            5–15 Hz bandpass filtrelenmiş sinyal.
        """
        nyq = 0.5 * self.fs
        low = 5.0 / nyq
        high = 15.0 / nyq
        b, a = butter(2, [low, high], btype="band")
        return filtfilt(b, a, signal)

    # ── Adım 2: Diferansiyel (5 Noktalı Türev) ──
    def _derivative(self, signal: np.ndarray) -> np.ndarray:
        """
        5 noktalı türev filtresi uygular.

        Pan-Tompkins türev formülü:
            y[n] = (1/8T) × (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])

        QRS kompleksinin dik kenarları (pozitif ve negatif yamaçlar)
        bu türev ile vurgulanır.

        Parameters
        ----------
        signal : np.ndarray
            Bandpass filtrelenmiş sinyal.

        Returns
        -------
        np.ndarray
            Türev uygulanmış sinyal (aynı uzunlukta).
        """
        # 5 noktalı türev katsayıları: [-1, -2, 0, 2, 1] / (8 × T)
        # T = 1/fs → katsayı: fs/8
        derivative = np.zeros_like(signal)
        for i in range(2, len(signal) - 2):
            derivative[i] = (
                (-signal[i - 2] - 2 * signal[i - 1]
                 + 2 * signal[i + 1] + signal[i + 2])
            ) * (self.fs / 8.0)
        return derivative

    # ── Adım 3: Kare Alma ──
    def _squaring(self, signal: np.ndarray) -> np.ndarray:
        """
        Sinyal kareleme: y[n] = x[n]²

        Tüm değerler pozitif yapılır ve QRS kompleksinin büyük
        genlikli bileşenleri küçük gürültüye karşı üstel olarak
        vurgulanır.

        Parameters
        ----------
        signal : np.ndarray
            Türev uygulanmış sinyal.

        Returns
        -------
        np.ndarray
            Kareleme sonrası sinyal.
        """
        return signal ** 2

    # ── Adım 4: Hareketli Pencere Entegrasyonu (MWI) ──
    def _moving_window_integration(self, signal: np.ndarray) -> np.ndarray:
        """
        Hareketli Pencere Entegrasyonu (Moving Window Integration).

        Pencere genişliği: ~150 ms (QRS kompleksi süresi)
            window_size = round(0.150 × fs)

        Bu entegrasyon, kareleme sonrası sinyalin zarfını (envelope)
        çıkararak QRS bölgesini tek bir tepe olarak ortaya koyar.

        Parameters
        ----------
        signal : np.ndarray
            Kareleme sonrası sinyal.

        Returns
        -------
        np.ndarray
            MWI çıkışı — QRS zarfı.
        """
        window_size = int(round(0.150 * self.fs))
        if window_size < 1:
            window_size = 1

        # Kümülatif toplam ile O(N) hareketli ortalama
        cumsum = np.cumsum(np.insert(signal, 0, 0))
        mwi = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        # Başlangıç padding: uzunluğu eşitle
        pad_length = len(signal) - len(mwi)
        mwi = np.concatenate([np.zeros(pad_length), mwi])

        return mwi

    # ── Adım 5: Adaptif Çift Eşikleme + Search-Back ──
    def detect(self, signal: np.ndarray, fs: Optional[float] = None) -> np.ndarray:
        """
        Pan-Tompkins algoritmasının tam uygulaması ile R-tepe noktalarını tespit eder.

        Adaptif çift eşik mekanizması:
          - SPKI: Sinyal tepesi hareketli ortalaması
          - NPKI: Gürültü tepesi hareketli ortalaması
          - THRESHOLD_I1 = NPKI + 0.25 × (SPKI - NPKI) → Birincil eşik
          - THRESHOLD_I2 = 0.5 × THRESHOLD_I1 → Search-back eşiği

        RR aralık doğrulama:
          - Fizyolojik aralık: 200ms < RR < 2000ms (30–300 BPM)
          - Bu aralık dışındaki tepeler reddedilir

        Search-Back:
          - RR aralığı beklenen değerin %166'sını aştığında
          - Kaçırılmış tepe, düşük eşik (THRESHOLD_I2) ile aranır

        Parameters
        ----------
        signal : np.ndarray
            Temizlenmiş EKG sinyali (Modül 2 çıkışı).
        fs : float, optional
            Örnekleme frekansı. None ise __init__'teki değer kullanılır.

        Returns
        -------
        np.ndarray
            R-tepe noktası indeksleri (sample index).
        """
        if fs is not None:
            self.fs = fs

        # ── Sinyal işleme zinciri ──
        filtered = self._bandpass_filter(signal)
        differentiated = self._derivative(filtered)
        squared = self._squaring(differentiated)
        integrated = self._moving_window_integration(squared)

        # ── Tepe tespiti parametreleri ──
        # Refrakter periyot: QRS süresi ≈ 200ms → bu süre içinde
        # yeni tepe aranmaz (double-detection önleme)
        refractory_period = int(round(0.200 * self.fs))

        # Fizyolojik RR sınırları (30–300 BPM arası)
        rr_min = int(round(0.200 * self.fs))   # 300 BPM → 200ms
        rr_max = int(round(2.000 * self.fs))    # 30 BPM → 2000ms

        # ── Adaptif eşik başlangıç değerleri ──
        # İlk 2 saniyelik penceredeki maksimum ile başlat
        init_window = min(int(2.0 * self.fs), len(integrated))
        spki = np.max(integrated[:init_window]) * 0.25  # Sinyal tepesi tahmini
        npki = np.mean(integrated[:init_window]) * 0.50  # Gürültü tahmini
        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1

        # ── Tepe tarama ──
        r_peaks: List[int] = []
        peak_candidates: List[Tuple[int, float]] = []

        # Tüm yerel tepe noktalarını bul (basit yöntem)
        for i in range(1, len(integrated) - 1):
            if integrated[i] > integrated[i - 1] and integrated[i] >= integrated[i + 1]:
                peak_candidates.append((i, integrated[i]))

        # ── Adaptif sınıflandırma ve RR doğrulama ──
        rr_average = self.fs  # Başlangıç: 1 saniye (60 BPM)

        for idx, (peak_idx, peak_val) in enumerate(peak_candidates):
            # Refrakter periyot kontrolü
            if r_peaks and (peak_idx - r_peaks[-1]) < refractory_period:
                continue

            if peak_val > threshold_i1:
                # ── Sinyal tepesi olarak sınıfla ──
                # RR aralık doğrulama
                if r_peaks:
                    rr_interval = peak_idx - r_peaks[-1]
                    if rr_interval < rr_min or rr_interval > rr_max:
                        # Fizyolojik olmayan aralık → gürültü olarak işaretle
                        npki = 0.875 * npki + 0.125 * peak_val
                        threshold_i1 = npki + 0.25 * (spki - npki)
                        threshold_i2 = 0.5 * threshold_i1
                        continue

                r_peaks.append(peak_idx)

                # SPKI güncelle
                spki = 0.875 * spki + 0.125 * peak_val

                # RR ortalamasını güncelle (son 8 RR aralığı)
                if len(r_peaks) >= 2:
                    recent_rr = np.diff(r_peaks[-9:])
                    rr_average = np.mean(recent_rr)

            else:
                # ── Gürültü tepesi olarak sınıfla ──
                npki = 0.875 * npki + 0.125 * peak_val

            # Eşikleri güncelle
            threshold_i1 = npki + 0.25 * (spki - npki)
            threshold_i2 = 0.5 * threshold_i1

            # ══ SEARCH-BACK: Kaçırılmış tepe kurtarma ══
            # RR aralığı beklenenin %166'sını aştığında, düşük eşikle
            # geriye dönük arama yapılır.
            if len(r_peaks) >= 2:
                last_rr = r_peaks[-1] - r_peaks[-2]
                if last_rr > 1.66 * rr_average:
                    # Kaçırılmış aralıkta düşük eşikle arama
                    search_start = r_peaks[-2] + refractory_period
                    search_end = r_peaks[-1] - refractory_period

                    search_candidates = [
                        (p, v) for p, v in peak_candidates
                        if search_start < p < search_end and v > threshold_i2
                    ]

                    if search_candidates:
                        # En yüksek genlikli adayı seç
                        best = max(search_candidates, key=lambda x: x[1])
                        r_peaks.append(best[0])
                        r_peaks.sort()

                        # SPKI güncelle (search-back tepesi için farklı ağırlık)
                        spki = 0.75 * spki + 0.25 * best[1]

        # ── Orijinal sinyaldeki gerçek R-tepe konumlarına ince ayar ──
        # MWI çıkışındaki tepeyi, orijinal sinyaldeki yerel maksimuma eşle
        refined_peaks = []
        search_radius = int(round(0.075 * self.fs))  # ±75ms pencere

        for peak in r_peaks:
            start = max(0, peak - search_radius)
            end = min(len(signal), peak + search_radius + 1)
            local_max_idx = start + np.argmax(np.abs(signal[start:end]))
            refined_peaks.append(local_max_idx)

        return np.array(refined_peaks, dtype=np.int64)

    def get_rr_intervals(
        self,
        r_peaks: np.ndarray,
        fs: Optional[float] = None,
    ) -> np.ndarray:
        """
        R-tepe noktalarından RR aralıklarını hesaplar (milisaniye).

        Parameters
        ----------
        r_peaks : np.ndarray
            R-tepe indeksleri.
        fs : float, optional
            Örnekleme frekansı. None ise __init__'teki değer kullanılır.

        Returns
        -------
        np.ndarray
            RR aralıkları (ms cinsinden). Uzunluk = len(r_peaks) - 1.
        """
        if fs is None:
            fs = self.fs

        if len(r_peaks) < 2:
            warnings.warn(
                "RR aralığı hesaplamak için en az 2 R-tepe gerekli.",
                stacklevel=2,
            )
            return np.array([], dtype=np.float64)

        rr_samples = np.diff(r_peaks).astype(np.float64)
        rr_ms = (rr_samples / fs) * 1000.0  # örnek → ms dönüşümü
        return rr_ms

    def __repr__(self) -> str:
        return f"PanTompkinsDetector(fs={self.fs}Hz)"


# ---------------------------------------------------------------------------
#  2) Kalp Hızı Değişkenliği (HRV) Analizi
# ---------------------------------------------------------------------------
class HRVAnalyzer:
    """
    Heart Rate Variability (HRV) — Kalp Hızı Değişkenliği Analizi.

    Otonom sinir sistemi aktivitesinin (sempatik vs. parasempatik)
    dolaylı ölçümü için R-R aralıklarından klinik metrikler üretir.

    Zaman Düzlemi Metrikleri
    -------------------------
    - SDNN  : NN aralıklarının standart sapması → Genel HRV ölçütü
    - RMSSD : Ardışık farkların karekök ortalaması → Kısa süreli değişkenlik
    - pNN50 : Ardışık aralık farkı >50ms oranı → Parasempatik aktivite
    - Mean HR: Ortalama kalp hızı (BPM)

    Frekans Düzlemi Metrikleri (Welch PSD)
    ----------------------------------------
    - VLF  : Çok düşük frekans gücü (0.003–0.04 Hz) → Termoregülasyon
    - LF   : Düşük frekans gücü (0.04–0.15 Hz) → Sempatik + Parasempatik
    - HF   : Yüksek frekans gücü (0.15–0.4 Hz) → Parasempatik (vagal)
    - LF/HF: Otonom stres dengesi oranı
    - Total : Toplam güç

    Parameters
    ----------
    interpolation_fs : float
        RR tachogram'ın Welch PSD için yeniden örnekleme frekansı (Hz).
        Varsayılan: 4.0 Hz (Shannon kriteri: 2 × 0.4 Hz üst bandı = 0.8 Hz).
    """

    # Frekans bandı sınırları (Hz)
    VLF_BAND: Tuple[float, float] = (0.003, 0.04)
    LF_BAND: Tuple[float, float] = (0.04, 0.15)
    HF_BAND: Tuple[float, float] = (0.15, 0.4)

    def __init__(self, interpolation_fs: float = 4.0):
        self.interpolation_fs = interpolation_fs

    def compute_time_domain(self, rr_intervals_ms: np.ndarray) -> Dict:
        """
        RR aralıklarından zaman düzlemi HRV metriklerini hesaplar.

        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            RR aralıkları (milisaniye cinsinden).

        Returns
        -------
        dict
            'sdnn_ms'  : NN aralıklarının standart sapması (ms)
            'rmssd_ms' : Ardışık farkların karekök ortalaması (ms)
            'pnn50_%'  : >50ms ardışık fark oranı (%)
            'mean_hr_bpm' : Ortalama kalp hızı (BPM)
            'mean_rr_ms'  : Ortalama RR aralığı (ms)
            'min_hr_bpm'  : Minimum kalp hızı (BPM)
            'max_hr_bpm'  : Maksimum kalp hızı (BPM)
            'n_beats'     : Toplam atım sayısı
        """
        if len(rr_intervals_ms) < 2:
            warnings.warn(
                "Zaman düzlemi HRV için en az 2 RR aralığı gerekli.",
                stacklevel=2,
            )
            return {
                "sdnn_ms": np.nan, "rmssd_ms": np.nan,
                "pnn50_%": np.nan, "mean_hr_bpm": np.nan,
                "mean_rr_ms": np.nan, "min_hr_bpm": np.nan,
                "max_hr_bpm": np.nan, "n_beats": len(rr_intervals_ms),
            }

        rr = rr_intervals_ms.astype(np.float64)

        # SDNN: Tüm NN aralıklarının standart sapması
        sdnn = np.std(rr, ddof=1)

        # RMSSD: Ardışık NN farkları → karekök ortalama
        successive_diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))

        # pNN50: Ardışık fark > 50ms olan çiftlerin oranı (%)
        nn50_count = np.sum(np.abs(successive_diffs) > 50.0)
        pnn50 = (nn50_count / len(successive_diffs)) * 100.0

        # Ortalama kalp hızı (BPM)
        mean_rr = np.mean(rr)
        mean_hr = 60000.0 / mean_rr  # ms → BPM: 60000 / RR_ms

        # Min / Max HR
        min_hr = 60000.0 / np.max(rr)  # En uzun RR = en düşük HR
        max_hr = 60000.0 / np.min(rr)  # En kısa RR = en yüksek HR

        return {
            "sdnn_ms": float(sdnn),
            "rmssd_ms": float(rmssd),
            "pnn50_%": float(pnn50),
            "mean_hr_bpm": float(mean_hr),
            "mean_rr_ms": float(mean_rr),
            "min_hr_bpm": float(min_hr),
            "max_hr_bpm": float(max_hr),
            "n_beats": len(rr_intervals_ms) + 1,
        }

    def compute_frequency_domain(self, rr_intervals_ms: np.ndarray) -> Dict:
        """
        Welch Periodogram ile frekans düzlemi HRV metriklerini hesaplar.

        RR aralık serisi düzensiz aralıklıdır (her RR farklı süre).
        Welch PSD uygulamak için önce düzgün örneklemeli zaman serisine
        (tachogram) dönüştürülür (cubic interpolation).

        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            RR aralıkları (milisaniye cinsinden).

        Returns
        -------
        dict
            'vlf_power_ms2' : VLF band gücü (ms²)
            'lf_power_ms2'  : LF band gücü (ms²)
            'hf_power_ms2'  : HF band gücü (ms²)
            'lf_hf_ratio'   : LF/HF oranı (otonom denge)
            'total_power_ms2' : Toplam güç (ms²)
            'vlf_percent'   : VLF yüzde (%)
            'lf_percent'    : LF yüzde (%)
            'hf_percent'    : HF yüzde (%)
        """
        if len(rr_intervals_ms) < 10:
            warnings.warn(
                "Frekans düzlemi HRV için en az 10 RR aralığı önerilir. "
                f"Mevcut: {len(rr_intervals_ms)}",
                stacklevel=2,
            )
            return {
                "vlf_power_ms2": np.nan, "lf_power_ms2": np.nan,
                "hf_power_ms2": np.nan, "lf_hf_ratio": np.nan,
                "total_power_ms2": np.nan, "vlf_percent": np.nan,
                "lf_percent": np.nan, "hf_percent": np.nan,
            }

        rr_seconds = rr_intervals_ms / 1000.0  # ms → saniye

        # ── RR tachogram oluştur (düzgün örnekleme) ──
        # Kümülatif zaman ekseni
        rr_cumtime = np.cumsum(rr_seconds)
        rr_cumtime = np.insert(rr_cumtime, 0, 0.0)

        # Düzgün zaman ekseni (interpolation_fs Hz)
        t_uniform = np.arange(
            rr_cumtime[0],
            rr_cumtime[-1],
            1.0 / self.interpolation_fs,
        )

        # Cubic interpolation ile tachogram
        rr_values = np.concatenate([[rr_seconds[0]], rr_seconds])
        rr_interp = np.interp(t_uniform, rr_cumtime, rr_values)

        # DC bileşenini kaldır (ortalamayı çıkar)
        rr_interp = rr_interp - np.mean(rr_interp)

        # ── Welch PSD hesapla ──
        # Pencere: Hann, Segment: sinyal uzunluğunun 1/4'ü (min 256)
        nperseg = min(len(rr_interp), max(256, len(rr_interp) // 4))

        freqs, psd = welch(
            rr_interp,
            fs=self.interpolation_fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
            scaling="density",
        )

        # ── Bant gücü hesaplama (trapezoidal integral) ──
        def _band_power(f_low: float, f_high: float) -> float:
            """Belirli frekans bandındaki gücü hesaplar (ms² cinsinden)."""
            mask = (freqs >= f_low) & (freqs <= f_high)
            if not np.any(mask):
                return 0.0
            # PSD saniye² / Hz → ms² / Hz dönüşümü: × 1e6
            power = np.trapz(psd[mask], freqs[mask]) * 1e6
            return float(power)

        vlf_power = _band_power(*self.VLF_BAND)
        lf_power = _band_power(*self.LF_BAND)
        hf_power = _band_power(*self.HF_BAND)
        total_power = vlf_power + lf_power + hf_power

        # LF/HF oranı (sıfıra bölme koruması)
        lf_hf_ratio = lf_power / (hf_power + 1e-10)

        # Yüzdesel dağılım
        total_safe = total_power if total_power > 0 else 1e-10

        return {
            "vlf_power_ms2": vlf_power,
            "lf_power_ms2": lf_power,
            "hf_power_ms2": hf_power,
            "lf_hf_ratio": float(lf_hf_ratio),
            "total_power_ms2": total_power,
            "vlf_percent": float((vlf_power / total_safe) * 100.0),
            "lf_percent": float((lf_power / total_safe) * 100.0),
            "hf_percent": float((hf_power / total_safe) * 100.0),
        }

    def __repr__(self) -> str:
        return f"HRVAnalyzer(interp_fs={self.interpolation_fs}Hz)"


# ---------------------------------------------------------------------------
#  3) Örnek Entropisi (Sample Entropy — SampEn)
# ---------------------------------------------------------------------------
class SampleEntropyCalculator:
    """
    Örnek Entropi (Sample Entropy, SampEn) Hesaplayıcı.

    SampEn, bir zaman serisinin düzensizliğini (karmaşıklığını) ölçer.
    Fizyolojik sinyallerde düşük entropi, aşırı düzenlilik anlamına gelir
    ve kalp yetmezliği, otonom disfonksiyon gibi patolojilere işaret
    edebilir (Richman & Moorman, 2000).

    Yüksek SampEn → Sağlıklı, karmaşık kalp ritmi
    Düşük SampEn  → Patolojik, aşırı düzenli/monoton ritim

    Algoritma
    ---------
    1. m boyutlu vektörler oluştur: X_i = [x_i, x_i+1, ..., x_{i+m-1}]
    2. Her vektör çifti (i,j) için Chebyshev uzaklığını hesapla
    3. B_m: m boyutlu eşleşme sayısı (d < r)
    4. A_m: (m+1) boyutlu eşleşme sayısı
    5. SampEn = -ln(A_m / B_m)

    Parameters
    ----------
    m : int
        Gömme boyutu (embedding dimension). Varsayılan: 2
        → Ardışık 2 nokta kalıp olarak kullanılır.
    r_factor : float
        Tolerans çarpanı (r = r_factor × std). Varsayılan: 0.2
        → Standart sapmanın %20'si.
    """

    def __init__(self, m: int = 2, r_factor: float = 0.2):
        self.m = m
        self.r_factor = r_factor

    def compute(
        self,
        rr_intervals: np.ndarray,
        m: Optional[int] = None,
        r: Optional[float] = None,
    ) -> float:
        """
        RR aralık serisinin Örnek Entropisini hesaplar.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR aralıkları (ms veya normalize, 1D).
        m : int, optional
            Gömme boyutu. None ise __init__'teki değer kullanılır.
        r : float, optional
            Tolerans (mutlak). None ise r_factor × std(rr) kullanılır.

        Returns
        -------
        float
            SampEn değeri. Daha yüksek = daha düzensiz/karmaşık.
            Eğer hesaplanamıyorsa np.nan döner.
        """
        if m is None:
            m = self.m
        if r is None:
            r = self.r_factor * np.std(rr_intervals, ddof=1)

        N = len(rr_intervals)
        if N < m + 2:
            warnings.warn(
                f"SampEn hesaplamak için en az {m + 2} RR aralığı gerekli. "
                f"Mevcut: {N}",
                stacklevel=2,
            )
            return np.nan

        data = rr_intervals.astype(np.float64)

        def _count_matches(template_length: int) -> int:
            """
            template_length boyutlu vektörler arasında Chebyshev uzaklığı < r
            olan eşleşme sayısını hesaplar (self-match hariç).
            """
            count = 0
            n_templates = N - template_length
            for i in range(n_templates):
                for j in range(i + 1, n_templates):
                    # Chebyshev uzaklığı: max(|x_i[k] - x_j[k]|)
                    dist = np.max(
                        np.abs(
                            data[i: i + template_length]
                            - data[j: j + template_length]
                        )
                    )
                    if dist < r:
                        count += 1
            return count

        # B_m: m boyutlu eşleşme sayısı
        B = _count_matches(m)
        # A_m: (m+1) boyutlu eşleşme sayısı
        A = _count_matches(m + 1)

        # SampEn = -ln(A / B)
        if B == 0:
            warnings.warn(
                "SampEn hesaplanamadı: B_m = 0 (tolerans çok dar olabilir).",
                stacklevel=2,
            )
            return np.nan

        if A == 0:
            # A = 0 → sonsuz entropi (tamamen düzensiz)
            return float(np.inf)

        sampen = -np.log(A / B)
        return float(sampen)

    def __repr__(self) -> str:
        return f"SampleEntropyCalculator(m={self.m}, r_factor={self.r_factor})"


# ---------------------------------------------------------------------------
#  4) T-Dalga Alternansı (TWA) Analizi
# ---------------------------------------------------------------------------
class TWAAnalyzer:
    """
    T-Wave Alternans (T-Dalga Alternansı) Analizi.

    Atım-atım T-dalgası genlik farklılıklarını tespit eder.
    T-dalga alternansı, ventrikül repolarizasyonundaki atım-atım
    değişimleri gösterir ve ani kardiyak ölüm (SCD) için önemli
    bir risk belirteci olarak kullanılır (Rosenbaum et al., 1994).

    Yöntem
    ------
    1. R-tepe noktalarını referans alarak her atımın T-dalga bölgesini
       segmentler (R + offset → R + offset + T_window).
    2. Çift ve tek indeksli atımların T-dalga ortalamalarını hesaplar.
    3. TWA oranı = |T_even - T_odd| / mean(|T_all|)

    Parameters
    ----------
    t_wave_offset_ms : float
        R-tepe sonrası T-dalga başlangıç ofseti (ms). Varsayılan: 80ms.
    t_wave_window_ms : float
        T-dalga pencere süresi (ms). Varsayılan: 250ms.
    """

    def __init__(
        self,
        t_wave_offset_ms: float = 80.0,
        t_wave_window_ms: float = 250.0,
    ):
        self.t_wave_offset_ms = t_wave_offset_ms
        self.t_wave_window_ms = t_wave_window_ms

    def compute_twa(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: float,
    ) -> float:
        """
        T-dalga alternans oranını hesaplar.

        Parameters
        ----------
        signal : np.ndarray
            EKG sinyali (temizlenmiş).
        r_peaks : np.ndarray
            R-tepe noktası indeksleri.
        fs : float
            Örnekleme frekansı (Hz).

        Returns
        -------
        float
            TWA oranı (boyutsuz). Daha yüksek = daha belirgin alternans.
            Hesaplanamıyorsa np.nan döner.
        """
        if len(r_peaks) < 4:
            warnings.warn(
                "TWA hesaplamak için en az 4 R-tepe gerekli.",
                stacklevel=2,
            )
            return np.nan

        # T-dalga penceresi (örnek cinsinden)
        offset_samples = int(round(self.t_wave_offset_ms * fs / 1000.0))
        window_samples = int(round(self.t_wave_window_ms * fs / 1000.0))

        # Her atımdan T-dalga segmenti çıkar
        t_wave_amplitudes = []
        for peak in r_peaks:
            t_start = peak + offset_samples
            t_end = t_start + window_samples

            if t_end >= len(signal):
                break

            t_segment = signal[t_start:t_end]
            # T-dalga genliği: segment maksimumu (pikten pike)
            t_amp = np.max(t_segment) - np.min(t_segment)
            t_wave_amplitudes.append(t_amp)

        if len(t_wave_amplitudes) < 4:
            warnings.warn(
                "Yeterli T-dalga segmenti bulunamadı.",
                stacklevel=2,
            )
            return np.nan

        t_amps = np.array(t_wave_amplitudes)

        # Çift ve tek indeksli atımların T-dalga ortalaması
        t_even = np.mean(t_amps[0::2])  # Çift: 0, 2, 4, ...
        t_odd = np.mean(t_amps[1::2])   # Tek: 1, 3, 5, ...

        # TWA oranı
        mean_t = np.mean(np.abs(t_amps))
        if mean_t < 1e-10:
            return 0.0

        twa_ratio = np.abs(t_even - t_odd) / mean_t

        return float(twa_ratio)

    def __repr__(self) -> str:
        return (
            f"TWAAnalyzer(offset={self.t_wave_offset_ms}ms, "
            f"window={self.t_wave_window_ms}ms)"
        )


# ---------------------------------------------------------------------------
#  4.5) P-Q-R-S-T Kompleks Morfoloji Analizi (Osman Bey Güncellemesi)
# ---------------------------------------------------------------------------
class PQRSTExtractor:
    """
    R-tepe noktası bilinen bir EKG sinyalinde diğer majör dalgaların
    (P, Q, S, T) uç noktalarını tespit eder.

    Klinik Aralıklar:
    - QRS Genişliği: S_idx - Q_idx (ms)
    - PR Aralığı: R_idx - P_idx (ms)
    - QT Süresi: T_idx - Q_idx (ms)
    """
    def __init__(self, fs: float = 360.0):
        self.fs = fs

    def extract_waves(self, signal: np.ndarray, r_peak: int) -> Dict[str, Optional[int]]:
        """Tek bir R-tepesi etrafındaki P, Q, S, T dalgalarının indekslerini bulur."""
        q_window = int(0.05 * self.fs)   # 50 ms R öncesi
        s_window = int(0.05 * self.fs)   # 50 ms R sonrası
        p_window = int(0.20 * self.fs)   # Q öncesi 200 ms
        t_window = int(0.40 * self.fs)   # S sonrası 400 ms

        n = len(signal)
        waves = {"P": None, "Q": None, "R": r_peak, "S": None, "T": None}

        # Q Dalgası: R'den önceki lokal minimum
        q_search_start = max(0, r_peak - q_window)
        if q_search_start < r_peak:
            waves["Q"] = int(q_search_start + np.argmin(signal[q_search_start:r_peak]))

        # S Dalgası: R'den sonraki lokal minimum
        s_search_end = min(n, r_peak + s_window)
        if r_peak + 1 < s_search_end:
            waves["S"] = int(r_peak + 1 + np.argmin(signal[r_peak+1:s_search_end]))

        # P Dalgası: Q'dan önceki lokal maksimum
        if waves["Q"] is not None:
            q_idx = waves["Q"]
            p_search_start = max(0, q_idx - p_window)
            if p_search_start < q_idx:
                waves["P"] = int(p_search_start + np.argmax(signal[p_search_start:q_idx]))

        # T Dalgası: S'den sonraki lokal maksimum
        if waves["S"] is not None:
            s_idx = waves["S"]
            t_search_end = min(n, s_idx + t_window)
            if s_idx + 1 < t_search_end:
                waves["T"] = int(s_idx + 1 + np.argmax(signal[s_idx+1:t_search_end]))

        return waves

    def compute_intervals(self, waves: Dict[str, Optional[int]]) -> Dict[str, float]:
        """Dalga indekslerinden klinik süreleri (ms) hesaplar."""
        intervals = {"QRS_Width_ms": np.nan, "PR_Interval_ms": np.nan, "QT_Interval_ms": np.nan}
        ms_per_sample = 1000.0 / self.fs

        if waves["Q"] is not None and waves["S"] is not None:
            intervals["QRS_Width_ms"] = (waves["S"] - waves["Q"]) * ms_per_sample
        
        if waves["P"] is not None and waves["R"] is not None:
            intervals["PR_Interval_ms"] = (waves["R"] - waves["P"]) * ms_per_sample
            
        if waves["Q"] is not None and waves["T"] is not None:
            intervals["QT_Interval_ms"] = (waves["T"] - waves["Q"]) * ms_per_sample

        return intervals

# ---------------------------------------------------------------------------
#  5) Özellik Çıkarım Orkestratörü (Feature Extraction Pipeline)
# ---------------------------------------------------------------------------
class FeatureExtractionPipeline:
    """
    Modül 3 ana orkestratör sınıfı.

    Temizlenmiş EKG sinyalinden aşağıdaki özellik gruplarını çıkarır:
      - R-tepe noktaları (Pan-Tompkins)
      - RR aralıkları
      - HRV metrikleri (zaman + frekans düzlemi)
      - Örnek Entropi (SampEn)
      - T-Dalga Alternansı (TWA)
      - Bireysel kalp atımı segmentleri (CNN girişi için)

    Heartbeat Segmentation
    ----------------------
    Her atım, R-tepe noktası etrafında bir pencere ile kesilir:
      - pre_window  : R-tepe öncesi süre (varsayılan: 0.3s → ~108 örnek @ 360Hz)
      - post_window : R-tepe sonrası süre (varsayılan: 0.5s → ~180 örnek @ 360Hz)
      - Toplam segment: ~288 örnek @ 360Hz

    Kullanım
    --------
    >>> from ads1293_pipeline.module3_feature_extraction import FeatureExtractionPipeline
    >>> pipeline = FeatureExtractionPipeline()
    >>> result = pipeline.run(ecg_cleaned, fs=360.0)
    >>> print(result['hrv_time_domain'])
    >>> print(result['heartbeat_segments'].shape)

    Parameters
    ----------
    pre_window_sec : float
        R-tepe öncesi segment penceresi (saniye). Varsayılan: 0.3
    post_window_sec : float
        R-tepe sonrası segment penceresi (saniye). Varsayılan: 0.5
    sampen_m : int
        SampEn gömme boyutu. Varsayılan: 2
    sampen_r_factor : float
        SampEn tolerans çarpanı. Varsayılan: 0.2
    """

    def __init__(
        self,
        pre_window_sec: float = 0.3,
        post_window_sec: float = 0.5,
        sampen_m: int = 2,
        sampen_r_factor: float = 0.2,
    ):
        self.pre_window_sec = pre_window_sec
        self.post_window_sec = post_window_sec

        # Alt modülleri oluştur
        self.pan_tompkins = PanTompkinsDetector()
        self.hrv_analyzer = HRVAnalyzer()
        self.sampen_calculator = SampleEntropyCalculator(
            m=sampen_m, r_factor=sampen_r_factor,
        )
        self.twa_analyzer = TWAAnalyzer()
        self.pqrst_extractor = PQRSTExtractor()

    def _segment_heartbeats(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        R-tepe noktalarını referans alarak sinyali bireysel kalp atımı
        segmentlerine ayırır.

        Her segment: [R - pre_window, R + post_window]
        Sabit uzunlukta segmentler → CNN girişi için uygun tensör.

        Sınır dışı kalan atımlar (sinyal başı/sonu) atlanır.

        Parameters
        ----------
        signal : np.ndarray
            EKG sinyali.
        r_peaks : np.ndarray
            R-tepe indeksleri.
        fs : float
            Örnekleme frekansı.

        Returns
        -------
        segments : np.ndarray, shape=(n_valid_beats, segment_length)
            Heartbeat segmentleri.
        valid_peaks : np.ndarray
            Geçerli R-tepe indeksleri (sınır içi olanlar).
        """
        pre_samples = int(round(self.pre_window_sec * fs))
        post_samples = int(round(self.post_window_sec * fs))
        segment_length = pre_samples + post_samples

        segments = []
        valid_peaks = []

        for peak in r_peaks:
            start = peak - pre_samples
            end = peak + post_samples

            if start < 0 or end > len(signal):
                continue

            segment = signal[start:end]
            if len(segment) == segment_length:
                segments.append(segment)
                valid_peaks.append(peak)

        if not segments:
            warnings.warn(
                "Hiçbir geçerli heartbeat segmenti oluşturulamadı.",
                stacklevel=2,
            )
            return np.array([]).reshape(0, segment_length), np.array([], dtype=np.int64)

        return np.array(segments), np.array(valid_peaks, dtype=np.int64)

    def run(self, ecg_cleaned: np.ndarray, fs: float) -> Dict:
        """
        Modül 3 tam pipeline'ını çalıştırır.

        İşlem sırası:
        1. Pan-Tompkins ile R-tepe tespiti
        2. RR aralık hesaplama
        3. HRV analizi (zaman + frekans düzlemi)
        4. Örnek Entropi (SampEn)
        5. T-Dalga Alternansı (TWA)
        6. Heartbeat segmentasyonu

        Parameters
        ----------
        ecg_cleaned : np.ndarray
            Temizlenmiş EKG sinyali (Modül 2 çıkışı, μV veya Z-Score).
        fs : float
            Örnekleme frekansı (Hz).

        Returns
        -------
        dict
            'r_peaks'           : R-tepe indeksleri (np.ndarray)
            'rr_intervals_ms'   : RR aralıkları (ms, np.ndarray)
            'hrv_time_domain'   : Zaman düzlemi HRV (dict)
            'hrv_freq_domain'   : Frekans düzlemi HRV (dict)
            'sample_entropy'    : SampEn değeri (float)
            'twa_ratio'         : T-Dalga Alternansı oranı (float)
            'heartbeat_segments': Atım segmentleri (np.ndarray)
            'segment_r_peaks'   : Segment R-tepe indeksleri (np.ndarray)
            'segment_length'    : Segment uzunluğu (int)
            'fs'                : Örnekleme frekansı (float)
            'n_beats_detected'  : Tespit edilen atım sayısı (int)
        """
        # ── Adım 1: Pan-Tompkins R-tepe tespiti ──
        r_peaks = self.pan_tompkins.detect(ecg_cleaned, fs=fs)

        # ── Adım 2: RR aralık hesaplama ──
        rr_intervals_ms = self.pan_tompkins.get_rr_intervals(r_peaks, fs=fs)

        # ── Adım 3: HRV analizi ──
        hrv_time = self.hrv_analyzer.compute_time_domain(rr_intervals_ms)
        hrv_freq = self.hrv_analyzer.compute_frequency_domain(rr_intervals_ms)

        # ── Adım 4: Örnek Entropi ──
        sampen = self.sampen_calculator.compute(rr_intervals_ms)

        # ── Adım 5: T-Dalga Alternansı ──
        twa = self.twa_analyzer.compute_twa(ecg_cleaned, r_peaks, fs)

        # ── Adım 6: Heartbeat segmentasyonu ──
        segments, valid_peaks = self._segment_heartbeats(
            ecg_cleaned, r_peaks, fs,
        )

        # ── Adım 7: PQRST Analizi (Osman Bey Güncellemesi) ──
        self.pqrst_extractor.fs = fs
        pqrst_list = []
        for peak in r_peaks:
            waves = self.pqrst_extractor.extract_waves(ecg_cleaned, peak)
            intervals = self.pqrst_extractor.compute_intervals(waves)
            pqrst_list.append(intervals)
            
        # PQRST ortalamalarını al
        pqrst_avg = {
            "mean_QRS_Width_ms": np.nanmean([d["QRS_Width_ms"] for d in pqrst_list]),
            "mean_PR_Interval_ms": np.nanmean([d["PR_Interval_ms"] for d in pqrst_list]),
            "mean_QT_Interval_ms": np.nanmean([d["QT_Interval_ms"] for d in pqrst_list])
        }

        pre_samples = int(round(self.pre_window_sec * fs))
        post_samples = int(round(self.post_window_sec * fs))
        segment_length = pre_samples + post_samples

        return {
            "r_peaks": r_peaks,
            "rr_intervals_ms": rr_intervals_ms,
            "hrv_time_domain": hrv_time,
            "hrv_freq_domain": hrv_freq,
            "sample_entropy": sampen,
            "twa_ratio": twa,
            "pqrst_metrics": pqrst_avg,
            "heartbeat_segments": segments,
            "segment_r_peaks": valid_peaks,
            "segment_length": segment_length,
            "fs": fs,
            "n_beats_detected": len(r_peaks),
        }

    def __repr__(self) -> str:
        return (
            f"FeatureExtractionPipeline(\n"
            f"  pre_window={self.pre_window_sec}s, "
            f"post_window={self.post_window_sec}s,\n"
            f"  {self.pan_tompkins},\n"
            f"  {self.hrv_analyzer},\n"
            f"  {self.sampen_calculator},\n"
            f"  {self.twa_analyzer}\n"
            f")"
        )
