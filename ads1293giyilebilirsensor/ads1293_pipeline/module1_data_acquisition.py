# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 1: Donanım-Yazılım Arayüzü – Ham Veri Akışının (Raw Data Pipeline) Yönetimi
=============================================================================
Bu modül, ADS1293 çipinden gelen ham veri akışını yönetir:

  1. PhysioNet (MIT-BIH) üzerinden gerçek EKG verisi çekme ve
     ADS1293 simülasyonu (24-bit ADC formatına dönüştürme).
  2. Ring Buffer (Dairesel Bellek) mimarisi ile asenkron veri tamponlama.
  3. 24-Bit Two's Complement → μV (gerçek genlik) ölçekleme.
  4. Zaman damgası tabanlı kayıp paket tespiti ve
     Cubic Spline İnterpolasyonu ile Data Imputation.
  5. ADC Satürasyon (sinyal kırpılması) tespiti ve
     Dinamik Kazanç Kontrolü (PGA Recovery).

Referans Formül:
    V_LSB = V_REF / (Gain × 2^23)
    V_REF = 2.4 V, Gain = 3.5 → V_LSB ≈ 0.081711 μV/LSB
=============================================================================
"""

import numpy as np
from scipy.interpolate import CubicSpline
from collections import deque
from typing import Optional, Tuple, List, Dict
import warnings


# ---------------------------------------------------------------------------
#  1) PhysioNet MIT-BIH Veri Çekme ve ADS1293 Simülasyonu
# ---------------------------------------------------------------------------
class PhysioNetLoader:
    """
    PhysioNet MIT-BIH Aritmia Veritabanından gerçek EKG verisi çeker
    ve ADS1293 çipinden geliyormuş gibi 24-bit ham ADC formatına dönüştürür.

    Attributes
    ----------
    record_id : str
        MIT-BIH kayıt numarası (örn: '100', '101', '207').
    channel : int
        Kullanılacak EKG kanalı (0 veya 1).
    fs : float
        Örnekleme frekansı (MIT-BIH = 360 Hz).
    v_ref : float
        ADS1293 referans voltajı (2.4 V).
    gain : float
        ADS1293 PGA kazancı (3.5x).
    """

    # ADS1293 donanım sabitleri
    V_REF: float = 2.4            # Volt
    GAIN_DEFAULT: float = 3.5     # PGA varsayılan kazanç
    ADC_BITS: int = 24            # 24-bit çözünürlük
    ADC_MAX: int = 2**23          # Two's Complement pozitif maksimum
    ADC_RAIL: int = 0x7FFFFF      # Satürasyon üst sınırı

    def __init__(
        self,
        record_id: str = "100",
        channel: int = 0,
        database: str = "mitdb",
        v_ref: float = 2.4,
        gain: float = 3.5,
    ):
        self.record_id = record_id
        self.channel = channel
        self.database = database
        self.v_ref = v_ref
        self.gain = gain
        self.fs: Optional[float] = None
        self._raw_signal: Optional[np.ndarray] = None
        self._annotations = None

        # V_LSB hesapla: V_REF / (Gain × 2^23) → μV/LSB
        self.v_lsb = (self.v_ref / (self.gain * self.ADC_MAX)) * 1e6  # μV cinsinden

    def fetch_record(
        self,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        import wfdb
        import os
        import time

        # Local cache path
        cache_dir = os.path.expanduser("~/.ads1293_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{self.database}_{self.record_id}_ch{self.channel}.npz")

        sampfrom = 0
        sampto = None
        
        # Load from cache if exists
        if os.path.exists(cache_file):
            print(f"[PhysioNetLoader] Önbellekten yükleniyor: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            self.fs = data["fs"].item()
            ecg_uv = data["ecg_uv"]
            raw_adc_24bit = data["raw_adc_24bit"]
            timestamps = data["timestamps"]
            annotations_dict = data["annotations"].item() if "annotations" in data and data["annotations"].item() is not None else None
            
            if annotations_dict:
                class DummyAnn:
                    def __init__(self, sample, symbol):
                        self.sample = sample
                        self.symbol = symbol
                self._annotations = DummyAnn(annotations_dict["sample"], annotations_dict["symbol"])
            else:
                self._annotations = None

            if duration_seconds is not None:
                sampto = int(duration_seconds * self.fs)
                ecg_uv = ecg_uv[:sampto]
                raw_adc_24bit = raw_adc_24bit[:sampto]
                timestamps = timestamps[:sampto]
                if self._annotations is not None:
                    mask = self._annotations.sample < sampto
                    self._annotations.sample = self._annotations.sample[mask]
                    self._annotations.symbol = list(np.array(self._annotations.symbol)[mask])

            self._raw_signal = raw_adc_24bit
            return {
                "ecg_uv": ecg_uv,
                "raw_adc_24bit": raw_adc_24bit,
                "timestamps": timestamps,
                "annotations": self._annotations,
                "fs": self.fs,
            }

        # Otherwise, fetch from PhysioNet with retry logic
        print(f"[PhysioNetLoader] PhysioNet'ten indiriliyor: {self.database}/{self.record_id}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if duration_seconds is not None:
                    header = wfdb.rdheader(self.record_id, pn_dir=self.database)
                    self.fs = header.fs
                    sampto = int(duration_seconds * self.fs)

                record = wfdb.rdrecord(
                    self.record_id,
                    pn_dir=self.database,
                    sampfrom=sampfrom,
                    sampto=sampto,
                    channels=[self.channel],
                )
                self.fs = record.fs
                ecg_mv = record.p_signal[:, 0]  # mV
                break
            except Exception as e:
                print(f"İndirme hatası (Deneme {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        else:
            raise ConnectionError(f"PhysioNet'e bağlanılamadı: {self.database}/{self.record_id}")

        try:
            self._annotations = wfdb.rdann(
                self.record_id,
                "atr",
                pn_dir=self.database,
                sampfrom=sampfrom,
                sampto=sampto,
            )
        except Exception:
            self._annotations = None

        ecg_uv = ecg_mv * 1000.0  # 1 mV = 1000 μV
        raw_adc_float = ecg_uv / self.v_lsb
        raw_adc_24bit = np.clip(
            np.round(raw_adc_float).astype(np.int32),
            -self.ADC_MAX,
            self.ADC_MAX - 1,
        )
        timestamps = np.arange(len(ecg_uv)) / self.fs
        self._raw_signal = raw_adc_24bit
        
        # Save to cache
        ann_dict = None
        if self._annotations is not None:
            ann_dict = {"sample": self._annotations.sample, "symbol": self._annotations.symbol}
            
        np.savez_compressed(
            cache_file, 
            ecg_uv=ecg_uv, 
            raw_adc_24bit=raw_adc_24bit, 
            timestamps=timestamps, 
            fs=self.fs,
            annotations=ann_dict
        )

        return {
            "ecg_uv": ecg_uv,
            "raw_adc_24bit": raw_adc_24bit,
            "timestamps": timestamps,
            "annotations": self._annotations,
            "fs": self.fs,
        }

    def get_annotation_labels(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        if self._annotations is None:
            return None
        return self._annotations.sample, self._annotations.symbol



# ---------------------------------------------------------------------------
#  2) Ring Buffer (Dairesel Bellek) Mimarisi
# ---------------------------------------------------------------------------
class RingBuffer:
    """
    ADS1293 SPI arayüzünden gelen yüksek hızlı veri paketlerini
    tamponlamak için O(1) amortize karmaşıklıkta dairesel bellek.

    Paket düşmesini (Data Drop) önlemek için yazılım katmanında
    interrupt (kesme) mimarisini simüle eder.

    Parameters
    ----------
    capacity : int
        Tampon kapasitesi (örnek sayısı). Varsayılan: 2048 (≈4 saniyelik
        veri @ 500 Hz).
    """

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._overflow_count: int = 0
        self._total_writes: int = 0

    def write(self, sample: int, timestamp: float) -> None:
        """
        Yeni bir ADC örneği ve zaman damgasını tampona yazar.
        Tampon dolduğunda en eski veri otomatik silinir (FIFO).
        """
        if len(self._buffer) == self.capacity:
            self._overflow_count += 1

        self._buffer.append((sample, timestamp))
        self._total_writes += 1

    def write_batch(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Toplu yazma: bir veri bloğunu tampona yazar."""
        for s, t in zip(samples, timestamps):
            self.write(int(s), float(t))

    def read(self, n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tampondan n örnek okur ve tamponu boşaltır.
        n=None ise tüm tampon okunur.

        Returns
        -------
        samples : np.ndarray (int32)
        timestamps : np.ndarray (float64)
        """
        if n is None:
            n = len(self._buffer)

        n = min(n, len(self._buffer))
        samples, timestamps = [], []

        for _ in range(n):
            s, t = self._buffer.popleft()
            samples.append(s)
            timestamps.append(t)

        return np.array(samples, dtype=np.int32), np.array(timestamps, dtype=np.float64)

    @property
    def size(self) -> int:
        """Tamponda bekleyen örnek sayısı."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.capacity

    @property
    def overflow_count(self) -> int:
        """Kaç kez tampon taşması yaşandığı."""
        return self._overflow_count

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self.capacity}, "
            f"size={self.size}, overflows={self._overflow_count})"
        )


# ---------------------------------------------------------------------------
#  3) 24-Bit Two's Complement → μV Ölçekleme Motoru
# ---------------------------------------------------------------------------
class ADCScaler:
    """
    ADS1293 24-bit ham ADC değerlerini gerçek elektriksel genliğe (μV) dönüştürür.

    Rapordaki formül:
        V_LSB = V_REF / (Gain × 2^23)
        V_REF = 2.4 V, Gain = 3.5
        → V_LSB ≈ 0.081711 μV/LSB

    Her ham byte, O(1) zaman karmaşıklığında bu katsayı ile çarpılarak
    doğrudan elektriksel genliğe ölçeklenir.
    """

    def __init__(self, v_ref: float = 2.4, gain: float = 3.5):
        self.v_ref = v_ref
        self.gain = gain
        self.adc_max = 2**23  # 8,388,608

        # V_LSB hesabı: μV/LSB
        self.v_lsb_uv = (self.v_ref / (self.gain * self.adc_max)) * 1e6
        # Hesaplanan değer ≈ 0.081711 μV/LSB

    def raw_to_uv(self, raw_adc: np.ndarray) -> np.ndarray:
        """
        24-bit ham ADC (Two's Complement int32) → μV dönüşümü.
        O(1) per-sample karmaşıklık (vektörize çarpma).

        Parameters
        ----------
        raw_adc : np.ndarray, dtype=int32
            Ham 24-bit ADC değerleri [-2^23, 2^23-1].

        Returns
        -------
        np.ndarray, dtype=float64
            Mikrovolt (μV) cinsinden sinyal genliği.
        """
        return raw_adc.astype(np.float64) * self.v_lsb_uv

    def uv_to_raw(self, ecg_uv: np.ndarray) -> np.ndarray:
        """Ters dönüşüm: μV → 24-bit raw ADC (simülasyon amaçlı)."""
        raw = np.round(ecg_uv / self.v_lsb_uv).astype(np.int32)
        return np.clip(raw, -self.adc_max, self.adc_max - 1)

    def __repr__(self) -> str:
        return (
            f"ADCScaler(V_REF={self.v_ref}V, Gain={self.gain}x, "
            f"V_LSB≈{self.v_lsb_uv:.6f} μV/LSB)"
        )


# ---------------------------------------------------------------------------
#  4) Veri Senkronizasyonu ve Kayıp Paket Kurtarma (Data Imputation)
# ---------------------------------------------------------------------------
class DataSynchronizer:
    """
    Giyilebilir sistem BLE/SPI iletiminde oluşan paket kayıplarını tespit eder
    ve Cubic Spline İnterpolasyonu ile kayıp veriyi kurtarır.

    Ayrıca ADC satürasyon (Rail-to-Rail flatline) tespiti yapar.

    Parameters
    ----------
    fs : float
        Örnekleme frekansı (Hz).
    gap_threshold_factor : float
        Normal örnekleme periyodunun kaç katından büyük boşlukların
        kayıp paket olarak işaretleneceği. Varsayılan: 1.5
    saturation_threshold : int
        ADC satürasyon eşiği. Varsayılan: 0x7FFFFF (24-bit max)
    saturation_consecutive : int
        Art arda kaç satüre örneğin satürasyon olarak sayılacağı.
    """

    def __init__(
        self,
        fs: float = 360.0,
        gap_threshold_factor: float = 1.5,
        saturation_threshold: int = 0x7FFFFF,
        saturation_consecutive: int = 5,
    ):
        self.fs = fs
        self.sample_period = 1.0 / fs  # saniye
        self.gap_threshold = self.sample_period * gap_threshold_factor
        self.saturation_threshold = saturation_threshold
        self.saturation_consecutive = saturation_consecutive

    def detect_missing_packets(
        self, timestamps: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Zaman damgalarındaki boşlukları tespit eder.

        Returns
        -------
        list of (start_idx, end_idx, gap_duration_ms)
            Tespit edilen her boşluk için başlangıç indeksi,
            bitiş indeksi ve boşluk süresi (ms).
        """
        gaps = []
        dt = np.diff(timestamps)

        for i, delta in enumerate(dt):
            if delta > self.gap_threshold:
                gap_ms = delta * 1000.0
                gaps.append((i, i + 1, gap_ms))

        return gaps

    def interpolate_missing(
        self,
        raw_adc: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kayıp paketleri Cubic Spline İnterpolasyonu ile doldurur.
        Sıfırla doldurmak yapay zekayı bozar — bunun yerine kübik eğri
        uydurma ile kaybolan verinin ne olabileceği matematiksel olarak
        tahmin edilir.

        Returns
        -------
        filled_adc : np.ndarray
            Boşlukları doldurulmuş ADC verileri.
        filled_timestamps : np.ndarray
            Yeniden oluşturulmuş düzgün zaman ekseni.
        """
        gaps = self.detect_missing_packets(timestamps)

        if not gaps:
            return raw_adc.copy(), timestamps.copy()

        # Düzgün (uniform) zaman ekseni oluştur
        t_start = timestamps[0]
        t_end = timestamps[-1]
        n_total = int(np.round((t_end - t_start) * self.fs)) + 1
        t_uniform = np.linspace(t_start, t_end, n_total)

        # Mevcut verilere Cubic Spline uydur
        cs = CubicSpline(timestamps, raw_adc.astype(np.float64), extrapolate=False)
        filled = cs(t_uniform)

        # NaN'leri en yakın geçerli değerle doldur (extrapolation sınırlarında)
        nan_mask = np.isnan(filled)
        if np.any(nan_mask):
            filled[nan_mask] = np.interp(
                t_uniform[nan_mask], timestamps, raw_adc.astype(np.float64)
            )

        return np.round(filled).astype(np.int32), t_uniform

    def detect_saturation(
        self, raw_adc: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        ADC satürasyon (Rail-to-Rail flatline) bölgelerini tespit eder.

        Hasta aniden hareket ettiğinde veya elektrota statik elektrik
        geldiğinde, PGA aşırı yüklenir ve ham veri tepe noktaya yapışır.

        Returns
        -------
        list of (start_idx, end_idx)
            Satüre olmuş bölgelerin başlangıç ve bitiş indeksleri.
        """
        abs_adc = np.abs(raw_adc)
        saturated = abs_adc >= (self.saturation_threshold - 1)  # ±max değer

        regions = []
        in_saturation = False
        start = 0
        consecutive = 0

        for i, is_sat in enumerate(saturated):
            if is_sat:
                if not in_saturation:
                    start = i
                    in_saturation = True
                consecutive += 1
            else:
                if in_saturation and consecutive >= self.saturation_consecutive:
                    regions.append((start, i))
                in_saturation = False
                consecutive = 0

        # Son bölge kontrolü
        if in_saturation and consecutive >= self.saturation_consecutive:
            regions.append((start, len(raw_adc)))

        return regions

    def apply_dynamic_gain_control(
        self,
        raw_adc: np.ndarray,
        current_gain: float = 3.5,
        recovery_gain: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Dinamik Kazanç Kontrolü: Satürasyon tespit edildiğinde
        PGA kazancını düşürerek sinyalin okunabilir bölgeye dönmesini sağlar.

        Parameters
        ----------
        raw_adc : np.ndarray
            Ham ADC verileri.
        current_gain : float
            Mevcut PGA kazancı.
        recovery_gain : float
            Satürasyon durumunda uygulanacak düşük kazanç.

        Returns
        -------
        corrected_adc : np.ndarray
            Kazanç düzeltmesi uygulanmış veriler.
        effective_gain : float
            Uygulanan efektif kazanç değeri.
        """
        saturation_regions = self.detect_saturation(raw_adc)

        if not saturation_regions:
            return raw_adc.copy(), current_gain

        corrected = raw_adc.astype(np.float64).copy()
        gain_ratio = current_gain / recovery_gain

        for start, end in saturation_regions:
            # Satüre bölgelerde kazancı düşür (sinyal daraltma)
            corrected[start:end] = corrected[start:end] / gain_ratio

        return np.round(corrected).astype(np.int32), recovery_gain


# ---------------------------------------------------------------------------
#  5) Modül 1 Entegre Pipeline
# ---------------------------------------------------------------------------
class RawDataPipeline:
    """
    Modül 1 ana orkestratör sınıfı.
    PhysioNet verisi çeker → Ring Buffer'a yazar →
    24-bit ölçekleme → Kayıp veri kurtarma → ADC kontrol.

    Kullanım
    --------
    >>> pipeline = RawDataPipeline(record_id='100', duration_seconds=30)
    >>> result = pipeline.run()
    >>> print(result['ecg_uv'].shape)    # μV cinsinden temiz sinyal
    >>> print(result['fs'])              # Örnekleme frekansı
    """

    def __init__(
        self,
        record_id: str = "100",
        channel: int = 0,
        duration_seconds: Optional[float] = 30.0,
        ring_buffer_capacity: int = 4096,
        v_ref: float = 2.4,
        gain: float = 3.5,
        simulate_packet_loss: bool = True,
        packet_loss_rate: float = 0.005,
    ):
        self.loader = PhysioNetLoader(
            record_id=record_id,
            channel=channel,
            v_ref=v_ref,
            gain=gain,
        )
        self.ring_buffer = RingBuffer(capacity=ring_buffer_capacity)
        self.scaler = ADCScaler(v_ref=v_ref, gain=gain)
        self.synchronizer: Optional[DataSynchronizer] = None

        self.duration_seconds = duration_seconds
        self.simulate_packet_loss = simulate_packet_loss
        self.packet_loss_rate = packet_loss_rate

    def _inject_packet_loss(
        self,
        raw_adc: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gerçek dünya BLE paket kaybını simüle eder.
        Rastgele örnekleri kaldırarak zaman damgasında boşluklar oluşturur.
        """
        n = len(raw_adc)
        n_drop = max(1, int(n * self.packet_loss_rate))

        # Rastgele bloklar halinde paket düşür (tekil değil, ardışık)
        rng = np.random.default_rng(42)
        drop_starts = rng.choice(n - 10, size=n_drop // 3 + 1, replace=False)

        drop_mask = np.ones(n, dtype=bool)
        for ds in drop_starts:
            block_len = rng.integers(2, 8)
            drop_mask[ds : ds + block_len] = False

        return raw_adc[drop_mask], timestamps[drop_mask]

    def fetch_data(self, duration_seconds: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Doğrudan PhysioNetLoader'dan veri çeker (gui_app.py uyumluluğu için)."""
        if duration_seconds is not None:
            self.duration_seconds = duration_seconds
        return self.loader.fetch_record(duration_seconds=self.duration_seconds)

    def run(self) -> Dict:
        """
        Modül 1 tam pipeline'ını çalıştırır.

        Returns
        -------
        dict
            'ecg_uv'          : μV cinsinden ölçeklenmiş sinyal
            'raw_adc_24bit'   : Ham 24-bit ADC değerleri (kurtarma sonrası)
            'timestamps'      : Zaman damgaları
            'fs'              : Örnekleme frekansı
            'annotations'     : MIT-BIH anotasyonları
            'pipeline_stats'  : Pipeline istatistikleri
        """
        # ── Adım 1: PhysioNet'ten veri çek ──
        data = self.loader.fetch_record(duration_seconds=self.duration_seconds)
        raw_adc = data["raw_adc_24bit"]
        timestamps = data["timestamps"]
        fs = data["fs"]

        self.synchronizer = DataSynchronizer(fs=fs)

        # ── Adım 2: Paket kaybı simülasyonu ──
        gaps_before = []
        if self.simulate_packet_loss:
            raw_adc, timestamps = self._inject_packet_loss(raw_adc, timestamps)

        # ── Adım 3: Ring Buffer üzerinden veri akışı ──
        self.ring_buffer.write_batch(raw_adc, timestamps)
        buffered_adc, buffered_ts = self.ring_buffer.read()

        # ── Adım 4: Kayıp paket tespiti ve Cubic Spline kurtarma ──
        detected_gaps = self.synchronizer.detect_missing_packets(buffered_ts)
        filled_adc, filled_ts = self.synchronizer.interpolate_missing(
            buffered_adc, buffered_ts
        )

        # ── Adım 5: ADC satürasyon tespiti ──
        saturation_regions = self.synchronizer.detect_saturation(filled_adc)
        corrected_adc, effective_gain = self.synchronizer.apply_dynamic_gain_control(
            filled_adc
        )

        # ── Adım 6: 24-bit → μV ölçekleme (O(1) per-sample) ──
        ecg_uv = self.scaler.raw_to_uv(corrected_adc)

        # ── Pipeline istatistikleri ──
        stats = {
            "total_samples_raw": len(data["raw_adc_24bit"]),
            "samples_after_loss": len(buffered_adc),
            "samples_after_interpolation": len(filled_adc),
            "detected_gaps": len(detected_gaps),
            "saturation_regions": len(saturation_regions),
            "ring_buffer_overflows": self.ring_buffer.overflow_count,
            "effective_gain": effective_gain,
            "v_lsb_uv": self.scaler.v_lsb_uv,
            "adc_scaler": repr(self.scaler),
        }

        return {
            "ecg_uv": ecg_uv,
            "raw_adc_24bit": corrected_adc,
            "timestamps": filled_ts,
            "fs": fs,
            "annotations": data["annotations"],
            "pipeline_stats": stats,
        }

# ---------------------------------------------------------------------------
#  6) Modül 1 Çevrimdışı (Offline) Veri Akışı (Yeni Veriler İçin)
# ---------------------------------------------------------------------------
class OfflineDataPipeline:
    """
    İnternet bağlantısı olmadan yerel .csv veya .txt dosyalarından (Osman Bey'in verileri)
    EKG verisini okuyan ve pipeline'a sokan sınıf.
    """
    def __init__(self, file_path: str, fs: float = 360.0):
        self.file_path = file_path
        self.fs = fs

    def run(self) -> Dict:
        """
        Yerel dosyayı okur ve yapay zekanın beklediği formata sokar.
        """
        import os
        
        # Dosya uzantısına göre okuma stratejisi
        if self.file_path.endswith('.csv'):
            # Tek sütunlu csv varsayımı (lifeline.csv)
            raw_data = np.genfromtxt(self.file_path, delimiter=',')
        elif self.file_path.endswith('.txt'):
            # Çok sütunlu virgülle ayrılmış txt varsayımı (2025-06-12_Rec001.txt)
            raw_data = np.genfromtxt(self.file_path, delimiter=',', usecols=0)
        else:
            raise ValueError("Desteklenmeyen dosya formatı. Sadece .csv veya .txt")

        # Veri içindeki NaN değerlerini temizle
        raw_data = np.nan_to_num(raw_data)
        
        # Normalde bu veri ham mV veya ADC olabilir. 
        # Biz doğrudan mikrovolt (uV) gibi veya normalize edilecek şekilde pipeline'a veriyoruz.
        # Simülasyon için 1D array olmalı.
        ecg_uv = raw_data.flatten()
        
        # Zaman damgaları
        timestamps = np.arange(len(ecg_uv)) / self.fs
        
        return {
            "ecg_uv": ecg_uv,
            "raw_adc_24bit": ecg_uv, # Simüle adc
            "timestamps": timestamps,
            "fs": self.fs,
            "annotations": None,
            "pipeline_stats": {"source": "offline_file", "total_samples": len(ecg_uv)}
        }

