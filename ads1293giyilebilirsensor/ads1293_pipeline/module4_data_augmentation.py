# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 4: Veri Artırma ve Sınıf Dengesizliği Çözümü
=============================================================================
Bu modül, özellikle tıbbi veri setlerinde (MIT-BIH vb.) sıkça karşılaşılan
aşırı sınıf dengesizliği problemini çözmek için geliştirilmiştir.

1. BorderlineSMOTEBalancer
   - Azınlık sınıflarına sentetik veriler ekler. Sınırda bulunan (borderline)
     örneklere odaklanarak daha dayanıklı bir sınıflandırma sınırı oluşturur.
     
2. TimeSeriesAugmentor
   - Zaman serisi verilerine yönelik fiziksel koşulları simüle eden
     augmentasyon (veri artırma) yöntemleri uygular:
     * Temporal Warping: Kalp hızındaki fizyolojik değişimleri simüle eder.
     * Sensor Dropout: Elektrot temas kaybını (örn: terleme) simüle eder.
     * Gaussian Noise: Genel elektriksel gürültü ekler.
     * Amplitude Scaling: Sinyal genlik dalgalanmalarını simüle eder.

3. DataAugmentationPipeline
   - Orkestratör sınıfı, hem Borderline-SMOTE ile tablo verilerini hem de
     zaman serisi augmentasyonu ile raw EKG segmentlerini işler.
=============================================================================
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings

# imblearn opsiyonel bağımlılık
try:
    from imblearn.over_sampling import BorderlineSMOTE
    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False

from scipy.interpolate import interp1d

class BorderlineSMOTEBalancer:
    """
    Imbalanced-learn kütüphanesindeki BorderlineSMOTE algoritmasının sarmalayıcısı.
    MIT-BIH sınıfları arasındaki (Normal vs diğer aritmiler) aşırı dengesizliği
    gidermek için kullanılır.
    """
    def __init__(self, sampling_strategy='auto', k_neighbors=5, m_neighbors=10, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state
        
        if _IMBLEARN_AVAILABLE:
            self.smote = BorderlineSMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                m_neighbors=self.m_neighbors,
                random_state=self.random_state
            )
        else:
            self.smote = None
            warnings.warn("imblearn kütüphanesi yüklü değil. SMOTE işlemi yapılamayacak.")

    def map_mit_bih_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        MIT-BIH veri setindeki string etiketleri integer sınıflara çevirir.
        N -> 0 (Normal)
        L -> 1 (LBBB)
        R -> 2 (RBBB)
        V -> 3 (PVC)
        A -> 4 (APC)
        """
        mapping = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}
        mapped_labels = []
        for label in labels:
            mapped_labels.append(mapping.get(str(label), 0)) # Bilinmeyenler 0 (Normal) kabul edilir
        return np.array(mapped_labels)

    def balance(self, X_features: np.ndarray, y_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        X özelliklerini ve y etiketlerini dengeleyerek sentetik örnekler oluşturur.
        """
        if not _IMBLEARN_AVAILABLE:
            warnings.warn("BorderlineSMOTE uygulanamadı (imblearn eksik). Orijinal veri dönülüyor.")
            return X_features, y_labels
        
        # Sadece 1'den fazla sınıf varsa SMOTE uygulanabilir
        unique_classes = np.unique(y_labels)
        if len(unique_classes) <= 1:
            warnings.warn("SMOTE uygulamak için en az 2 sınıf gereklidir.")
            return X_features, y_labels
            
        X_resampled, y_resampled = self.smote.fit_resample(X_features, y_labels)
        return X_resampled, y_resampled


class TimeSeriesAugmentor:
    """
    EKG segmentleri için zaman serisi tabanlı veri artırma (Data Augmentation) sınıfı.
    """
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)

    def temporal_warp(self, signal: np.ndarray, speed_factor: Optional[float] = None) -> np.ndarray:
        """
        Zaman eksenini sıkıştırır veya genişletir (farklı nabız hızlarını simüle etmek için).
        """
        if speed_factor is None:
            speed_factor = self.rng.uniform(0.8, 1.2)
        
        orig_len = len(signal)
        orig_indices = np.linspace(0, orig_len - 1, num=orig_len)
        new_indices = np.linspace(0, orig_len - 1, num=int(orig_len * speed_factor))
        
        interpolator = interp1d(orig_indices, signal, kind='cubic', fill_value="extrapolate")
        warped_signal = interpolator(new_indices)
        
        # Orijinal uzunluğa geri dönmek için kırp veya pad yap
        if len(warped_signal) > orig_len:
            start = (len(warped_signal) - orig_len) // 2
            return warped_signal[start:start+orig_len]
        elif len(warped_signal) < orig_len:
            pad_len = orig_len - len(warped_signal)
            return np.pad(warped_signal, (0, pad_len), mode='edge')
        return warped_signal

    def sensor_dropout(self, signal: np.ndarray, max_dropout_ratio: float = 0.1) -> np.ndarray:
        """
        Sinyalin rastgele bir kısmını sıfırlar. Sensör/elektrot temas kaybını simüle eder.
        """
        sig_len = len(signal)
        dropout_len = int(sig_len * self.rng.uniform(0.01, max_dropout_ratio))
        start_idx = self.rng.randint(0, sig_len - dropout_len)
        
        augmented = signal.copy()
        augmented[start_idx:start_idx+dropout_len] = 0.0
        return augmented

    def gaussian_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Belirli bir SNR değerine sahip Gauss gürültüsü ekler.
        """
        sig_power = np.mean(signal**2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise

    def amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        """
        Rastgele bir çarpanla genliği ölçekler.
        """
        factor = self.rng.uniform(0.7, 1.3)
        return signal * factor

    def augment(self, signal: np.ndarray, method: str = 'random') -> np.ndarray:
        """
        Belirtilen yönteme göre tek bir sinyali augment eder.
        """
        methods = ['temporal_warp', 'sensor_dropout', 'gaussian_noise', 'amplitude_scaling']
        if method == 'random':
            method = self.rng.choice(methods)
            
        if method == 'temporal_warp':
            return self.temporal_warp(signal)
        elif method == 'sensor_dropout':
            return self.sensor_dropout(signal)
        elif method == 'gaussian_noise':
            return self.gaussian_noise(signal)
        elif method == 'amplitude_scaling':
            return self.amplitude_scaling(signal)
        else:
            return signal

    def augment_batch(self, signals: np.ndarray, labels: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Toplu haldeki sinyalleri çoğaltarak yeni bir veri seti oluşturur.
        """
        aug_signals = []
        aug_labels = []
        
        # Orijinalleri ekle
        for s, l in zip(signals, labels):
            aug_signals.append(s)
            aug_labels.append(l)
            
            # Belirtilen faktör kadar augmentasyon ekle
            for _ in range(augmentation_factor - 1):
                aug_signals.append(self.augment(s, method='random'))
                aug_labels.append(l)
                
        return np.array(aug_signals), np.array(aug_labels)


class DataAugmentationPipeline:
    """
    Hem SMOTE hem de Time Series Augmentation işlemlerini yöneten orkestratör sınıfı.
    """
    def __init__(self, random_state=42):
        self.balancer = BorderlineSMOTEBalancer(random_state=random_state)
        self.ts_augmentor = TimeSeriesAugmentor(random_state=random_state)

    def run(self, X_segments: np.ndarray, y_labels: np.ndarray, X_features: Optional[np.ndarray] = None) -> Dict:
        """
        Veri artırma pipeline'ını çalıştırır.
        """
        results = {}
        
        # Adım 1: Sinyalleri çoğalt
        X_aug_segments, y_aug_labels = self.ts_augmentor.augment_batch(X_segments, y_labels, augmentation_factor=2)
        results['X_segments'] = X_aug_segments
        results['y_labels'] = y_aug_labels
        
        # Adım 2: Tablo verileri (HRV özellikleri vb) varsa onları SMOTE ile dengele
        if X_features is not None:
            X_bal_features, y_bal_labels = self.balancer.balance(X_features, y_labels)
            results['X_features'] = X_bal_features
            results['y_features_labels'] = y_bal_labels
            
        return results
