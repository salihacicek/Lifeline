# -*- coding: utf-8 -*-
"""
Modül 5 — Uzamsal-Zamansal (Spatio-Temporal) Derin Öğrenme Mimarisi
=====================================================================
ADS1293 Giyilebilir EKG Telemetri Sistemi · TEKNOFEST

Bu modül, tek-kanal EKG segmentlerinden aritmi sınıflandırması yapan bir
**1D-CNN + BiLSTM + Bahdanau Attention** hibrit mimarisi içerir.  İkincil
olarak, HRV / istatistiksel öznitelikler için bir **XGBoost** sınıflandırıcı
sarmalayıcısı da sunar.

Sınıf Hiyerarşisi
------------------
1. ``MultiScaleCNN1D``   — Çoklu çekirdek boyutunda 1D konvolüsyon bloğu
2. ``BahdanauAttention`` — Dikkat mekanizması (kritik segmentlere odaklanma)
3. ``ECGHybridModel``    — Ana hibrit model (CNN → BiLSTM → Attention → FC)
4. ``XGBoostClassifier`` — Tablo verisi için XGBoost sarmalayıcısı

Yardımcı Fonksiyonlar
---------------------
- ``get_device()`` — Apple Silicon MPS > CUDA > CPU cihaz seçimi

Hedef Sınıflar (MIT-BIH standardı)
-----------------------------------
0 = Normal (N), 1 = LBBB (Sol dal bloğu), 2 = RBBB (Sağ dal bloğu),
3 = PVC (Erken ventrikül kasılması), 4 = APC (Erken atriyal kasılma)

Notlar
------
* Tüm PyTorch katmanları ``float32`` hassasiyetinde çalışır.
* MPS arka ucu, Apple Silicon (M1/M2/M3/M4) çiplerinde GPU ivmelemesi sağlar.
* XGBoost bileşeni CPU üzerinde çalışır (gradient boosting için GPU avantajı sınırlı).

Yazar  : ADS1293 Telemetri Takımı
Tarih  : 2025
Lisans : MIT
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# XGBoost opsiyonel — yüklü değilse uyarı verilir, hata fırlatılmaz.
try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Yardımcı: Cihaz Seçimi
# ──────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Apple Silicon MPS > NVIDIA CUDA > CPU sıralamasıyla en uygun
    PyTorch hesaplama cihazını döndürür.

    Returns
    -------
    torch.device
        Seçilen hesaplama cihazı.

    Examples
    --------
    >>> device = get_device()
    >>> model = ECGHybridModel().to(device)
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS cihazı seçildi.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("NVIDIA CUDA cihazı seçildi: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        logger.info("CPU cihazı seçildi (GPU bulunamadı).")
        return torch.device("cpu")


# ======================================================================
# Sınıf 1 — MultiScaleCNN1D
# ======================================================================

class MultiScaleCNN1D(nn.Module):
    """Çoklu ölçekli 1D konvolüsyon bloğu.

    Üç paralel konvolüsyon dalı farklı çekirdek boyutlarıyla uzamsal
    (morfolojik) örüntüleri yakalar:

    - **kernel=31** → Yavaş T-dalgası morfolojileri
    - **kernel=15** → Orta ölçekli P/T dalga geçişleri
    - **kernel=7**  → Keskin QRS kompleksi

    Her dal şu ardışık düzeni uygular::

        Conv1d  →  BatchNorm1d  →  ReLU  →  MaxPool1d(2)

    Son olarak üç dalın çıktıları kanal boyutunda birleştirilir
    (concatenate).

    Parameters
    ----------
    in_channels : int, default=1
        Girdi kanal sayısı (tek derivasyon EKG için 1).
    out_channels_per_branch : int, default=32
        Her dalın ürettiği filtre sayısı.  Toplam çıktı kanalı
        ``3 * out_channels_per_branch`` olur.
    dropout : float, default=0.3
        Birleştirilmiş çıktıya uygulanan dropout oranı.

    Shape
    -----
    - Girdi:  ``(batch, in_channels, seq_len)``
    - Çıktı: ``(batch, 3*out_channels_per_branch, seq_len//2)``
      (MaxPool stride=2 nedeniyle uzunluk yarılanır.)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels_per_branch: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.out_channels_per_branch = out_channels_per_branch

        # Dal 1 — Yavaş T-dalgası (geniş çekirdek)
        self.branch_slow = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels_per_branch,
                kernel_size=31,
                padding=15,   # "same" benzeri dolgu
                bias=False,
            ),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dal 2 — Orta ölçek
        self.branch_medium = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels_per_branch,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dal 3 — Keskin QRS (dar çekirdek)
        self.branch_sharp = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels_per_branch,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Birleştirilmiş çıktı için dropout
        self.dropout = nn.Dropout(p=dropout)

    # ------------------------------------------------------------------
    @property
    def total_out_channels(self) -> int:
        """Birleştirilmiş toplam çıktı kanalı sayısı."""
        return 3 * self.out_channels_per_branch

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş.

        Parameters
        ----------
        x : Tensor, shape ``(B, C_in, L)``
            B = batch, C_in = girdi kanalı, L = sekans uzunluğu.

        Returns
        -------
        Tensor, shape ``(B, 3*out_channels_per_branch, L//2)``
        """
        out_slow = self.branch_slow(x)      # (B, 32, L//2)
        out_med = self.branch_medium(x)     # (B, 32, L//2)
        out_sharp = self.branch_sharp(x)    # (B, 32, L//2)

        # Kanal boyutunda birleştir → (B, 96, L//2)
        concatenated = torch.cat([out_slow, out_med, out_sharp], dim=1)
        return self.dropout(concatenated)


# ======================================================================
# Sınıf 2 — BahdanauAttention
# ======================================================================

class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Dikkat Mekanizması.

    LSTM'den gelen tüm zaman adımlarının gizli durumlarına dikkat uygular
    ve kritik segmentlere (aritmi anları, QRS tepe noktaları vb.) ağırlık
    verir.

    Matematiksel Formülasyon
    ------------------------
    .. math::

        e_t     &= V^T \\tanh(W \\cdot h_t) \\\\
        \\alpha_t &= \\text{softmax}(e_t) \\\\
        c        &= \\sum_t \\alpha_t \\cdot h_t

    Burada :math:`h_t` zaman adımı *t*'deki LSTM gizli durumu,
    :math:`\\alpha_t` dikkat ağırlığı ve :math:`c` bağlam vektörüdür.

    Parameters
    ----------
    hidden_dim : int
        LSTM gizli boyut (bidirectional ise 2*hidden_size).
    attention_dim : int, default=64
        Dikkat ara katman boyutu (W dönüşümünün çıktı boyutu).

    Shape
    -----
    - Girdi:  ``(B, T, hidden_dim)``  — LSTM çıktıları
    - Çıktılar:
      - ``context``   : ``(B, hidden_dim)``
      - ``weights``   : ``(B, T)``
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64) -> None:
        super().__init__()

        # Öğrenilebilir hizalama ağırlıkları
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self, lstm_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """İleri geçiş.

        Parameters
        ----------
        lstm_outputs : Tensor, shape ``(B, T, H)``
            BiLSTM'in tüm zaman adımlarındaki çıktıları.

        Returns
        -------
        context : Tensor, shape ``(B, H)``
            Ağırlıklı toplam bağlam vektörü.
        attention_weights : Tensor, shape ``(B, T)``
            Her zaman adımının dikkat ağırlığı (toplamı 1).
        """
        # score = V^T * tanh(W * h)   →   (B, T, 1)
        energy = torch.tanh(self.W(lstm_outputs))   # (B, T, attn_dim)
        score = self.V(energy).squeeze(-1)           # (B, T)

        # Softmax ile normalleştir → dikkat ağırlıkları
        attention_weights = F.softmax(score, dim=-1)  # (B, T)

        # Bağlam vektörü = ağırlıklı toplam
        # (B, T, 1) * (B, T, H) → sum → (B, H)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, T)
            lstm_outputs,                    # (B, T, H)
        ).squeeze(1)                         # (B, H)

        return context, attention_weights


# ======================================================================
# Sınıf 3 — ECGHybridModel
# ======================================================================

class ECGHybridModel(nn.Module):
    """Hibrit EKG Aritmi Sınıflandırma Modeli.

    Mimari aşamaları:

    1. **MultiScaleCNN1D** — Uzamsal / morfolojik örüntü çıkarımı
    2. **BiLSTM**          — Zamansal bağlam (çift yönlü)
    3. **BahdanauAttention** — Kritik segmentlere odaklanma
    4. **Tam Bağlantılı (FC)** — Sınıflandırma kafası

    ::

        (B,1,L) → CNN → (B,96,L') → permute → BiLSTM → Attention → FC → (B,C)

    Parameters
    ----------
    in_channels : int, default=1
        Girdi EKG kanal sayısı.
    num_classes : int, default=5
        Çıktı sınıf sayısı (N, LBBB, RBBB, PVC, APC).
    cnn_filters_per_branch : int, default=32
        CNN dalı başına filtre sayısı.
    lstm_hidden : int, default=64
        Tek yönlü LSTM gizli birim sayısı.
        BiLSTM çıktısı ``2 * lstm_hidden`` olur.
    lstm_layers : int, default=2
        LSTM katman sayısı.
    attention_dim : int, default=64
        Dikkat ara katman boyutu.
    dropout : float, default=0.3
        Dropout oranı (CNN + FC katmanlarında).

    Attributes
    ----------
    _last_attention_weights : Tensor | None
        Son ileri geçişin dikkat ağırlıkları.  ``get_attention_weights()``
        ile erişilebilir.  Grad-CAM / XAI görselleştirmelerinde kullanılır.

    Examples
    --------
    >>> device = get_device()
    >>> model = ECGHybridModel(num_classes=5).to(device)
    >>> x = torch.randn(16, 1, 500).to(device)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([16, 5])
    """

    # MIT-BIH sınıf etiketleri — referans
    CLASS_NAMES: List[str] = ["Normal", "LBBB", "RBBB", "PVC", "APC", "AF", "MI"]

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        cnn_filters_per_branch: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Aşama 1: Çoklu ölçekli CNN ─────────────────────────────
        self.cnn = MultiScaleCNN1D(
            in_channels=in_channels,
            out_channels_per_branch=cnn_filters_per_branch,
            dropout=dropout,
        )

        cnn_out_channels = self.cnn.total_out_channels  # 3 * 32 = 96

        # ── Aşama 2: BiLSTM ────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,       # Her zaman adımında 96 öznitelik
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = 2 * lstm_hidden  # Bidirectional → çıktı boyutu iki katı

        # ── Aşama 3: Bahdanau Attention ─────────────────────────────
        self.attention = BahdanauAttention(
            hidden_dim=lstm_out_dim,
            attention_dim=attention_dim,
        )

        # ── Aşama 4: Tam bağlantılı sınıflandırıcı ─────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        # Dikkat ağırlıklarını sakla (XAI / Grad-CAM için)
        self._last_attention_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş.

        Veri akışı::

            x (B, 1, L)
            → CNN    → (B, 96, L//2)
            → permute → (B, L//2, 96)   ← LSTM sekans formatı
            → BiLSTM → (B, L//2, 128)
            → Attention → context (B, 128)
            → FC → logits (B, num_classes)

        Parameters
        ----------
        x : Tensor, shape ``(B, 1, seq_len)``
            Tek kanallı EKG segmenti batch'i.

        Returns
        -------
        Tensor, shape ``(B, num_classes)``
            Sınıf logit'leri (softmax uygulanmamış).
        """
        # Aşama 1 — CNN
        cnn_out = self.cnn(x)  # (B, 96, L')

        # Boyut dönüşümü: (B, C, L') → (B, L', C)  ← LSTM beklentisi
        lstm_in = cnn_out.permute(0, 2, 1)  # (B, L', 96)

        # Aşama 2 — BiLSTM
        lstm_out, _ = self.lstm(lstm_in)  # (B, L', 2*hidden)

        # Aşama 3 — Attention
        context, attn_weights = self.attention(lstm_out)  # (B, 2*hidden), (B, L')
        self._last_attention_weights = attn_weights.detach()

        # Aşama 4 — Sınıflandırıcı
        logits = self.classifier(context)  # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Son ileri geçişin dikkat ağırlıklarını döndürür.

        Returns
        -------
        Tensor | None
            Shape ``(B, T)`` dikkat ağırlıkları.  Henüz ileri geçiş
            yapılmadıysa ``None`` döner.

        Notes
        -----
        Bu ağırlıklar Grad-CAM / SHAP tarzı XAI görselleştirmeleri
        oluşturmak için kullanılabilir.
        """
        return self._last_attention_weights

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Model mimarisinin okunabilir özetini döndürür.

        Returns
        -------
        str
            Katman listesi ve toplam parametre sayısı.
        """
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("ECGHybridModel — Mimari Özeti")
        lines.append("=" * 70)

        total_params = 0
        trainable_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            lines.append(
                f"  {name:45s}  shape={str(list(param.shape)):20s}  "
                f"params={param.numel():>8,}"
            )

        lines.append("-" * 70)
        lines.append(f"  Toplam parametre       : {total_params:>12,}")
        lines.append(f"  Eğitilebilir parametre : {trainable_params:>12,}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ======================================================================
# Sınıf 4 — XGBoostClassifier
# ======================================================================

class XGBoostClassifier:
    """XGBoost tabanlı tablo sınıflandırıcı sarmalayıcısı.

    HRV (Kalp Hızı Değişkenliği) ve istatistiksel özniteliklerden
    sınıflandırma yapar.  ``xgboost.XGBClassifier`` üzerine
    yüksek seviyeli bir API sağlar.

    Parameters
    ----------
    n_estimators : int, default=200
        Boosting tur sayısı (ağaç sayısı).
    max_depth : int, default=6
        Her ağacın maksimum derinliği.
    learning_rate : float, default=0.1
        Shrinkage oranı (öğrenme hızı).
    num_classes : int, default=5
        Çıktı sınıf sayısı.
    random_state : int, default=42
        Tekrarlanabilirlik için rastgele tohum.
    **kwargs
        ``XGBClassifier``'a aktarılacak ek parametreler.

    Raises
    ------
    ImportError
        ``xgboost`` paketi yüklü değilse.

    Examples
    --------
    >>> clf = XGBoostClassifier()
    >>> clf.train(X_train, y_train)
    >>> preds = clf.predict(X_test)
    >>> probs = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_classes: int = 7,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError(
                "XGBoost kurulu değil.  Lütfen `pip install xgboost` ile yükleyin."
            )

        self.num_classes = num_classes

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=num_classes,
            random_state=random_state,
            verbosity=0,
            **kwargs,
        )

        self._is_fitted: bool = False
        logger.info(
            "XGBoostClassifier oluşturuldu — n_estimators=%d, max_depth=%d, lr=%.4f",
            n_estimators,
            max_depth,
            learning_rate,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> "XGBoostClassifier":
        """Modeli eğitir.

        Parameters
        ----------
        X : ndarray, shape ``(n_samples, n_features)``
            Öznitelik matrisi.
        y : ndarray, shape ``(n_samples,)``
            Hedef etiketler (0-indexed tamsayılar).
        eval_set : list of (X, y), optional
            Doğrulama kümeleri.
        verbose : bool, default=False
            Eğitim loglarını yazdır.

        Returns
        -------
        self
            Zincirleme çağrı (fluent API) için.
        """
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        self._is_fitted = True
        logger.info("XGBoost eğitimi tamamlandı — %d örnek, %d öznitelik.", *X.shape)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sınıf tahminleri döndürür.

        Parameters
        ----------
        X : ndarray, shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray, shape ``(n_samples,)``
            Tahmin edilen sınıf etiketleri.
        """
        self._check_fitted()
        return self.model.predict(X)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sınıf olasılıkları döndürür.

        Parameters
        ----------
        X : ndarray, shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray, shape ``(n_samples, num_classes)``
            Her sınıfın olasılığı.
        """
        self._check_fitted()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def get_feature_importance(
        self, importance_type: str = "weight"
    ) -> Dict[str, float]:
        """Öznitelik önem skorlarını döndürür.

        Parameters
        ----------
        importance_type : str, default="weight"
            ``'weight'`` (kaç kez kullanıldı), ``'gain'`` (ortalama kazanç),
            veya ``'cover'`` (ortalama kapsam).

        Returns
        -------
        dict[str, float]
            Öznitelik adı → önem skoru eşlemesi.

        Notes
        -----
        Dönen sözlük SHAP değerleri hesaplamak için öncül bilgi olarak
        kullanılabilir.
        """
        self._check_fitted()
        return self.model.get_booster().get_score(importance_type=importance_type)

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Model eğitilmemişse ``RuntimeError`` fırlatır."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model henüz eğitilmedi.  Önce `train(X, y)` çağrısı yapın."
            )
