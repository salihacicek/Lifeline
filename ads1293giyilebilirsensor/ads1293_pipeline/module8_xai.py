# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 8: Klinik Karar Sınırları ve Açıklanabilir Yapay Zeka (XAI)
=============================================================================
Bu modül, sınıflandırma modellerinin kararlarını klinik açıdan yorumlanabilir
hale getirir. 

1. ClinicalMetrics
   - AUPRC, F1-Macro, Sensitivity, Specificity, PPV, NPV ve Confusion Matrix
     gibi tıbbi teşhis doğruluğu için kritik metrikleri hesaplar.

2. YoudenOptimizer
   - Youden J indeksini (Sensitivity + Specificity - 1) maksimize ederek
     yanlış negatif (missed disease) ihtimalini minimize edecek optimum
     karar eşiğini (threshold) bulur.
     
3. GradCAMExplainer
   - PyTorch 1D-CNN tabanlı derin öğrenme modeli için Gradient-weighted
     Class Activation Mapping (Grad-CAM) uygular. EKG sinyali üzerinde
     modelin hangi bölgeye (örn: T dalgası) odaklandığını vurgular.

4. SHAPExplainer
   - XGBoost için SHAP (SHapley Additive exPlanations) kullanarak, tablo
     verilerindeki özelliklerin (örneğin LF/HF oranı) karar üzerindeki
     etki yüzdesini/katkısını gösterir.
=============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, confusion_matrix,
    roc_curve
)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

class ClinicalMetrics:
    """
    Tıbbi teşhis doğruluğunu ölçmek için sınıflandırma metriklerini hesaplar.
    Aşırı dengesiz veriler için AUROC yerine AUPRC tercih edilmektedir.
    """
    @staticmethod
    def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return auc(recall, precision)

    @staticmethod
    def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average='macro')

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) == 2 else (0,0,0,0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        f1_mac = ClinicalMetrics.compute_f1_macro(y_true, y_pred)
        
        metrics = {
            'sensitivity_recall': sensitivity,
            'specificity': specificity,
            'ppv_precision': ppv,
            'npv': npv,
            'f1_macro': f1_mac
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2: # Sadece binary class için şimdilik AUPRC
            metrics['auprc'] = ClinicalMetrics.compute_auprc(y_true, y_prob)
            
        return metrics

    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'AUPRC = {pr_auc:.3f}')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str = None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
                
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Tahmin Edilen')
        ax.set_ylabel('Gerçek Sınıf')
        ax.set_title('Confusion Matrix', pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class YoudenOptimizer:
    """
    Optimum karar eşiğini (Decision Threshold) belirlemek için Youden's J İndeksini kullanır.
    """
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr # Sensitivity + Specificity - 1
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx], j_scores[optimal_idx]

    @staticmethod
    def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
        return (y_prob >= threshold).astype(int)

    @staticmethod
    def plot_youden_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Eğrisi')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', marker='x', s=100, 
                    label=f'Optimal J (Threshold: {thresholds[optimal_idx]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC ve Youden J İndeksi Optimizasyonu')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class GradCAMExplainer:
    """
    1D-CNN PyTorch modelleri için EKG Sinyali üzerinde Grad-CAM uygulaması.
    Hangi zaman aralığının / morfolojinin kararı tetiklediğini gösterir.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook mekanizmaları
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def explain(self, input_signal: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Girdi (Batch=1) için ısı haritasını (Heatmap) üretir.
        """
        self.model.eval()
        output = self.model(input_signal)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Grad-CAM hesaplama
        gradients = self.gradients.detach().cpu().numpy()[0] # (Channels, Length)
        activations = self.activations.detach().cpu().numpy()[0] # (Channels, Length)
        
        # Her bir kanal için ağırlıklar (Global Average Pooling on Gradients)
        weights = np.mean(gradients, axis=1) # (Channels,)
        
        # Ağırlıklı aktivasyon toplamı
        heatmap = np.zeros(activations.shape[1], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # ReLU uygula (Sadece karara pozitif etkisi olanları al)
        heatmap = np.maximum(heatmap, 0)
        
        # Max-Min Normalizasyon (0-1 aralığına)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        # Giriş boyutuna interpolasyon
        input_len = input_signal.shape[-1]
        heatmap = F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0), 
            size=input_len, 
            mode='linear', 
            align_corners=False
        ).squeeze().numpy()
        
        return heatmap

    def plot_explanation(self, signal: np.ndarray, heatmap: np.ndarray, prediction: int, save_path: str = None):
        """
        Isı haritasını orijinal sinyalin üzerine renk koduyla (overlay) çizer.
        """
        signal_np = signal.squeeze() if isinstance(signal, np.ndarray) else signal.cpu().numpy().squeeze()
        
        fig, ax = plt.subplots(figsize=(10, 3))
        # Sinyal siyah renkli, heatmap arkada renk olarak
        ax.plot(signal_np, color='black', linewidth=1.5, label='EKG Sinyali')
        
        # Heatmap'i arka plana yansıt
        extent = [0, len(signal_np), min(signal_np) - 0.5, max(signal_np) + 0.5]
        im = ax.imshow(heatmap[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.5, extent=extent)
        
        ax.set_title(f'1D Grad-CAM Analizi (Tahmin Edilen Sınıf: {prediction})')
        ax.set_ylabel('Genlik / Z-Score')
        ax.set_xlabel('Örnek (Sample)')
        fig.colorbar(im, ax=ax, orientation='vertical', label='Dikkat (Attention) Yoğunluğu')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class SHAPExplainer:
    """
    XGBoost tabanlı Tablo Verisi/Özellik sınıflandırıcısı için SHAP analizi.
    Karara hangi HRV/Entropy özelliğinin ne kadar etki ettiğini yüzdesel olarak açıklar.
    """
    def __init__(self, model, X_background: np.ndarray):
        self.model = model
        self.X_background = X_background
        
        if _SHAP_AVAILABLE:
            self.explainer = shap.Explainer(self.model, self.X_background)
        else:
            self.explainer = None
            warnings.warn("SHAP kütüphanesi yüklü değil.")

    def explain(self, X_sample: np.ndarray):
        if not _SHAP_AVAILABLE:
            return None
        return self.explainer(X_sample)

    def plot_feature_importance(self, shap_values, save_path: str = None):
        if not _SHAP_AVAILABLE:
            return
        
        plt.figure()
        # shap.plots.bar default davranışı
        shap.summary_plot(shap_values, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Genel)')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class XAIPipeline:
    """
    Clinical Metrics, Youden J Optimizasyonu, Grad-CAM ve SHAP orkestratörü.
    """
    def run(self) -> Dict:
        pass # Test scripti içerisinde modüler sınıflar doğrudan çağrılacak.
