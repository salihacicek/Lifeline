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

# Modüllerden importlar
from module5_deep_learning import ECGHybridModel
from module7_edge_ai import ModelQuantizer, ModelPruner, ONNXExporter, EdgeAIPipeline
from module8_xai import ClinicalMetrics, YoudenOptimizer, GradCAMExplainer

def main():
    print("=" * 70)
    print("=== Modül 7-8 Entegre Testi Başlatılıyor ===")
    print("=" * 70)
    
    # 1. Test Modeli Oluşturma
    print("\n1. Test Modeli Oluşturuluyor...")
    # Basit test modeli: (Batch, 1, 288) -> (Batch, 2) (Binary Classification)
    model = ECGHybridModel(in_channels=1, num_classes=2)
    model.eval() # Test moduna al
    
    sample_input = torch.randn(1, 1, 288, dtype=torch.float32)
    
    # 2. Modül 7: Edge AI Testleri
    print("\n2. Modül 7 (Edge AI & Gömülü Donanım Optimizasyonu) Testleri...")
    
    # A. Quantization (QAT) Testi
    try:
        print("  -> Quantization test ediliyor (CPU'da çalışır)...")
        quantizer = ModelQuantizer()
        qat_model = quantizer.prepare_qat(model)
        # 1-2 dummy pass for QAT observation
        _ = qat_model(sample_input)
        quantized_model = quantizer.convert_to_quantized(qat_model)
        size_comp = quantizer.compare_sizes(model, quantized_model)
        print(f"     Orijinal Boyut: {size_comp['original_size_mb']:.2f} MB")
        print(f"     Quantized Boyut: {size_comp['quantized_size_mb']:.2f} MB")
        print(f"     Küçülme Oranı: {size_comp['reduction_ratio']:.2f}x")
    except Exception as e:
        print(f"  -> Quantization Testi Hata: {e}")
        
    # B. Pruning (Budama) Testi
    try:
        print("\n  -> Ağırlık Budaması (Pruning) test ediliyor...")
        pruner = ModelPruner()
        # %30'luk budama
        pruned_model = pruner.apply_unstructured_pruning(model, amount=0.3)
        report = pruner.get_sparsity_report(pruned_model)
        print(f"     Toplam Sparsity: %{report.get('total_sparsity_percent', 0):.2f}")
    except Exception as e:
        print(f"  -> Pruning Testi Hata: {e}")
        
    # C. ONNX Export Testi
    try:
        print("\n  -> ONNX Dışa Aktarma test ediliyor...")
        exporter = ONNXExporter()
        onnx_path = exporter.export(model, sample_input, "test_model.onnx", opset_version=13)
        valid = exporter.validate_onnx(onnx_path)
        print(f"     ONNX Export Başarılı: {valid} (Dosya: {onnx_path})")
    except Exception as e:
        print(f"  -> ONNX Export Testi Uyarı/Hata (Kütüphane eksik olabilir): {e}")

    # 3. Modül 8: XAI ve Klinik Metrik Testleri
    print("\n3. Modül 8 (Klinik XAI) Testleri...")
    
    # Sentetik gerçek etiketler (0: Normal, 1: Hasta) ve modelin ürettiği olasılıklar
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.8, 0.2, 0.6, 0.9, 0.3, 0.1, 0.4, 0.85]) 
    # Not: 8. indeksteki hasta 0.4 prob almış (Missed Disease - False Negative for thr=0.5)
    
    # A. Klinik Metrikler
    print("\n  -> Klinik Karar Metrikleri (Threshold: 0.5):")
    y_pred_default = (y_prob >= 0.5).astype(int)
    metrics = ClinicalMetrics.compute_all_metrics(y_true, y_pred_default, y_prob)
    print(f"     AUPRC       : {metrics.get('auprc', 0):.3f}")
    print(f"     Sensitivity : {metrics['sensitivity_recall']:.3f}")
    print(f"     Specificity : {metrics['specificity']:.3f}")
    print(f"     F1-Macro    : {metrics['f1_macro']:.3f}")
    
    # B. Youden Optimizasyonu
    print("\n  -> Youden J Index Optimizasyonu:")
    optimal_thr, j_score = YoudenOptimizer.find_optimal_threshold(y_true, y_prob)
    y_pred_opt = YoudenOptimizer.apply_threshold(y_prob, optimal_thr)
    metrics_opt = ClinicalMetrics.compute_all_metrics(y_true, y_pred_opt, y_prob)
    
    print(f"     Optimum Threshold: {optimal_thr:.3f} (J-Score: {j_score:.3f})")
    print(f"     Yeni Sensitivity : {metrics_opt['sensitivity_recall']:.3f} (Kritik hastalıkları kaçırmamak için arttı!)")
    
    # C. Grad-CAM Testi
    print("\n  -> 1D Grad-CAM Uygulaması...")
    try:
        # CNN1D içerisinde kullanabileceğimiz ilk conv layer'a hedefleme yapıyoruz.
        target_layer = model.stage1.branch_31.conv
        explainer = GradCAMExplainer(model, target_layer)
        
        heatmap = explainer.explain(sample_input, target_class=1)
        print(f"     Grad-CAM Isı Haritası Üretildi! Boyut: {heatmap.shape}")
    except Exception as e:
        print(f"     Grad-CAM Hata: {e}")
        heatmap = np.zeros(288) # Hata olursa görselleştirme patlamasın

    # 4. Görselleştirme Kaydetme
    print("\n4. Grafik Çıktıları Kaydediliyor...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 4.1 Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[0].plot(recall, precision, color='purple', lw=2, label=f'AUPRC = {auc(recall, precision):.3f}')
    axes[0].set_title('AUPRC (Precision-Recall Curve)', fontweight='bold')
    axes[0].set_xlabel('Recall (Sensitivity)')
    axes[0].set_ylabel('Precision (PPV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 4.2 Youden ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC Eğrisi')
    axes[1].scatter(fpr[np.argmax(tpr-fpr)], tpr[np.argmax(tpr-fpr)], color='red', marker='X', s=150,
                    label=f'Optimal J (Thr: {optimal_thr:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[1].set_title('Youden J İndeksi (ROC Üzerinde Karar Sınırı)', fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 4.3 Grad-CAM 1D Sinyal Vurgusu
    sig_np = sample_input.squeeze().numpy()
    axes[2].plot(sig_np, color='black', label='Orijinal EKG Segmenti')
    extent = [0, len(sig_np), min(sig_np) - 0.5, max(sig_np) + 0.5]
    im = axes[2].imshow(heatmap[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.5, extent=extent)
    axes[2].set_title('1D Grad-CAM (Model Nereye Odaklandı?)', fontweight='bold')
    axes[2].set_xlabel('Örnek Sayısı (Time)')
    axes[2].set_ylabel('Genlik')
    fig.colorbar(im, ax=axes[2], label='Attention (Grad-CAM)')
    
    plt.tight_layout()
    plt.savefig('module7_8_pipeline_output.png', dpi=150)
    print("Kayıt tamamlandı: module7_8_pipeline_output.png")

if __name__ == '__main__':
    main()
