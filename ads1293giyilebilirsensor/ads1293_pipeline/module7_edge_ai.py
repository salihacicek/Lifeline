# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 7: Gömülü Yapay Zeka (Edge AI) ve Donanımsal Hızlandırma
=============================================================================
Bu modül, eğitilmiş derin öğrenme modellerinin gömülü sistemlere (STM32,
Raspberry Pi, Jetson Nano vb.) deploy edilmesi için gerekli optimizasyon
ve dönüşüm adımlarını gerçekleştirir:

  1. ModelQuantizer  — Quantization-Aware Training (QAT) ve Post-Training
                       Quantization (PTQ).  Float32 ağırlıkları Int8'e
                       indirgeyerek ~4× model boyutu küçültme.
  2. ModelPruner     — Yapısal (structured) ve yapısal-olmayan (unstructured)
                       ağırlık budaması.  Sıfıra yakın önemsiz bağlantıları
                       kaldırarak inference süresini kısaltma.
  3. ONNXExporter    — PyTorch modelinin ONNX formatına aktarımı ve
                       doğrulanması.  C/C++ tabanlı çıkarım motorlarında
                       (ONNX Runtime, TensorRT, OpenVINO) kullanım için.
  4. EdgeAIPipeline  — Orkestratör: Quantization → Pruning → ONNX Export
                       adımlarını sıralı biçimde yürütür ve her aşamanın
                       karşılaştırmalı ölçüm raporunu üretir.

Referanslar
-----------
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- PyTorch Pruning:      https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- ONNX:                 https://onnx.ai/
- ONNX Runtime:         https://onnxruntime.ai/

Notlar
------
* QAT (Quantization-Aware Training) yalnızca CPU üzerinde çalışır.
  MPS / CUDA cihazlarında model otomatik olarak CPU'ya transfer edilir.
* ``onnx`` ve ``onnxruntime`` isteğe bağlı bağımlılıklardır; yüklü
  değilse ilgili fonksiyonlar uyarı verir ve ``None`` döndürür.
=============================================================================
"""

import os
import time
import copy
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.quantization as tq
from torch.nn.utils import prune

# ---------------------------------------------------------------------------
#  İsteğe bağlı bağımlılıklar — yüklü değilse ``None`` olarak kalır
# ---------------------------------------------------------------------------
try:
    import onnx
    import onnx.checker
except ImportError:
    onnx = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


# ===========================================================================
#  Yardımcı Fonksiyonlar
# ===========================================================================

def _get_model_size_mb(model: nn.Module) -> float:
    """
    Modelin disk üzerindeki boyutunu megabayt cinsinden hesaplar.

    Geçici bir dosyaya ``state_dict`` kaydedilir, boyut okunur ve
    dosya silinir.

    Parameters
    ----------
    model : nn.Module
        Boyutu ölçülecek PyTorch modeli.

    Returns
    -------
    float
        Model boyutu (MB).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_bytes = os.path.getsize(tmp.name)
    os.unlink(tmp.name)
    return size_bytes / (1024.0 * 1024.0)


def _benchmark_torch_inference(
    model: nn.Module,
    sample_input: torch.Tensor,
    n_runs: int = 100,
) -> float:
    """
    PyTorch modelinin ortalama çıkarım (inference) süresini ölçer.

    Parameters
    ----------
    model : nn.Module
        Ölçüm yapılacak model.
    sample_input : torch.Tensor
        Tek bir örnek giriş tensörü.
    n_runs : int
        Tekrar sayısı (varsayılan 100).

    Returns
    -------
    float
        Ortalama çıkarım süresi (milisaniye).
    """
    model.eval()
    # Isınma (warm-up) — ilk çalıştırmalar JIT derleme nedeniyle yavaş
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return float(sum(times) / len(times))


# ===========================================================================
#  Class 1: ModelQuantizer
# ===========================================================================

class ModelQuantizer:
    """
    Quantization-Aware Training (QAT) ve Post-Training Quantization (PTQ)
    yöneticisi.

    Float32 ağırlıkları Int8 formatına indirger ve ~4× model boyutu
    küçültme sağlar.  QAT yalnızca CPU üzerinde çalışır; modelin MPS /
    CUDA'daki kopyası otomatik olarak CPU'ya taşınır.

    Kullanım
    --------
    >>> quantizer = ModelQuantizer()
    >>> qat_model = quantizer.prepare_qat(model)
    >>> # ... fine-tune qat_model on CPU ...
    >>> q_model = quantizer.convert_to_quantized(qat_model)
    >>> report = quantizer.compare_sizes(model, q_model)

    Attributes
    ----------
    qconfig : torch.quantization.QConfig
        Kullanılacak kuantizasyon yapılandırması.
    """

    def __init__(self, qconfig: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        qconfig : torch.quantization.QConfig, optional
            Özel kuantizasyon yapılandırması.  ``None`` ise
            ``torch.quantization.get_default_qat_qconfig('x86')``
            kullanılır.
        """
        if qconfig is not None:
            self.qconfig = qconfig
        else:
            # x86 backend varsayılan QAT yapılandırması
            try:
                self.qconfig = tq.get_default_qat_qconfig("x86")
            except Exception:
                # Eski PyTorch sürümleri için geri dönüş
                self.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # -----------------------------------------------------------------
    def prepare_qat(self, model: nn.Module) -> nn.Module:
        """
        Modeli Quantization-Aware Training (QAT) için hazırlar.

        ``FakeQuantize`` katmanları eklenir; eğitim sırasında bu katmanlar
        kuantizasyon gürültüsünü simüle eder ve model buna karşı dayanıklı
        hale gelir.

        Parameters
        ----------
        model : nn.Module
            Hazırlanacak orijinal (Float32) model.

        Returns
        -------
        nn.Module
            QAT için hazırlanmış model (CPU üzerinde).
        """
        model_cpu = copy.deepcopy(model).cpu()
        model_cpu.train()

        # Global qconfig atanması
        model_cpu.qconfig = self.qconfig  # type: ignore[assignment]

        # Fuse eligible modules (Conv-BN-ReLU vb.) — başarısız olursa atla
        try:
            model_cpu = tq.fuse_modules(model_cpu, self._find_fusable_modules(model_cpu))
        except Exception:
            # Fuse edilecek uygun katman bulunamazsa devam et
            pass

        # QAT hazırlığı: FakeQuantize düğümleri eklenir
        tq.prepare_qat(model_cpu, inplace=True)
        return model_cpu

    # -----------------------------------------------------------------
    def convert_to_quantized(self, model: nn.Module) -> nn.Module:
        """
        QAT ile hazırlanmış (veya normal) modeli tam kuantize Int8 modele
        dönüştürür (Post-Training Quantization).

        Parameters
        ----------
        model : nn.Module
            ``prepare_qat`` ile hazırlanmış veya Float32 modeli.

        Returns
        -------
        nn.Module
            Int8 kuantize model.
        """
        model_cpu = copy.deepcopy(model).cpu()
        model_cpu.eval()

        # Eğer qconfig yoksa atanır (PTQ senaryosu)
        if not hasattr(model_cpu, "qconfig") or model_cpu.qconfig is None:
            model_cpu.qconfig = tq.get_default_qconfig("x86")  # type: ignore[assignment]
            tq.prepare(model_cpu, inplace=True)
            # Kalibrasyon: tek bir sıfır tensörü ile
            with torch.no_grad():
                try:
                    dummy = torch.zeros(1, 1, 500)  # (B, C, L) varsayılan
                    model_cpu(dummy)
                except Exception:
                    pass

        quantized = tq.convert(model_cpu, inplace=False)
        return quantized

    # -----------------------------------------------------------------
    def compare_sizes(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
    ) -> Dict[str, float]:
        """
        Orijinal ve kuantize modelin boyutlarını karşılaştırır.

        Parameters
        ----------
        original_model : nn.Module
            Orijinal (Float32) model.
        quantized_model : nn.Module
            Kuantize (Int8) model.

        Returns
        -------
        dict
            ``original_size_mb``, ``quantized_size_mb``,
            ``compression_ratio``, ``size_reduction_percent`` anahtarlarını
            içeren sözlük.
        """
        orig_sz = _get_model_size_mb(original_model)
        q_sz = _get_model_size_mb(quantized_model)
        ratio = orig_sz / q_sz if q_sz > 0 else float("inf")
        reduction = (1.0 - q_sz / orig_sz) * 100.0 if orig_sz > 0 else 0.0

        return {
            "original_size_mb": round(orig_sz, 4),
            "quantized_size_mb": round(q_sz, 4),
            "compression_ratio": round(ratio, 2),
            "size_reduction_percent": round(reduction, 2),
        }

    # -----------------------------------------------------------------
    def benchmark_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        n_runs: int = 100,
    ) -> float:
        """
        Modelin ortalama çıkarım süresini ölçer.

        Parameters
        ----------
        model : nn.Module
            Ölçüm yapılacak model.
        sample_input : torch.Tensor
            Tek örnek giriş tensörü.
        n_runs : int
            Tekrar sayısı.

        Returns
        -------
        float
            Ortalama çıkarım süresi (milisaniye).
        """
        model_cpu = copy.deepcopy(model).cpu()
        inp_cpu = sample_input.cpu()
        return _benchmark_torch_inference(model_cpu, inp_cpu, n_runs)

    # -----------------------------------------------------------------
    #  Dahili yardımcı
    # -----------------------------------------------------------------
    @staticmethod
    def _find_fusable_modules(model: nn.Module) -> list:
        """
        Modeldeki Conv-BN-ReLU veya Linear-ReLU gibi birleştirilebilir
        (fusable) katman gruplarını bulur.

        Returns
        -------
        list[list[str]]
            Birleştirilebilir modül isim listeleri.
        """
        fusable: list[list[str]] = []
        named = dict(model.named_modules())
        names = list(named.keys())

        for i, name in enumerate(names):
            mod = named[name]
            if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                group = [name]
                # BatchNorm takip ediyor mu?
                if i + 1 < len(names):
                    next_mod = named[names[i + 1]]
                    if isinstance(next_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        group.append(names[i + 1])
                        # ReLU takip ediyor mu?
                        if i + 2 < len(names):
                            nn_mod = named[names[i + 2]]
                            if isinstance(nn_mod, nn.ReLU):
                                group.append(names[i + 2])
                if len(group) > 1:
                    fusable.append(group)

        return fusable


# ===========================================================================
#  Class 2: ModelPruner
# ===========================================================================

class ModelPruner:
    """
    Ağırlık Budama (Weight Pruning) yöneticisi.

    Sıfıra yakın (önemsiz) ağırlık bağlantılarını kaldırarak modelin
    seyrekliğini (sparsity) artırır.  Hem yapısal (structured) hem de
    yapısal-olmayan (unstructured) budama desteklenir.

    Kullanım
    --------
    >>> pruner = ModelPruner()
    >>> pruned = pruner.apply_unstructured_pruning(model, amount=0.3)
    >>> report = pruner.get_sparsity_report(pruned)
    >>> pruner.remove_pruning_reparametrization(pruned)

    Notes
    -----
    * ``amount=0.3`` parametresi, ağırlıkların %30'unun budanacağı
      anlamına gelir.
    * ``remove_pruning_reparametrization`` çağrılmadan model hâlâ
      ``weight_orig`` + ``weight_mask`` ikili yapısı taşır.
    """

    # -----------------------------------------------------------------
    def apply_structured_pruning(
        self,
        model: nn.Module,
        amount: float = 0.3,
    ) -> nn.Module:
        """
        Yapısal budama uygular (kanal bazlı L1-norm).

        Tüm Conv1d/Conv2d katmanlarına ``ln_structured`` budama
        (dim=0, n=1 → L1-norm) uygulanır.

        Parameters
        ----------
        model : nn.Module
            Budanacak model.
        amount : float
            Budanacak kanal oranı (0–1).

        Returns
        -------
        nn.Module
            Budanmış model (yerinde değiştirilir).
        """
        model = copy.deepcopy(model)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                prune.ln_structured(
                    module, name="weight", amount=amount, n=1, dim=0
                )
            elif isinstance(module, nn.Linear):
                prune.ln_structured(
                    module, name="weight", amount=amount, n=1, dim=0
                )
        return model

    # -----------------------------------------------------------------
    def apply_unstructured_pruning(
        self,
        model: nn.Module,
        amount: float = 0.3,
    ) -> nn.Module:
        """
        Yapısal-olmayan (unstructured) budama uygular (L1-norm).

        Tek tek ağırlık elemanları sıfıra yakınlığına göre maskelenir.

        Parameters
        ----------
        model : nn.Module
            Budanacak model.
        amount : float
            Budanacak ağırlık oranı (0–1).

        Returns
        -------
        nn.Module
            Budanmış model (derin kopya üzerinde).
        """
        model = copy.deepcopy(model)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=amount)
        return model

    # -----------------------------------------------------------------
    def get_sparsity_report(self, model: nn.Module) -> Dict[str, Any]:
        """
        Modelin katman bazlı ve genel seyreklik raporunu üretir.

        Parameters
        ----------
        model : nn.Module
            Seyreklik analizi yapılacak model.

        Returns
        -------
        dict
            ``layer_sparsity`` (katman bazlı seyreklik yüzdeleri),
            ``total_zeros``, ``total_params``, ``global_sparsity_percent``
            anahtarlarını içerir.
        """
        layer_report: Dict[str, float] = {}
        total_zeros = 0
        total_params = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight"):
                    w = module.weight.data
                    n_zeros = int(torch.sum(w == 0).item())
                    n_total = int(w.nelement())
                    sparsity = 100.0 * n_zeros / n_total if n_total > 0 else 0.0
                    layer_report[name] = round(sparsity, 2)
                    total_zeros += n_zeros
                    total_params += n_total

        global_sparsity = (
            100.0 * total_zeros / total_params if total_params > 0 else 0.0
        )
        return {
            "layer_sparsity": layer_report,
            "total_zeros": total_zeros,
            "total_params": total_params,
            "global_sparsity_percent": round(global_sparsity, 2),
        }

    # -----------------------------------------------------------------
    def remove_pruning_reparametrization(self, model: nn.Module) -> nn.Module:
        """
        Budama yeniden parametrelendirmesini kaldırarak ağırlıkları
        kalıcı hale getirir.

        ``weight_orig`` + ``weight_mask`` → ``weight`` olarak birleştirilir.

        Parameters
        ----------
        model : nn.Module
            Budanmış model.

        Returns
        -------
        nn.Module
            Kalıcı budaması uygulanmış model.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    # Budama uygulanmamış katman — atla
                    pass
        return model


# ===========================================================================
#  Class 3: ONNXExporter
# ===========================================================================

class ONNXExporter:
    """
    PyTorch modelini ONNX formatına aktarır, doğrular ve benchmark yapar.

    ONNX formatı, C/C++ tabanlı çıkarım motorlarında (ONNX Runtime,
    TensorRT, OpenVINO) kullanım için standarttır.

    Kullanım
    --------
    >>> exporter = ONNXExporter()
    >>> path = exporter.export(model, sample_input, "model.onnx")
    >>> valid = exporter.validate_onnx(path)
    >>> avg_ms = exporter.benchmark_onnx(path, sample_input)

    Notes
    -----
    ``onnx`` ve ``onnxruntime`` paketleri yüklü değilse ilgili metotlar
    uyarı verir ve ``None`` döndürür.
    """

    # -----------------------------------------------------------------
    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        opset_version: int = 13,
    ) -> Optional[str]:
        """
        Modeli ONNX formatında dışa aktarır.

        Parameters
        ----------
        model : nn.Module
            Dışa aktarılacak model.
        sample_input : torch.Tensor
            Modelin beklediği formatta örnek giriş tensörü.
        output_path : str
            ONNX dosyasının kaydedileceği yol.
        opset_version : int
            ONNX opset sürümü (varsayılan 13).

        Returns
        -------
        str or None
            Başarılı ise dosya yolu, aksi halde ``None``.
        """
        model_cpu = copy.deepcopy(model).cpu().eval()
        inp_cpu = sample_input.cpu()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            torch.onnx.export(
                model_cpu,
                inp_cpu,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["ecg_input"],
                output_names=["prediction"],
                dynamic_axes={
                    "ecg_input": {0: "batch_size"},
                    "prediction": {0: "batch_size"},
                },
            )
            return output_path
        except Exception as exc:
            warnings.warn(f"ONNX export başarısız: {exc}")
            return None

    # -----------------------------------------------------------------
    def validate_onnx(self, onnx_path: str) -> bool:
        """
        ONNX dosyasını ``onnx.checker`` ile doğrular.

        Parameters
        ----------
        onnx_path : str
            Doğrulanacak ONNX dosyasının yolu.

        Returns
        -------
        bool
            Geçerli ise ``True``, aksi halde ``False``.
        """
        if onnx is None:
            warnings.warn(
                "onnx paketi yüklü değil. `pip install onnx` ile yükleyin."
            )
            return False

        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            return True
        except Exception as exc:
            warnings.warn(f"ONNX doğrulama hatası: {exc}")
            return False

    # -----------------------------------------------------------------
    def benchmark_onnx(
        self,
        onnx_path: str,
        sample_input: torch.Tensor,
        n_runs: int = 100,
    ) -> Optional[float]:
        """
        ONNX Runtime kullanarak modelin ortalama çıkarım süresini ölçer.

        Parameters
        ----------
        onnx_path : str
            ONNX dosya yolu.
        sample_input : torch.Tensor
            Örnek giriş tensörü.
        n_runs : int
            Tekrar sayısı.

        Returns
        -------
        float or None
            Ortalama çıkarım süresi (ms) veya ort yoksa ``None``.
        """
        if ort is None:
            warnings.warn(
                "onnxruntime yüklü değil. `pip install onnxruntime` ile yükleyin."
            )
            return None

        try:
            session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )
            input_name = session.get_inputs()[0].name
            np_input = sample_input.cpu().numpy()

            # Isınma
            for _ in range(5):
                session.run(None, {input_name: np_input})

            times: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                session.run(None, {input_name: np_input})
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            return float(sum(times) / len(times))

        except Exception as exc:
            warnings.warn(f"ONNX Runtime benchmark hatası: {exc}")
            return None

    # -----------------------------------------------------------------
    def get_model_info(self, onnx_path: str) -> Optional[Dict[str, Any]]:
        """
        ONNX modelinin giriş/çıkış şekillerini ve dosya boyutunu döndürür.

        Parameters
        ----------
        onnx_path : str
            ONNX dosya yolu.

        Returns
        -------
        dict or None
            ``inputs``, ``outputs``, ``size_mb`` anahtarlarını içerir.
        """
        if onnx is None:
            warnings.warn("onnx paketi yüklü değil.")
            return None

        try:
            model = onnx.load(onnx_path)
            inputs_info = []
            for inp in model.graph.input:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    shape.append(
                        dim.dim_value if dim.dim_value > 0 else "dynamic"
                    )
                inputs_info.append({"name": inp.name, "shape": shape})

            outputs_info = []
            for out in model.graph.output:
                shape = []
                for dim in out.type.tensor_type.shape.dim:
                    shape.append(
                        dim.dim_value if dim.dim_value > 0 else "dynamic"
                    )
                outputs_info.append({"name": out.name, "shape": shape})

            size_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0)

            return {
                "inputs": inputs_info,
                "outputs": outputs_info,
                "size_mb": round(size_mb, 4),
                "opset_version": model.opset_import[0].version,
            }

        except Exception as exc:
            warnings.warn(f"ONNX bilgi alma hatası: {exc}")
            return None


# ===========================================================================
#  Class 4: EdgeAIPipeline
# ===========================================================================

class EdgeAIPipeline:
    """
    Edge AI optimizasyon orkestratörü.

    Sıralı adımlar:
      1. Orijinal model boyutu ve çıkarım süresi ölçümü
      2. Quantization (PTQ)
      3. Pruning (Unstructured, %30)
      4. ONNX Export + Doğrulama + Benchmark
      5. Karşılaştırmalı rapor üretimi

    Kullanım
    --------
    >>> pipeline = EdgeAIPipeline()
    >>> results = pipeline.run(model, sample_input, output_dir="./edge_ai_output")

    Returns
    -------
    dict
        Her adımın ölçüm sonuçlarını içeren kapsamlı rapor sözlüğü.
    """

    def __init__(self) -> None:
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner()
        self.exporter = ONNXExporter()

    # -----------------------------------------------------------------
    def run(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: str = "./edge_ai_output",
    ) -> Dict[str, Any]:
        """
        Tam Edge AI optimizasyon hattını çalıştırır.

        Parameters
        ----------
        model : nn.Module
            Orijinal eğitilmiş model.
        sample_input : torch.Tensor
            Modelin beklediği formatta örnek giriş.
        output_dir : str
            Çıktı dosyalarının kaydedileceği dizin.

        Returns
        -------
        dict
            ``original``, ``quantized``, ``pruned``, ``onnx``,
            ``comparison`` alt-sözlüklerini içeren rapor.
        """
        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, Any] = {}

        # ----- 1) Orijinal model ölçümleri ----------------------------------
        orig_size = _get_model_size_mb(model)
        orig_time = _benchmark_torch_inference(
            copy.deepcopy(model).cpu(),
            sample_input.cpu(),
            n_runs=50,
        )
        results["original"] = {
            "size_mb": round(orig_size, 4),
            "avg_inference_ms": round(orig_time, 4),
        }

        # ----- 2) Quantization (PTQ) ----------------------------------------
        try:
            q_model = self.quantizer.convert_to_quantized(model)
            q_size = _get_model_size_mb(q_model)
            q_time = _benchmark_torch_inference(
                q_model, sample_input.cpu(), n_runs=50
            )
            results["quantized"] = {
                "size_mb": round(q_size, 4),
                "avg_inference_ms": round(q_time, 4),
                "compression_ratio": round(orig_size / q_size, 2) if q_size > 0 else None,
            }
        except Exception as exc:
            results["quantized"] = {"error": str(exc)}
            q_model = None

        # ----- 3) Pruning (Unstructured %30) ---------------------------------
        pruned_model = self.pruner.apply_unstructured_pruning(model, amount=0.3)
        sparsity = self.pruner.get_sparsity_report(pruned_model)
        self.pruner.remove_pruning_reparametrization(pruned_model)
        pruned_size = _get_model_size_mb(pruned_model)
        pruned_time = _benchmark_torch_inference(
            copy.deepcopy(pruned_model).cpu(),
            sample_input.cpu(),
            n_runs=50,
        )
        results["pruned"] = {
            "size_mb": round(pruned_size, 4),
            "avg_inference_ms": round(pruned_time, 4),
            "sparsity": sparsity,
        }

        # ----- 4) ONNX Export ------------------------------------------------
        onnx_path = os.path.join(output_dir, "model_optimized.onnx")
        export_result = self.exporter.export(
            pruned_model, sample_input, onnx_path
        )
        if export_result is not None:
            valid = self.exporter.validate_onnx(onnx_path)
            onnx_time = self.exporter.benchmark_onnx(
                onnx_path, sample_input, n_runs=50
            )
            onnx_info = self.exporter.get_model_info(onnx_path)
            results["onnx"] = {
                "path": onnx_path,
                "valid": valid,
                "avg_inference_ms": onnx_time,
                "info": onnx_info,
            }
        else:
            results["onnx"] = {"error": "ONNX export başarısız."}

        # ----- 5) Karşılaştırmalı özet --------------------------------------
        results["comparison"] = {
            "original_size_mb": results["original"]["size_mb"],
            "quantized_size_mb": results.get("quantized", {}).get("size_mb"),
            "pruned_size_mb": results["pruned"]["size_mb"],
            "original_inference_ms": results["original"]["avg_inference_ms"],
            "pruned_inference_ms": results["pruned"]["avg_inference_ms"],
        }

        return results
