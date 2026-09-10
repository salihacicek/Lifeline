# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 6: Model Eğitimi, Optimizasyon ve Kayıp Fonksiyonları
=============================================================================
Bu modül, ECGHybridModel'in (Modül 5) eğitim süreçlerini yönetir.

1. FocalLoss
   - Sınıf dengesizliğini yönetmek için çapraz entropi (Cross-Entropy) yerine
     kullanılır. Kolay örneklere verilen ağırlığı düşürerek, modelin nadir
     görülen aritmilere odaklanmasını sağlar.

2. ECGTrainer
   - Eğitim (Train) ve Doğrulama (Validation) döngülerini yönetir.
   - AdamW optimizer ve CosineAnnealingWarmRestarts scheduler kullanır.
   - Early Stopping mekanizması ve model checkpointing içerir.
   
3. TransferLearningManager
   - Modelin CNN kısmını dondurarak (freeze) sadece son katmanların
     eğitilmesine imkan tanır. Transfer learning uygulamaları içindir.

4. TrainingPipeline
   - NumPy dizilerini alıp DataLoader'lara dönüştürür ve eğitimin uçtan
     uca tamamlanmasını sağlar.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import copy
from typing import Dict, Optional, Tuple, Union

# Modül 5 import
from ads1293_pipeline.module5_deep_learning import get_device, ECGHybridModel

class FocalLoss(nn.Module):
    """
    Focal Loss - Dengesiz veri setleri için odaklanmış kayıp fonksiyonu.
    """
    def __init__(self, alpha: Optional[Union[float, torch.Tensor]] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Focal Loss hesaplama
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss
                
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ECGTrainer:
    """
    ECGHybridModel için PyTorch eğitim döngüsü (Training Loop).
    """
    def __init__(self, model: nn.Module, device: torch.device, lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 10):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        self.criterion = FocalLoss(gamma=2.0)
        self.patience = patience
        
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50) -> Dict:
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Eğitim aşaması
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
                
            self.scheduler.step()
            train_loss = train_loss / total
            train_acc = correct / total
            
            # Doğrulama aşaması
            val_loss, val_acc = self._evaluate_loader(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping tetiklendi! Epoch: {epoch+1}")
                    break
                    
        # En iyi modeli geri yükle
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return history
        
    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
                
        return val_loss / total, correct / total
        
    def evaluate(self, test_loader: DataLoader) -> Dict:
        loss, acc = self._evaluate_loader(test_loader)
        return {'test_loss': loss, 'test_acc': acc}
        
    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model kaydedildi: {path}")
        
    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f"Model yüklendi: {path}")

class TransferLearningManager:
    """
    Önceden eğitilmiş modellerin (pre-trained) yeni görevlere adaptasyonu.
    """
    def freeze_backbone(self, model: nn.Module, freeze_until: str = 'lstm'):
        for name, param in model.named_parameters():
            if 'cnn' in name:
                param.requires_grad = False
            if freeze_until == 'attention' and 'lstm' in name:
                param.requires_grad = False

    def unfreeze_all(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def get_trainable_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_fine_tuning(self, model: nn.Module, num_classes_new: int, lr_backbone: float = 1e-5, lr_head: float = 1e-3):
        # FC (Classifier) katmanını yeni sınıf sayısına göre değiştir
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes_new)
        return model

class TrainingPipeline:
    """
    Tüm eğitim sürecinin (veri hazırlığından, eğitime ve doğrulamaya kadar) orkestratörü.
    """
    def run(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Dict:
        device = get_device()
        print(f"Eğitim başlıyor. Kullanılan cihaz: {device}")
        
        # NumPy -> PyTorch Tensörleri
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) # (Batch, 1, Seq_len)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
        
        # Trainer
        # Modül 5'teki class parametresine göre dinamik sınıf olarak üretiliyor
        model = ECGHybridModel(num_classes=getattr(self, 'num_classes', 7)) 
        
        # Trainer
        trainer = ECGTrainer(model=model, device=device)
        history = trainer.train(train_loader, val_loader, epochs=epochs)
        
        return {'history': history, 'best_model': model, 'trainer': trainer}
