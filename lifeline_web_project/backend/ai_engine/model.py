import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss helps address class imbalance by down-weighting well-classified examples.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class HybridECGModel(nn.Module):
    """
    1D-CNN + BiLSTM architecture for ECG waveform analysis.
    Takes raw ECG signals as input.
    """
    def __init__(self, input_channels=1, num_classes=1):
        super(HybridECGModel, self).__init__()
        
        # 1D CNN for spatial feature extraction (morphology)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # BiLSTM for temporal dependencies (arrhythmias over time)
        # Assuming input sequence length after pooling is e.g. L/4
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, 
                            batch_first=True, bidirectional=True)
        
        # Fully connected for classification
        self.fc1 = nn.Linear(64 * 2, 64) # BiLSTM gives 2 * hidden_size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Length) -> e.g. (32, 1, 1000)
        
        # CNN layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Prepare for LSTM: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1) 
        
        # BiLSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the last hidden state from both directions
        # lstm_out shape: (Batch, Seq_Len, hidden_size * 2)
        # We can just take the last time step, or use max pooling over time.
        # Let's use max pooling over time
        x, _ = torch.max(lstm_out, dim=1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class XGBoostEnsemble:
    """
    Placeholder for the XGBoost model that takes PQRST durations (HR, PR, QT, QRS)
    and acts as a secondary classifier.
    """
    def __init__(self):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
        
    def train(self, features, labels):
        # features: array of [HR, QRS, PR, QT]
        self.model.fit(features, labels)
        
    def predict(self, features):
        return self.model.predict(features)
