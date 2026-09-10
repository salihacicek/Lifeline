import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from signal_processing import extract_pqrst_features

DATA_DIR = "/Users/salihacicek/Desktop/lifeline_web_project/Lifeline_Verileri/EKG Veri"

class HybridECGModel(nn.Module):
    def __init__(self):
        super(HybridECGModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
try:
    model = HybridECGModel().to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_engine", "ecg_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except:
    model = None

results = []

for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(('.txt', '.csv')):
        continue
        
    file_path = os.path.join(DATA_DIR, filename)
    try:
        if filename.endswith('.txt'):
            df = pd.read_csv(file_path, header=None, sep=',', on_bad_lines='skip')
            raw_signal = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
        else:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            raw_signal = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
            
        total_seconds = len(raw_signal) / 213.0
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        duration_str = f"{mins} dk {secs} sn"
        
        # Process up to 50000 points to save time in the script
        process_len = min(len(raw_signal), 50000)
        sig_to_process = raw_signal[:process_len]
        
        filtered_sig, _, intervals = extract_pqrst_features(sig_to_process, fs=213)
        
        analyzed_beats = 0
        normal_beats = 0
        abnormal_beats = 0
        
        for idx in range(213, len(filtered_sig), 213):
            window = filtered_sig[idx-213:idx]
            if len(window) == 213:
                analyzed_beats += 1
                if model is not None:
                    tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(tensor_x)
                        pred = torch.argmax(out, dim=1).item()
                        if pred == 1:
                            abnormal_beats += 1
                        else:
                            normal_beats += 1
                else:
                    # Fallback
                    int_idx = min(idx // 213, len(intervals["hr"])-1) if intervals["hr"] else 0
                    hr = intervals["hr"][int_idx] if intervals["hr"] else 75
                    if hr > 100 or hr < 60:
                        abnormal_beats += 1
                    else:
                        normal_beats += 1
                        
        if analyzed_beats == 0:
            result_str = "Bilinmiyor"
        else:
            risk_ratio = (abnormal_beats / analyzed_beats) * 100
            if risk_ratio == 0:
                result_str = "Sağlıklı"
            elif risk_ratio < 10:
                result_str = "Düşük Risk (Erken Atım)"
            elif risk_ratio < 30:
                result_str = "Hasta (Orta Risk - Taşikardi)"
            else:
                result_str = "Hasta (Kritik Aritmi)"
                
        results.append(f"- **{filename}**: Süre: {duration_str} | Teşhis: {result_str} (Anormal atım: {abnormal_beats}/{analyzed_beats})")
    except Exception as e:
        results.append(f"- **{filename}**: Okunamadı ({e})")

for r in results:
    print(r)
