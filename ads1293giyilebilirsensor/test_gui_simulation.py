import wfdb
import numpy as np
import torch
import pandas as pd
from ads1293_pipeline.module5_deep_learning import ECGHybridModel
from ads1293_pipeline.module3_feature_extraction import PQRSTExtractor
import scipy.signal
from ads1293_pipeline.module2_signal_processing import PanTompkinsDetector

# Load classes
scp_df = pd.read_csv("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv", index_col=0)
PTBXL_CLASSES = list(scp_df.index)
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

model = ECGHybridModel(in_channels=1, num_classes=76)
model.load_state_dict(torch.load("ecg_model_weights_76class.pth", map_location="cpu", weights_only=True))
model.eval()

record = wfdb.rdrecord("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/00000/00001_lr", channels=[0])
ecg_uv = record.p_signal[:, 0] * 1000.0

# Simulate live stream (as in gui_app.py run_ai_analysis)
fs = 100
pt_detector = PanTompkinsDetector(fs=fs)
pqrst_extractor = PQRSTExtractor(fs=fs)

# The GUI usually processes chunks.
peaks = pt_detector.run(ecg_uv)["r_peaks"]

predictions = []
for i in range(1, len(peaks)):
    r_peak = peaks[i]
    start = max(0, r_peak - int(0.2 * fs))
    end = min(len(ecg_uv), r_peak + int(0.4 * fs))
    segment = ecg_uv[start:end]
    
    if len(segment) != 250:
        segment = scipy.signal.resample(segment, 250)
        
    input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    logits = model(input_tensor)
    cnn_probs = torch.nn.functional.softmax(logits, dim=1)[0].detach().numpy()
    
    pred_class = np.argmax(cnn_probs)
    pred_name = ALL_CLASSES[pred_class]
    cat = scp_df.loc[pred_name]['diagnostic_class'] if pred_name in PTBXL_CLASSES else "MIT"
    
    predictions.append((pred_name, cat, cnn_probs[pred_class]))

for p in predictions:
    print(p)

