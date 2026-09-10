import torch
import numpy as np
import scipy.signal
import wfdb
import pandas as pd

from ads1293_pipeline.module1_data_acquisition import ADCScaler
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import FeatureExtractionPipeline
from ads1293_pipeline.module5_deep_learning import ECGHybridModel

scp_df = pd.read_csv("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv", index_col=0)
PTBXL_CLASSES = list(scp_df.index)
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

model = ECGHybridModel(in_channels=1, num_classes=76)
model.load_state_dict(torch.load("ecg_model_weights_76class.pth", map_location="cpu", weights_only=True))
model.eval()

record = wfdb.rdrecord("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/00000/00001_lr", channels=[0])
ecg_mv = record.p_signal[:, 0]
fs = record.fs

ecg_uv = ecg_mv * 1000.0
scaler = ADCScaler(v_ref=2.4)
raw_adc_float = ecg_uv / scaler.v_lsb_uv
raw_adc_24bit = np.clip(np.round(raw_adc_float).astype(np.int32), -scaler.adc_max, scaler.adc_max - 1)

pipeline2 = SignalCleaningPipeline(fs=fs)
cleaned_signal = pipeline2.run(raw_adc_24bit)["ecg_cleaned"]
    
pipeline3 = FeatureExtractionPipeline()
features = pipeline3.run(cleaned_signal, fs=fs)

for segment in features.get("heartbeat_segments", []):
    if len(segment) != 250:
        segment = scipy.signal.resample(segment, 250)
    input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    logits = model(input_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    pred = np.argmax(probs)
    print("Pred:", ALL_CLASSES[pred], "Category:", scp_df.loc[ALL_CLASSES[pred]]['diagnostic_class'] if ALL_CLASSES[pred] in PTBXL_CLASSES else "MIT", "Conf:", probs[pred]*100)
    
