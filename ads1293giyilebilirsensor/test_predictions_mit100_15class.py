import torch
import numpy as np
import scipy.signal
import wfdb

from ads1293_pipeline.module1_data_acquisition import RawDataPipeline, ADCScaler
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import FeatureExtractionPipeline
from ads1293_pipeline.module5_deep_learning import ECGHybridModel

PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

model = ECGHybridModel(in_channels=1, num_classes=15)
model.load_state_dict(torch.load("ecg_model_weights_15class.pth", map_location="cpu", weights_only=True))
model.eval()

pipeline = RawDataPipeline(record_id="100")
data = pipeline.fetch_data(duration_seconds=30)
raw_adc_24bit = data['raw_adc_24bit']
fs = data['fs']

pipeline2 = SignalCleaningPipeline(fs=fs)
cleaned_signal = pipeline2.run(raw_adc_24bit)["ecg_cleaned"]
    
pipeline3 = FeatureExtractionPipeline()
features = pipeline3.run(cleaned_signal, fs=fs)

print("Number of heartbeats extracted:", len(features.get("heartbeat_segments", [])))
for i, segment in enumerate(features.get("heartbeat_segments", [])):
    if len(segment) != 250:
        segment = scipy.signal.resample(segment, 250)
    input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    logits = model(input_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    pred = np.argmax(probs)
    print(f"Beat {i+1} Pred: {ALL_CLASSES[pred]} Conf: {probs[pred]*100:.1f}%")
