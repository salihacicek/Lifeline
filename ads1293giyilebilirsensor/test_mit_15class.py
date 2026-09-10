import torch
import numpy as np
import scipy.signal

from ads1293_pipeline.module1_data_acquisition import RawDataPipeline
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import PanTompkinsDetector, FeatureExtractionPipeline
from ads1293_pipeline.module5_deep_learning import ECGHybridModel

PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

model = ECGHybridModel(in_channels=1, num_classes=15)
model.load_state_dict(torch.load("ecg_model_weights_15class.pth", map_location="cpu", weights_only=True))
model.eval()

pipeline1 = RawDataPipeline(record_id="100", duration_seconds=60)
data = pipeline1.fetch_data()
raw_adc = data["raw_adc_24bit"]
fs = data["fs"]

pipeline2 = SignalCleaningPipeline(fs=fs)
clean_res = pipeline2.run(raw_adc, normalize=True)
ecg_norm = clean_res["ecg_normalized"]

pipeline3 = FeatureExtractionPipeline()
features = pipeline3.run(ecg_norm, fs=fs)
segments = features.get("heartbeat_segments", [])

preds = {}
for beat in segments:
    if len(beat) != 250:
        beat = scipy.signal.resample(beat, 250)
    input_tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(input_tensor), dim=1)[0].numpy()
        p = np.argmax(probs)
        cls_name = ALL_CLASSES[p]
        preds[cls_name] = preds.get(cls_name, 0) + 1

print("Record 100 Predictions:", preds)

# Test record 101 as well
pipeline1 = RawDataPipeline(record_id="101", duration_seconds=60)
data = pipeline1.fetch_data()
clean_res = pipeline2.run(data["raw_adc_24bit"], normalize=True)
features = pipeline3.run(clean_res["ecg_normalized"], fs=fs)
preds = {}
for beat in features.get("heartbeat_segments", []):
    if len(beat) != 250:
        beat = scipy.signal.resample(beat, 250)
    input_tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        p = np.argmax(model(input_tensor)[0].numpy())
        cls_name = ALL_CLASSES[p]
        preds[cls_name] = preds.get(cls_name, 0) + 1
print("Record 101 Predictions:", preds)
