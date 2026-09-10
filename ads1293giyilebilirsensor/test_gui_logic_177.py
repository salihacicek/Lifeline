import torch
import numpy as np
import scipy.signal
import wfdb

from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import PanTompkinsDetector
from ads1293_pipeline.module5_deep_learning import ECGHybridModel

PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

model = ECGHybridModel(in_channels=1, num_classes=15)
model.load_state_dict(torch.load("ecg_model_weights_15class.pth", map_location="cpu", weights_only=True))
model.eval()

record = wfdb.rdrecord("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/00000/00177_lr", channels=[0])
ecg_mv = record.p_signal[:, 0]
orig_fs = record.fs

# Resample to 360
num_samples = int(len(ecg_mv) * 360 / orig_fs)
ecg_mv = scipy.signal.resample(ecg_mv, num_samples)
sim_data = ecg_mv * 1000.0

fs = 360
pipeline2 = SignalCleaningPipeline(fs=fs)
pt_detector = PanTompkinsDetector(fs=fs)

# Simulate 5-second sliding windows
window_samples = 5 * fs

for start_idx in range(0, len(sim_data) - window_samples, fs): # slide by 1 second
    buffer = sim_data[start_idx : start_idx + window_samples]
    
    clean_res = pipeline2.run(buffer * 1000.0, normalize=True)
    clean_data_buffer = clean_res["ecg_normalized"]
    
    r_peaks = pt_detector.detect(clean_data_buffer)
    if len(r_peaks) < 2:
        continue
        
    rp = r_peaks[len(r_peaks)//2]
    beat_start = rp - int(fs * 0.3)
    beat_end = rp + int(fs * 0.5)
    
    if beat_start < 0 or beat_end > len(clean_data_buffer):
        continue
        
    beat = clean_data_buffer[beat_start:beat_end]
    beat_resampled = scipy.signal.resample(beat, 250)
    
    input_tensor = torch.tensor(beat_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    logits = model(input_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    pred = np.argmax(probs)
    print(f"Window {start_idx//fs}s Pred: {ALL_CLASSES[pred]} Conf: {probs[pred]*100:.1f}%")

