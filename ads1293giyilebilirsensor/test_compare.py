import numpy as np
import scipy.signal
import wfdb
import matplotlib.pyplot as plt

from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import PanTompkinsDetector

record = wfdb.rdrecord("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/00000/00177_lr", channels=[0])
ecg_mv = record.p_signal[:, 0]
ecg_uv = ecg_mv * 1000.0
raw_adc_float = ecg_uv / 0.286102294921875
raw_adc_24bit = np.clip(np.round(raw_adc_float).astype(np.int32), -8388608, 8388607)

# Method 1 (Training style)
fs = 100
pipeline_100 = SignalCleaningPipeline(fs=fs)
clean_100 = pipeline_100.run(raw_adc_24bit)["ecg_cleaned"]
pt_100 = PanTompkinsDetector(fs=fs)
peaks_100 = pt_100.detect(clean_100)
rp_100 = peaks_100[len(peaks_100)//2]
beat_100 = clean_100[rp_100 - int(fs * 0.3) : rp_100 + int(fs * 0.5)]
beat_100_resampled = scipy.signal.resample(beat_100, 250)

# Method 2 (GUI style)
orig_fs = 100
new_fs = 360
num_samples = int(len(ecg_mv) * new_fs / orig_fs)
ecg_mv_360 = scipy.signal.resample(ecg_mv, num_samples)
sim_data_360 = ecg_mv_360 * 1000.0

pipeline_360 = SignalCleaningPipeline(fs=new_fs)
clean_360 = pipeline_360.run(sim_data_360 * 1000.0, normalize=True)["ecg_normalized"]
pt_360 = PanTompkinsDetector(fs=new_fs)
peaks_360 = pt_360.detect(clean_360)
rp_360 = peaks_360[len(peaks_360)//2]
beat_360 = clean_360[rp_360 - int(new_fs * 0.3) : rp_360 + int(new_fs * 0.5)]
beat_360_resampled = scipy.signal.resample(beat_360, 250)

diff = np.abs(beat_100_resampled - beat_360_resampled).mean()
print("Mean absolute difference:", diff)
print("Max absolute difference:", np.abs(beat_100_resampled - beat_360_resampled).max())
print("Mean of 100:", beat_100_resampled.mean(), "Mean of 360:", beat_360_resampled.mean())
print("Std of 100:", beat_100_resampled.std(), "Std of 360:", beat_360_resampled.std())
