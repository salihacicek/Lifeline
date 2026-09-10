import wfdb
import numpy as np

record = wfdb.rdrecord("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/00000/00177_lr", channels=[0])
ecg_mv = record.p_signal[:, 0]
print(f"p_signal shape: {ecg_mv.shape}")
print(f"p_signal min: {np.min(ecg_mv)}, max: {np.max(ecg_mv)}")
print(f"p_signal first 10 values: {ecg_mv[:10]}")
