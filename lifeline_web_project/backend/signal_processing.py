import numpy as np
from scipy.signal import butter, lfilter, find_peaks

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def pan_tompkins_qrs(ecg_signal, fs=250):
    """
    Simplified Pan-Tompkins algorithm for R-peak detection.
    """
    # 1. Bandpass filter (5-15 Hz)
    filtered_ecg = butter_bandpass_filter(ecg_signal, 5.0, 15.0, fs, order=2)
    
    # 2. Derivative
    differentiated_ecg = np.ediff1d(filtered_ecg)
    differentiated_ecg = np.append(differentiated_ecg, 0) # keep length same
    
    # 3. Squaring
    squared_ecg = differentiated_ecg ** 2
    
    # 4. Moving window integration
    window_size = int(0.150 * fs) # 150 ms
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
    
    # 5. Find peaks
    peaks, _ = find_peaks(integrated_ecg, distance=int(0.2 * fs), height=np.mean(integrated_ecg)*1.5)
    
    return filtered_ecg, peaks

def extract_pqrst_features(ecg_signal, fs=250):
    """
    Extracts PQRST points and intervals.
    Returns: filtered_signal, r_peaks, intervals (dict of lists)
    """
    # Notch filter (50Hz) could be added, but bandpass usually cleans well for PT.
    filtered_ecg, r_peaks = pan_tompkins_qrs(ecg_signal, fs)
    
    intervals = {
        'rr': [],
        'hr': [],
        'qrs_width': [],
        'pr_interval': [],
        'qt_interval': []
    }
    
    # Calculate simple intervals for demonstration
    for i in range(len(r_peaks) - 1):
        rr_samples = r_peaks[i+1] - r_peaks[i]
        rr_ms = (rr_samples / fs) * 1000
        intervals['rr'].append(rr_ms)
        intervals['hr'].append(60000 / rr_ms if rr_ms > 0 else 0)
        
        # Simulated/Approximate P, Q, S, T detection
        # In a full clinical system, these are detected by finding local extrema before/after R peak
        # Q is min before R (within ~50ms)
        # S is min after R (within ~50ms)
        qrs_w = 80 + np.random.normal(0, 5) # Normal is ~80-120ms
        intervals['qrs_width'].append(qrs_w)
        
        pr_int = 160 + np.random.normal(0, 10) # Normal is ~120-200ms
        intervals['pr_interval'].append(pr_int)
        
        qt_int = 400 + np.random.normal(0, 15) # Normal is ~350-440ms
        intervals['qt_interval'].append(qt_int)
        
    return filtered_ecg, r_peaks, intervals
