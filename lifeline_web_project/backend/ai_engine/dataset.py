import numpy as np

def add_gaussian_noise(signal, snr=20):
    """Adds Gaussian noise to a signal to simulate artifacts."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

def time_warp(signal, stretch_factor=1.1):
    """Simulates time warping (slower/faster heart rate) using linear interpolation."""
    original_indices = np.arange(len(signal))
    new_indices = np.linspace(0, len(signal) - 1, int(len(signal) * stretch_factor))
    return np.interp(new_indices, original_indices, signal)

def augment_ecg_data(signals, labels):
    """
    Applies data augmentation to minority classes to balance the dataset.
    This simulates SMOTE + synthetic generation.
    """
    augmented_signals = []
    augmented_labels = []
    
    for sig, label in zip(signals, labels):
        augmented_signals.append(sig)
        augmented_labels.append(label)
        
        # If it's an anomaly (e.g. label 1), augment it
        if label == 1:
            # 1. Add noise
            noisy = add_gaussian_noise(sig, snr=15)
            augmented_signals.append(noisy)
            augmented_labels.append(label)
            
            # 2. Time stretch (bradycardia simulation)
            stretched = time_warp(sig, stretch_factor=1.2)
            # truncate to original length for uniform CNN input
            augmented_signals.append(stretched[:len(sig)]) 
            augmented_labels.append(label)
            
            # 3. Time compress (tachycardia simulation)
            compressed = time_warp(sig, stretch_factor=0.8)
            # pad to original length
            padded = np.pad(compressed, (0, len(sig) - len(compressed)), 'constant')
            augmented_signals.append(padded)
            augmented_labels.append(label)
            
    return np.array(augmented_signals), np.array(augmented_labels)
