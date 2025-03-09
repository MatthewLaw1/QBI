import numpy as np
from scipy import signal

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to EEG data
    
    Args:
        data: 1D array of EEG data
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        fs: Sampling rate
        order: Filter order
        
    Returns:
        Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Create bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data 