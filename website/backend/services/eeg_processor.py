import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.eeg_data import EEGData
from scipy import signal
import mne
from mne.preprocessing import ICA
import warnings

# Silence warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

def apply_bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    Apply a Chebyshev Type II bandpass filter to the data
    
    Args:
        data: Input signal
        lowcut: Lower cutoff frequency
        highcut: Higher cutoff frequency
        fs: Sampling frequency
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Use Chebyshev Type II filter (matches preprocessing.py)
    sos = signal.cheby2(order, 40, [low, high], btype='band', output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    
    return filtered_data

def apply_ica(data, sampling_rate, n_components=3):
    """
    Apply ICA to remove artifacts from EEG data
    
    Args:
        data: EEG data with shape (channels, samples)
        sampling_rate: Sampling rate in Hz
        n_components: Number of ICA components
        
    Returns:
        Cleaned data
    """
    # Create MNE-compatible data structure
    ch_names = ['TP9', 'FP1', 'FP2', 'TP10']  # Adjust based on your channel names
    info = mne.create_info(ch_names=ch_names[:data.shape[0]], sfreq=sampling_rate, ch_types='eeg')
    
    # Create an MNE Raw object
    raw = mne.io.RawArray(data, info)
    
    # Apply 1 Hz high-pass filter specifically for ICA (as in preprocessing.py)
    raw_ica = raw.copy()
    raw_ica.filter(l_freq=1.0, h_freq=None, verbose=False)
    
    # Apply ICA
    try:
        ica = ICA(n_components=n_components, random_state=42, method='fastica', verbose=False)
        ica.fit(raw_ica, verbose=False)
        
        # Find and remove EOG artifacts
        eog_indices, _ = ica.find_bads_eog(raw_ica, ch_name=['FP1', 'FP2'], verbose=False)
        
        if eog_indices:
            ica.exclude = eog_indices
            cleaned = ica.apply(raw.copy(), verbose=False)
            return cleaned.get_data()
        else:
            return raw.get_data()
    except Exception as e:
        print(f"ICA failed with error: {e}")
        return data  # Return original data if ICA fails

def process_eeg_data(data: EEGData) -> dict:
    """
    Process incoming EEG data
    
    Args:
        data: Raw EEG data from Muse 2
        
    Returns:
        Processed data ready for inference
    """
    # Convert to numpy array for processing
    channels_data = np.array(data.channels)
    
    # Step 1: Apply bandpass filtering (Chebyshev Type II, 6th order)
    filtered_channels = np.zeros_like(channels_data, dtype=np.float32)
    for i in range(len(channels_data)):
        filtered_channels[i] = apply_bandpass_filter(
            channels_data[i], 
            lowcut=0.5, 
            highcut=50.0, 
            fs=256,  # Muse 2 sampling rate
            order=6   # 6th order as requested
        )
    
    # Step 2: Apply ICA for artifact removal
    try:
        cleaned_channels = apply_ica(filtered_channels, sampling_rate=256)
    except Exception as e:
        print(f"Error in ICA processing: {e}")
        cleaned_channels = filtered_channels  # Fallback to filtered data
    
    # Step 3: Z-score normalization
    normalized_channels = np.zeros_like(cleaned_channels, dtype=np.float32)
    for i in range(len(cleaned_channels)):
        channel = cleaned_channels[i]
        mean = np.mean(channel)
        std = np.std(channel)
        if std > 0:  # Avoid division by zero
            normalized_channels[i] = (channel - mean) / std
        else:
            normalized_channels[i] = channel  # Keep original if std is 0
    
    # Return processed data
    return {
        "timestamp": data.timestamp,
        "processed_channels": normalized_channels.tolist(),
        "accelerometer": data.accelerometer,
        "gyroscope": data.gyroscope
    } 


if __name__ == "__main__":
    # Generate random values between 400 and 500
    channels = 400 + np.random.rand(4, 2000) * 100  # 4 channels, 2000 samples each
    timestamp = 123.456
    accelerometer = [0.1, 0.2, 0.3]
    gyroscope = [0.01, 0.02, 0.03]
    eeg_data = EEGData(channels=channels)
    print(process_eeg_data(eeg_data))