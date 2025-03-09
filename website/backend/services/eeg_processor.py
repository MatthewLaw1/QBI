import numpy as np
from models.eeg_data import EEGData
from utils.preprocessing import apply_bandpass_filter

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
    
    # Apply preprocessing (e.g., bandpass filtering)
    processed_channels = np.zeros_like(channels_data, dtype=np.float32)
    
    for i in range(len(channels_data)):
        processed_channels[i] = apply_bandpass_filter(
            channels_data[i], 
            lowcut=0.5, 
            highcut=50.0, 
            fs=256  # Muse 2 sampling rate
        )
    
    # Return processed data
    return {
        "timestamp": data.timestamp,
        "processed_channels": processed_channels.tolist(),
        "accelerometer": data.accelerometer,
        "gyroscope": data.gyroscope
    } 