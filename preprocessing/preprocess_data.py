#preprrocess all the data in the parquet file

import numpy as np
import pandas as pd
import pickle
from scipy import signal
from tqdm import tqdm
from mne.preprocessing import ICA
import mne
import matplotlib.pyplot as plt
import scipy.stats

# Load data from pickle file
print("Loading EEG dataset from pickle file...")
with open("eeg_dataset.pkl", 'rb') as f:
    data = pickle.load(f)

# Extract dataset and labels
dataset = data['dataset']
labels = data['labels']

print(f"Dataset shape: {dataset.shape}")
print(f"Labels shape: {labels.shape}")

# Define preprocessing parameters
sampling_rate = 250  # Hz (adjust based on your actual sampling rate)
lowcut_general = 0.5  # Hz - for general filtering
highcut = 50  # Hz
filter_order = 4  # Reduced from 6 to 4 for better stability

# New parameter for ICA-specific filtering
lowcut_ica = 1.0  # Hz - higher cutoff specifically for ICA

# Step 1: Band-Pass Filtering
print("\nApplying band-pass filtering...")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Apply general filtering to each sample and channel
filtered_dataset = np.zeros_like(dataset)
for i in tqdm(range(len(dataset)), desc="Filtering"):
    for ch in range(dataset[i].shape[0]):
        filtered_dataset[i, ch] = apply_bandpass_filter(
            dataset[i, ch], lowcut_general, highcut, sampling_rate, filter_order)

# Step 2: Artifact Removal using ICA
print("\nPerforming artifact removal with ICA...")

def apply_ica(data, n_components=3):
    """Apply ICA to remove artifacts from EEG data using MNE's built-in detection"""
    # Create an MNE-compatible data structure
    ch_names = ['TP9', 'FP1', 'FP2', 'TP10']
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
    
    # Process each sample
    cleaned_data = np.zeros_like(data)
    for i in tqdm(range(data.shape[0]), desc="ICA processing"):
        # Create an MNE Raw object
        raw = mne.io.RawArray(data[i], info)
        
        # Apply 1 Hz high-pass filter specifically for ICA
        raw_ica = raw.copy()
        raw_ica.filter(l_freq=lowcut_ica, h_freq=None, filter_length=612)
        
        # Apply ICA with fixed number of components and reduced verbosity
        ica = ICA(n_components=n_components, random_state=42, method='fastica', verbose=False)
        
        # Fit ICA on the high-pass filtered data with reduced verbosity
        ica.fit(raw_ica, verbose=False)
        
        # Use MNE's built-in method to find artifacts with reduced verbosity
        eog_indices, scores = ica.find_bads_eog(raw_ica, ch_name=['FP1', 'FP2'], verbose=False)
        
        # Remove the artifacts
        if eog_indices:
            if i % 10000 == 0:  # Print occasionally
                print(f"Sample {i}: Removing {len(eog_indices)} artifact components")
            ica.exclude = eog_indices
            cleaned = ica.apply(raw.copy(), verbose=False)  # Add verbose=False here
            cleaned_data[i] = cleaned.get_data()
        else:
            # If no artifacts found, keep original
            cleaned_data[i] = raw.get_data()
    
    return cleaned_data

# Apply ICA to remove artifacts
try:
    cleaned_dataset = apply_ica(filtered_dataset)
except Exception as e:
    print(f"ICA failed with error: {e}")
    print("Continuing without artifact removal")
    cleaned_dataset = filtered_dataset

# Plot a sample before and after preprocessing for verification
plt.figure(figsize=(15, 8))

# Original sample
plt.subplot(3, 1, 1)
plt.title("Original Signal (First Sample, First Channel)")
plt.plot(dataset[0, 0])

# Filtered sample
plt.subplot(3, 1, 2)
plt.title("After Filtering")
plt.plot(filtered_dataset[0, 0])

# Final preprocessed sample
plt.subplot(3, 1, 3)
plt.title("After ICA Artifact Removal")
plt.plot(cleaned_dataset[0, 0])

plt.tight_layout()
plt.savefig("preprocessing_visualization.png")
plt.close()

# Save the preprocessed dataset to parquet
print("\nSaving preprocessed dataset to parquet...")

# Create a dictionary to hold all data
data_dict = {'label': labels}

# Pre-create all column names
column_names = []
for ch_idx in range(cleaned_dataset[0].shape[0]):
    for ts_idx in range(cleaned_dataset[0].shape[1]):
        column_names.append(f'ch{ch_idx}_ts{ts_idx}')

# Initialize empty arrays for each column
for col in column_names:
    data_dict[col] = np.zeros(len(cleaned_dataset))

# Fill in the data
for i in tqdm(range(len(cleaned_dataset)), desc="Flattening data for parquet"):
    sample = cleaned_dataset[i]
    
    # Flatten the sample into columns
    for ch_idx in range(sample.shape[0]):
        for ts_idx in range(sample.shape[1]):
            col_name = f'ch{ch_idx}_ts{ts_idx}'
            data_dict[col_name][i] = sample[ch_idx, ts_idx]

# Create DataFrame all at once (no fragmentation)
processed_df = pd.DataFrame(data_dict)

# Add preprocessing metadata
processed_df['preprocessing'] = 'bandpass_ica'
processed_df['lowcut'] = lowcut_general
processed_df['highcut'] = highcut

# Write to parquet
output_file = "processed_parquet.parquet"
processed_df.to_parquet(output_file, compression='snappy')

print(f"Preprocessed dataset saved to {output_file}")
print("\nPreprocessing complete! Visualization saved to preprocessing_visualization.png")