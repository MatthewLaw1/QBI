import requests
import gdown  # You'll need to install this: pip install gdown
import os.path  # Add this import for file existence check
import numpy as np  # Add numpy for statistical operations

# Your file ID
file_id = '1AnnW4R9-pzEUcl8V0LucvfeJ2CUOf5KI'

# Option 1: Using gdown (recommended for Google Drive files)
output_file = "downloaded_file.txt"

# Only download if the file doesn't already exist
if not os.path.exists(output_file):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
    print(f"File downloaded to {output_file}")
else:
    print(f"File {output_file} already exists, skipping download")

# Read the file
with open(output_file, "r") as file:
    data = file.read()

# Calculate mean and standard deviation from first 200 values
try:
    # Initialize a list to store the first 200 values
    first_200_values = []
    
    # Parse the data to extract values
    for line in data.strip().split('\n'):
        fields = line.split('\t')
        if len(fields) < 7:  # Skip malformed lines
            continue
            
        device = fields[2]
        if device != "MU":  # Only process MUSE data
            continue
        
        # Get the raw values from the line
        raw_values = fields[6].split(',')
        
        # Convert values to float and add to our list
        try:
            values = [float(val.strip()) for val in raw_values]
            # Add values to our list, but only until we reach 200
            first_200_values.extend(values[:min(len(values), 200 - len(first_200_values))])
            
            # If we have 200 values, break the loop
            if len(first_200_values) >= 200:
                break
                
        except ValueError:
            print(f"Warning: Could not parse values in a line. Skipping.")
    
    # Calculate mean and standard deviation
    if first_200_values:
        mean_value = np.mean(first_200_values)
        std_value = np.std(first_200_values)
        
        print(f"\nStatistics from first 200 values:")
        print(f"Mean: {mean_value:.4f}")
        print(f"Standard Deviation: {std_value:.4f}")
    else:
        print("No values found to calculate statistics.")
        mean_value = 0
        std_value = 1  # Default to avoid division by zero

    # Define the expected channel order
    channel_order = ["TP9", "FP1", "FP2", "TP10"]
    
    # Create dataset directly in a single pass
    # We'll use a dictionary to track the current sample for each code and channel
    current_samples = {}  # {code: {channel: [values]}}
    dataset = []
    labels = []
    
    # Import tqdm for progress tracking and pickle for saving
    from tqdm import tqdm
    import pickle
    
    # Process all lines and build dataset directly
    for i, line in enumerate(tqdm(data.strip().split('\n'), desc="Processing data")):
        fields = line.split('\t')
        if len(fields) < 7:  # Skip malformed lines
            continue
            
        device = fields[2]
        if device != "MU":  # Only process MUSE data
            continue
            
        channel = fields[3]
        if channel not in channel_order:
            continue  # Skip channels we don't want
            
        code = int(fields[4])  # The digit being thought/seen
        raw_values = fields[6].split(',')
        
        # Convert values to float
        try:
            values = [float(val.strip()) for val in raw_values]
            
            # Normalize values
            normalized_values = [(v - mean_value) / std_value for v in values]
            
            # Initialize code entry if needed
            if code not in current_samples:
                current_samples[code] = {ch: [] for ch in channel_order}
            
            # Add values to the current sample for this channel
            current_samples[code][channel].append(normalized_values)
            
            # Check if we have data for all channels
            all_channels_have_data = all(len(current_samples[code][ch]) > 0 for ch in channel_order)
            
            if all_channels_have_data:
                # Find minimum length across all channels
                min_length = min(len(current_samples[code][ch][0]) for ch in channel_order)
                
                if min_length > 0:
                    # Create a sample with shape (n_channels, n_timesteps)
                    sample = np.zeros((len(channel_order), min_length))
                    
                    # Fill in data for each channel
                    for ch_idx, ch in enumerate(channel_order):
                        sample[ch_idx, :min_length] = current_samples[code][ch][0][:min_length]
                    
                    # Add to dataset
                    dataset.append(sample)
                    labels.append(code)
                    
                    # Remove the used data
                    for ch in channel_order:
                        current_samples[code][ch].pop(0)
            
        except ValueError:
            print(f"Warning: Could not parse values in a line. Skipping.")
    
    # Convert to numpy arrays
    # First, find the maximum length across all samples
    max_length = 0
    for sample in dataset:
        for channel in sample:
            max_length = max(max_length, len(channel))
    
    # Create a padded dataset with consistent dimensions
    padded_dataset = []
    for sample in dataset:
        # Create a zero-padded sample
        padded_sample = np.zeros((len(channel_order), max_length))
        for ch_idx, channel_data in enumerate(sample):
            # Copy the data (will be truncated if longer than max_length)
            padded_sample[ch_idx, :len(channel_data)] = channel_data[:max_length]
        padded_dataset.append(padded_sample)
    
    # Now convert to numpy array
    dataset = np.array(padded_dataset)  # Shape: (batch_size, n_channels, n_timesteps)
    labels = np.array(labels)    # Shape: (batch_size,)
    
    print(f"\nDataset created with shape: {dataset.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Save dataset and labels to pickle file
    output_pkl = "eeg_dataset.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump({'dataset': dataset, 'labels': labels}, f)
    print(f"Dataset saved to {output_pkl}")
    
except Exception as e:
    print(f"Error processing data: {e}")
    print("You may need to adjust the data parsing logic based on your file format")

print(data[:500])

# Option 2: If gdown doesn't work, you can try using a shareable link
# 1. Go to Google Drive, right-click your file
# 2. Select "Share" > "Anyone with the link"
# 3. Copy the link and use it like this:
# shared_link = "YOUR_SHARED_LINK_HERE"
# output = "downloaded_file.txt"
# gdown.download(shared_link, output, quiet=False)