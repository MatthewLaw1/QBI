print("Starting conversion...")

import pickle
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Load the pickle file
print("Loading pickle file...")
with open("eeg_dataset.pkl", 'rb') as f:
    data = pickle.load(f)

# Extract dataset and labels
dataset = data['dataset']
labels = data['labels']

def convert_parquet_to_np(parquet_file):
    """
    Convert a parquet file back to numpy arrays (dataset and labels).
    Automatically detects whether the file uses binary blob or delta encoding format.
    
    Args:
        parquet_file (str): Path to the parquet file
        
    Returns:
        tuple: (dataset, labels) as numpy arrays
    """
    # Read the parquet file
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    
    # Check which format was used (binary or delta)
    if 'data' in df.columns:
        # Binary blob format
        print("Detected binary blob format")
        dataset = []
        for i in range(len(df)):
            sample = np.frombuffer(df['data'][i], dtype=np.dtype(df['dtype'][i])).reshape(
                df['shape_0'][i], df['shape_1'][i])
            dataset.append(sample)
        dataset = np.array(dataset)
        labels = df['label'].values
    else:
        # Delta encoding format
        print("Detected delta encoding format")
        # Determine the shape from the column names
        ch_count = sum(1 for col in df.columns if col.startswith('ch') and col.endswith('_start'))
        
        dataset = []
        for i in tqdm(range(len(df)), desc="Reconstructing samples"):
            # Get the shape of the first sample's deltas to determine time dimension
            first_deltas = df.loc[i, 'ch0_deltas'].split(',')
            time_dim = len(first_deltas) + 1  # +1 because deltas are differences
            
            sample = np.zeros((ch_count, time_dim))
            for ch_idx in range(ch_count):
                # Get starting value
                start = df.loc[i, f'ch{ch_idx}_start']
                # Get deltas and convert back to array
                deltas = np.array([float(x) for x in df.loc[i, f'ch{ch_idx}_deltas'].split(',')])
                # Reconstruct the channel data
                channel_data = np.zeros(len(deltas) + 1)
                channel_data[0] = start
                channel_data[1:] = start + np.cumsum(deltas)
                sample[ch_idx, :] = channel_data
            dataset.append(sample)
        dataset = np.array(dataset)
        labels = df['label'].values
    
    return dataset, labels

print(f"Dataset shape: {dataset.shape}")
print(f"Labels shape: {labels.shape}")

# APPROACH 1: Store as binary blobs with high compression
print("Converting to optimized format...")

# Create arrays for our data
serialized_data = []
for sample in tqdm(dataset, desc="Serializing samples"):
    # Store as binary data
    serialized_data.append(sample.tobytes())

# Create a PyArrow table with binary data
table = pa.table({
    'data': pa.array(serialized_data, type=pa.binary()),
    'shape_0': pa.array([sample.shape[0] for sample in dataset]),
    'shape_1': pa.array([sample.shape[1] for sample in dataset]),
    'dtype': pa.array([str(dataset.dtype)] * len(dataset)),
    'label': pa.array(labels)
})

# Try different compression algorithms
print("\nTesting compression algorithms...")
best_file = None
best_size = float('inf')

# Test file paths
test_files = {
    "zstd_high": ("eeg_dataset_zstd.parquet", "zstd", 22),
    "gzip_high": ("eeg_dataset_gzip.parquet", "gzip", 9),
    "brotli_high": ("eeg_dataset_brotli.parquet", "brotli", 11)
}

for name, (file_path, algorithm, level) in test_files.items():
    try:
        print(f"  Testing {name}...")
        pq.write_table(
            table, 
            file_path,
            compression=algorithm,
            compression_level=level,
            row_group_size=min(len(dataset), 10000),  # Larger row groups
            use_dictionary=True
        )
        
        size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  {name}: {size:.2f} MB")
        
        if size < best_size:
            best_size = size
            best_file = file_path
    except Exception as e:
        print(f"  Error with {name}: {e}")

# APPROACH 2: Try delta encoding for time series
print("\nTrying delta encoding approach...")

# Create a dataframe with delta-encoded values
delta_df = pd.DataFrame()
delta_df['label'] = labels

for i in tqdm(range(len(dataset)), desc="Delta encoding"):
    sample = dataset[i]
    
    # For each channel, store the first value and then differences
    for ch_idx in range(sample.shape[0]):
        channel_data = sample[ch_idx]
        
        # Store first value
        delta_df.loc[i, f'ch{ch_idx}_start'] = channel_data[0]
        
        # Store differences (delta encoding)
        diffs = np.diff(channel_data)
        
        # Quantize differences to reduce unique values (improves compression)
        # Round to 3 decimal places
        diffs = np.round(diffs, 3)
        
        # Store as comma-separated string (more compressible than separate columns)
        delta_df.loc[i, f'ch{ch_idx}_deltas'] = ','.join(map(str, diffs))

# Save delta-encoded version
delta_file = "eeg_dataset_delta.parquet"
delta_df.to_parquet(delta_file, compression='zstd', compression_level=22)
delta_size = os.path.getsize(delta_file) / (1024 * 1024)
print(f"  Delta encoding: {delta_size:.2f} MB")

if delta_size < best_size:
    best_size = delta_size
    best_file = delta_file

# Select the best file and rename it
if best_file and best_file != "eeg_dataset.parquet":
    os.rename(best_file, "eeg_dataset.parquet")
    print(f"\nBest compression achieved with {best_file}")
else:
    print("\nNo improvement found")

# Clean up test files
for file_path, _, _ in test_files.values():
    if os.path.exists(file_path) and file_path != best_file:
        os.remove(file_path)

if delta_file != best_file and os.path.exists(delta_file):
    os.remove(delta_file)

# Print file size comparison
pickle_size = os.path.getsize("eeg_dataset.pkl") / (1024 * 1024)  # MB
parquet_size = os.path.getsize("eeg_dataset.parquet") / (1024 * 1024)  # MB
reduction = (1 - parquet_size/pickle_size) * 100

print(f"\nFile size comparison:")
print(f"Pickle file: {pickle_size:.2f} MB")
print(f"Parquet file: {parquet_size:.2f} MB")
print(f"Size reduction: {reduction:.2f}%")

# Provide loading instructions
print("\nTo load the compressed data:")
if best_file == delta_file:
    print("""
# For delta-encoded format:
df = pd.read_parquet('eeg_dataset.parquet')
dataset = []
for i in range(len(df)):
    sample = np.zeros((4, 612))  # Adjust shape as needed
    for ch_idx in range(4):
        # Get starting value
        start = df.loc[i, f'ch{ch_idx}_start']
        # Get deltas and convert back to array
        deltas = np.array([float(x) for x in df.loc[i, f'ch{ch_idx}_deltas'].split(',')])
        # Reconstruct the channel data
        channel_data = np.zeros(len(deltas) + 1)
        channel_data[0] = start
        channel_data[1:] = start + np.cumsum(deltas)
        sample[ch_idx, :len(channel_data)] = channel_data
    dataset.append(sample)
dataset = np.array(dataset)
labels = df['label'].values
""")
else:
    print("""
# For binary format:
table = pq.read_table('eeg_dataset.parquet')
df = table.to_pandas()
dataset = []
for i in range(len(df)):
    sample = np.frombuffer(df['data'][i], dtype=np.dtype(df['dtype'][i])).reshape(
        df['shape_0'][i], df['shape_1'][i])
    dataset.append(sample)
dataset = np.array(dataset)
labels = df['label'].values
""")