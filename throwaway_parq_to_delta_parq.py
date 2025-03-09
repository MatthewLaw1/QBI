# new file to convert parquet to delta parquet
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def convert_flattened_to_delta_parquet(input_parquet, output_parquet="eeg_dataset_delta.parquet"):
    """
    Convert a flattened columnar parquet file to delta-encoded parquet format.
    
    Args:
        input_parquet (str): Path to the input parquet file (flattened format)
        output_parquet (str): Path to save the delta-encoded parquet file
    """
    print(f"Converting {input_parquet} to delta-encoded format...")
    
    # Read the input parquet file
    try:
        df = pd.read_parquet(input_parquet)
        print(f"Successfully loaded parquet with {len(df)} rows")
        print(f"Number of columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return
    
    # Extract labels
    labels = df['label'].values
    
    # Determine the structure of the data
    channel_cols = [col for col in df.columns if col.startswith('ch') and '_ts' in col]
    
    # Find the number of channels and time points
    ch_pattern = set()
    ts_pattern = set()
    
    for col in channel_cols:
        parts = col.split('_')
        ch_pattern.add(parts[0])
        ts_pattern.add(parts[1])
    
    n_channels = len(ch_pattern)
    n_timepoints = len(ts_pattern)
    
    print(f"Detected {n_channels} channels and {n_timepoints} time points")
    
    # Reconstruct the original 3D dataset
    print("Reconstructing original dataset...")
    dataset = np.zeros((len(df), n_channels, n_timepoints))
    
    for ch_idx in range(n_channels):
        for ts_idx in range(n_timepoints):
            col_name = f'ch{ch_idx}_ts{ts_idx}'
            if col_name in df.columns:
                dataset[:, ch_idx, ts_idx] = df[col_name].values
    
    # Create a dataframe with delta-encoded values
    print("Creating delta-encoded dataframe...")
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
    
    # Copy over any metadata columns
    metadata_cols = [col for col in df.columns if col not in channel_cols and col != 'label']
    for col in metadata_cols:
        delta_df[col] = df[col]
    
    # Save delta-encoded version
    print(f"Saving delta-encoded parquet to {output_parquet}...")
    delta_df.to_parquet(output_parquet, compression='zstd', compression_level=22)
    
    # Print file size comparison
    input_size = os.path.getsize(input_parquet) / (1024 * 1024)  # MB
    output_size = os.path.getsize(output_parquet) / (1024 * 1024)  # MB
    reduction = (1 - output_size/input_size) * 100 if input_size > 0 else 0
    
    print(f"\nFile size comparison:")
    print(f"Original parquet: {input_size:.2f} MB")
    print(f"Delta-encoded parquet: {output_size:.2f} MB")
    print(f"Size reduction: {reduction:.2f}%")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    input_file = "processed_parquet.parquet"
    output_file = "processed_parquet_delta.parquet"
    
    convert_flattened_to_delta_parquet(input_file, output_file)
    
    print("\nTo load the delta-encoded data:")
    print("""
# For delta-encoded format:
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_parquet('eeg_dataset_delta.parquet')
dataset = []
ch_count = sum(1 for col in df.columns if col.startswith('ch') and col.endswith('_start'))

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
""")