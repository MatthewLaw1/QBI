
import pickle
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


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

if __name__ == "__main__":
    dataset, labels = convert_parquet_to_np("eeg_dataset.parquet")
    print(dataset.shape)
    print(labels.shape)
    print(dataset[0])
    print(labels[0])
