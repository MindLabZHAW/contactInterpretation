import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random

class LoadSeqDataset(Dataset):
    def __init__(self, file_path: str, label: int, selected_features: list, mode: str, seq_num=28, gap=5):
        """
        An optimized dataset loader that uses vectorized operations for speed.
        
        Args:
            file_path (str): Path to the CSV file.
            label (int): The label to be applied.
            selected_features (list): List of feature columns to use.
            mode (str): Determines which part of the data to use. 
                        Accepts 'train' for the first half or 'val' for the second half.
            seq_num (int): The length of each sequence.
            gap (int): The step size between the start of consecutive sequences.
        """
        self.seq_num = seq_num
        self.gap = gap
        
        # Load the entire dataframe first
        df = pd.read_csv(file_path, usecols=selected_features + ['label', 'time'], engine='python')
        
        # --- NEW: Splitting logic based on 'mode' ---
        num_rows = len(df)
        mid_point = num_rows - num_rows // 3
        
        if mode == 'train':
            # Use the first half of the dataframe
            data_split = df.iloc[:mid_point]
        elif mode == 'val':
            # Use the second half of the dataframe
            data_split = df.iloc[mid_point:]
        else:
            raise ValueError("Mode must be either 'train' or 'val'")
        # --- End of new logic ---

        # Ensure there's enough data in the split to create at least one sequence
        if len(data_split) < self.seq_num:
            # If not, create an empty dataset
            self.sequences = np.array([])
            self.labels = np.array([])
        else:
            # Process features and labels from the selected data split
            features_data = data_split[selected_features].values.astype(np.float32)
            labels_data = data_split['label'].values * label.item()
            time_info = data_split['time'].values
            
            self.sequences, self.labels , self.times= self._make_sequences_vectorized(features_data, labels_data, time_info)

    def _make_sequences_vectorized(self, features, labels, time_info):
        """
        Creates sequences using efficient NumPy array manipulation.
        """
        # 1. Define the shape of the output array.
        #    This will be a 3D array: (number_of_sequences, number_of_features, sequence_length).
        #    'features.shape[0] - self.seq_num + 1' calculates the total number of
        #    possible sequences that can be extracted from the data.
        shape = (features.shape[0] - self.seq_num + 1, features.shape[1], self.seq_num)

        # 2. Define the strides for the new array view.
        #    Strides tell NumPy how many bytes to jump in memory to get to the next element
        #    along each axis. By reusing the stride of the first axis (features.strides[0]),
        #    we create the sliding window effect without copying any data.
        strides = (features.strides[0], features.strides[1], features.strides[0])

        # 3. Create a memory-efficient "view" of the data.
        #    'as_strided' creates a new NumPy array that looks at the original 'features'
        #    data with the new shape and strides. No data is duplicated, making this
        #    incredibly fast and memory-efficient.
        windows = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)
        
        # 4. Get the label for each sequence.
        #    The label for each sequence is determined by the label of its last time step.
        #    This slice starts from the end of the first possible window and selects
        #    all subsequent labels.
        sequence_labels = labels[self.seq_num - 1:]
        sequence_times = time_info[self.seq_num-1:]
        
        # 5. Apply the 'gap' to subsample the data.
        #    The [::self.gap] slice selects every Nth sequence and its corresponding label,
        #    where N is the gap size. This is a fast, vectorized way to reduce
        #    the overlap between consecutive sequences.
        return windows[::self.gap], sequence_labels[::self.gap], sequence_times[::self.gap]
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.sequences[idx])
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, target