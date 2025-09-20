import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random

class LoadSeqDataset(Dataset):
    def __init__(self, file_path: str, label: int, selected_features: list, seq_num=28, gap=5):
        """
        An optimized dataset loader that uses vectorized operations for speed.
        """
        self.seq_num = seq_num
        self.gap = gap
        
        # Load the necessary data directly into NumPy arrays for performance
        df = pd.read_csv(file_path, usecols=selected_features + ['label'], engine='python')
        
        # Process features and labels
        features_data = df[selected_features].values.astype(np.float32)
        # Create a single label for the entire file's content
        is_contact_file = 1 if label > 0 else 0
        labels_data = (df['label'].values > 0).astype(np.int8) * is_contact_file
        
        self.sequences, self.labels = self._make_sequences_vectorized(features_data, labels_data)

    def _make_sequences_vectorized(self, features, labels):
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
        
        # 5. Apply the 'gap' to subsample the data.
        #    The [::self.gap] slice selects every Nth sequence and its corresponding label,
        #    where N is the gap size. This is a fast, vectorized way to reduce
        #    the overlap between consecutive sequences.
        return windows[::self.gap], sequence_labels[::self.gap]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # The data is already in the correct [dof, seq_num] format
        features = torch.from_numpy(self.sequences[idx])
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, target