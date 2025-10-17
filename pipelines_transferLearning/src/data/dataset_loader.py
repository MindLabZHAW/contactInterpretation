import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import List, Optional

class LoadSeqDataset(Dataset):
    """
    An optimized, high-performance dataset loader for time-series data.

    This class reads CSV files and efficiently converts them into overlapping
    sequences (windows) for training sequential models like LSTMs or Transformers.
    It uses a vectorized approach with NumPy stride tricks to avoid slow loops,
    making it suitable for large datasets.

    The dataset can be split into 'train' and 'val' sets, where 'train' uses
    the first two-thirds of each file and 'val' uses the last one-third.
    """
    def __init__(self, file_path: Optional[str], label: torch.Tensor, selected_features: List[str], 
                 mode: str, window_length: int = 28, gap: int = 5, data_df: Optional[pd.DataFrame] = None):
        """
        Initializes and processes the dataset from a file or DataFrame.

        Args:
            file_path (str, optional): Path to the CSV file.
            label (torch.Tensor): The label to be applied to the data from this file.
            selected_features (List[str]): A list of column names to be used as features.
            mode (str): The dataset split to use ('train' or 'val').
            window_length (int): The length of each sequence or window.
            gap (int): The step size between the start of consecutive sequences. A smaller
                       gap creates more overlapping data.
            data_df (pd.DataFrame, optional): An existing DataFrame to use instead of reading a file.
        """
        if data_df is not None:
            df = data_df
        elif file_path is not None:
            df = pd.read_csv(file_path, usecols=selected_features + ['label', 'time'], engine='python')
        else:
            raise ValueError("You must provide either a 'file_path' or a 'data_df'.")
            
        self.window_length = window_length
        self.gap = gap
        
        # --- Data Splitting Logic ---
        num_rows = len(df)
        # Split point is set to use the first 2/3 for training and last 1/3 for validation
        split_point = num_rows - num_rows // 3
        
        if mode == 'train':
            data_split = df.iloc[:split_point]
        elif mode == 'val':
            data_split = df.iloc[split_point:]
        else:
            raise ValueError("Mode must be either 'train' or 'val'")

        # --- Sequence Creation ---
        if len(data_split) < self.window_length:
            # If the data split is too small to create even one window, create an empty dataset
            self.sequences = np.array([])
            self.labels = np.array([])
            self.times = np.array([])
        else:
            features_data = data_split[selected_features].values.astype(np.float32)
            # Apply the file-level label to the 'label' column
            labels_data = data_split['label'].values * label.item()
            time_info = data_split['time'].values
            
            self.sequences, self.labels, self.times = self._make_sequences_vectorized(features_data, labels_data, time_info)

    def _make_sequences_vectorized(self, features: np.ndarray, labels: np.ndarray, time_info: np.ndarray) -> tuple:
        """
        Creates sliding window sequences using a highly efficient, vectorized NumPy approach.
        This method avoids explicit loops for performance.
        """
        # 1. Define the shape of the output array of windows.
        num_windows = features.shape[0] - self.window_length + 1
        shape = (num_windows, features.shape[1], self.window_length)

        # 2. Define the strides for the new array. Strides control how NumPy steps
        #    through memory to create the "view" of the data. This creates the
        #    sliding window effect without duplicating any data.
        strides = (features.strides[0], features.strides[1], features.strides[0])

        # 3. Create the memory-efficient "view" of overlapping windows.
        windows = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)
        
        # 4. Get the label for each window, which is the label of its last time step.
        sequence_labels = labels[self.window_length - 1:]
        sequence_times = time_info[self.window_length - 1:]
        
        # 5. Apply the 'gap' to subsample the data, reducing overlap and dataset size.
        return windows[::self.gap], sequence_labels[::self.gap], sequence_times[::self.gap]
    
    def __len__(self) -> int:
        """Returns the total number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a single sequence and its corresponding label.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple: A tuple containing the feature tensor and the target tensor.
        """
        features = torch.from_numpy(self.sequences[idx])
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, target