import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import numpy as np

class TripletDataset(Dataset):
    """
    A dataset wrapper that returns triplets (anchor, positive, negative) for metric learning,
    PLUS the original class label of the anchor for classification learning.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = np.array([label.item() for _, label in self.dataset])
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor_data, anchor_label_tensor = self.dataset[index]
        anchor_label = anchor_label_tensor.item()
        
        # Find a positive sample (different sample, same class)
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.label_to_indices[anchor_label])
        positive_data, _ = self.dataset[positive_index]

        # Find a negative sample (different class)
        negative_label = random.choice(list(self.labels_set - {anchor_label}))
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_data, _ = self.dataset[negative_index]

        return anchor_data, positive_data, negative_data, anchor_label_tensor
