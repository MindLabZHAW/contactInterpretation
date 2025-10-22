import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import os
import logging
from tqdm import tqdm

# --- Model Architectures ---

class Conv1DNet(nn.Module):
    def __init__(self, in_channels, embedding_dim=64, window_length=28, task='localization'):
        super(Conv1DNet, self).__init__()
        # --- Layer Definitions ---
        self.task = task
        self.normalization = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=4)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
        # --- Dynamic Size Calculation ---
        with torch.no_grad():
            self._dummy_input = torch.zeros(1, in_channels, window_length)
            self._flat_size = self._get_flat_size()
            #self.embedding_dim = self._get_flat_size()

        
        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self._flat_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def _get_flat_size(self):
        x = self.pool1(self.relu(self.conv1(self._dummy_input)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        return x.flatten(1).shape[1]

    def forward(self, x):
        # For 1D conv, shape should be (batch, channels, length)
        # Your data might be (batch, length, channels), so we permute
        '''if x.dim() == 3 and x.shape[1] != self.conv1.in_channels:
             x = x.permute(0, 2, 1)'''

        x = self.pool1(self.bn1(self.relu(self.conv1(x))))
        x = self.pool2(self.bn2(self.relu(self.conv2(x))))
        x = self.pool3(self.bn3(self.relu(self.conv3(x))))

        x = x.flatten(1)
        x = self.dropout(x)#self.relu(self.fc1(x)))
        x = self.tanh(self.fc1(x)) # Tanh for final embedding
        return x


class Classifier(nn.Module):
    def __init__(self, embedding_dim=64, num_classes=3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        # Softmax is handled by CrossEntropyLoss during training

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
# --- Wrapper Model for Evaluation Compatibility ---

class DMLClassificationNet(nn.Module):
    """
    A wrapper that combines a DML-trained embedding network and a classifier 
    into a single model compatible with the existing `run_evaluation.py` script.

    This model needs to mimic the attributes and methods expected by the Evaluator,
    such as the `.task` attribute and a `.prediction()` method.
    """
    def __init__(self, embedding_net, classifier):
        super(DMLClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = classifier
        
        # --- Compatibility Attributes ---
        self.task = embedding_net.task 
        self.domain_discriminator = False

    def forward(self, x):
        embedding = self.embedding_net(x)
        output_logits = self.classifier(embedding)
        return output_logits, embedding

    def prediction(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs inference and returns the final class prediction.
        """
        output_logits, _ = self.forward(input)
        
        if self.task == 'detection':
            # --- THIS IS THE FIX ---
            # Apply sigmoid to the single logit, check if > 0.5, and cast to int
            probabilities = torch.sigmoid(output_logits)
            #return (probabilities.squeeze(-1) > 0.5).int()
            return torch.argmax(probabilities, dim=1)

            # --- END FIX ---
        else: # 'localization'
            # This part is correct for localization
            return torch.argmax(output_logits, dim=1) + 1

'''
class Conv1DNet(nn.Module):
    def __init__(self, in_channels, embedding_dim=64, window_length=28, task='localization'):
        super(Conv1DNet, self).__init__()
        # --- Layer Definitions ---
        self.task = task
        self.normalization = nn.LayerNorm(in_channels)
        self.cnn1 = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1, padding='same', padding_mode='circular')
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, padding='same', padding_mode='circular')
        self.cnn3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=4, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # --- Dynamic Size Calculation ---
        with torch.no_grad():
            self._dummy_input = torch.zeros(1, in_channels, window_length)
            self._flat_size = self._get_flat_size()
        
        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self._flat_size, embedding_dim)


    def _get_flat_size(self):
        cnn_input = self.normalization(self._dummy_input).permute(0, 2, 1)
        cnn_out = self.relu(self.bn1(self.cnn1(cnn_input)))
        cnn_out = self.relu(self.bn2(self.cnn2(cnn_out)))
        cnn_out = self.relu(self.bn3(self.cnn3(cnn_out)))
        # Reshape back for pooling: (batch, features, seq_len) -> (batch, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out = self.pool(cnn_out)
        return cnn_out.flatten(1).shape[1]

    def forward(self, x):
        # For 1D conv, shape should be (batch, channels, length)
        # Your data might be (batch, length, channels), so we permute
        normalized_input = self.normalization(input)

        # 2. CNN Layers
        # Reshape for CNN: (batch, seq_len, features) -> (batch, features, seq_len)
        cnn_input = normalized_input.permute(0, 2, 1)
        cnn_out = self.relu(self.bn1(self.cnn1(cnn_input)))
        cnn_out = self.relu(self.bn2(self.cnn2(cnn_out)))
        cnn_out = self.relu(self.bn3(self.cnn3(cnn_out)))
        # Reshape back for pooling: (batch, features, seq_len) -> (batch, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out = self.pool(cnn_out)
        embedding_out = self.fc1(cnn_out)
        return embedding_out
'''