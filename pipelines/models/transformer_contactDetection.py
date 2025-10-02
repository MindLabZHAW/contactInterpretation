import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    A Transformer model for sequence classification/regression.
    
    This model replaces the CNN and LSTM layers with a Transformer Encoder architecture.
    It first projects the input features to a higher-dimensional space (embedding),
    adds positional encodings, and then processes the sequence through multiple
    Transformer encoder layers.
    """
    def __init__(self, num_features=28, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        # Input feature projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # Important: input format is (batch, seq, feature)
        )
        
        # Stacking encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Layer Normalization
        self.normalization = nn.LayerNorm(num_features)
        self.d_model = d_model
        
        # Output layer
        self.decoder = nn.Linear(d_model, 1)
        
        # Pooling for final output
        self.output_pooling = nn.AdaptiveAvgPool1d(1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Forward pass of the model.
        Args:
            src: Tensor, shape [batch_size, seq_len, num_features]
        """
        # 1. Normalize input features
        src = self.normalization(src)
        
        # 2. Project input features to d_model
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # 3. Add positional encoding
        # Note: PyTorch Transformer expects (seq_len, batch_size, d_model) if batch_first=False
        # but we are using batch_first=True, so we keep it as (batch_size, seq_len, d_model)
        # However, our PositionalEncoding is designed for (seq_len, batch, d_model). Let's adapt.
        # To make it compatible, we can permute, add encoding, and permute back.
        src = src.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # [batch_size, seq_len, d_model]
        
        # 4. Pass through the transformer encoder
        output = self.transformer_encoder(src) # Shape: [batch_size, seq_len, d_model]
        
        # 5. Pool the output over the sequence dimension
        # To use AdaptiveAvgPool1d, we need (batch_size, channels, length)
        # So we permute (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        output = output.permute(0, 2, 1)
        pooled_output = self.output_pooling(output).squeeze(-1) # Shape: [batch_size, d_model]

        # 6. Final linear layer
        output = self.decoder(pooled_output) # Shape: [batch_size, 1]
        
        return output.squeeze(-1) # Shape: [batch_size]

    def prediction(self, input):
        """
        Takes the raw logit output from the forward pass, converts it to a
        probability using the sigmoid function, and then thresholds it at 0.5
        to return a binary prediction (0 or 1).
        """
        # Get the raw scores (logits) from the forward pass
        output = self.forward(input)

        # Apply the sigmoid function to convert logits to probabilities
        probabilities = torch.sigmoid(output)

        # Threshold the probabilities at 0.5 to get the final binary class
        return (probabilities > 0.5).int()
