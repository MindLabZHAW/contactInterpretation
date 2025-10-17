import torch
import torch.nn as nn
import math
from torch.autograd import Function

# 1. Gradient Reversal Layer (GRL)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# 2. Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.5):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
    
class cnnBiLSTM(nn.Module):
    """
    A unified CNN-BiLSTM model for robot contact analysis.

    This model can be configured for two distinct tasks:
    - task='detection': A binary classification task to determine if any
      contact has occurred. It processes the entire time series and outputs a
      single prediction.
    - task='localization': A multi-class classification task to identify which
      robot link is in contact. It outputs a score for each time step in the
      sequence, corresponding to each link.
    """
    def __init__(self, num_features: int = 28, hidden_size: int = 32, num_layers: int = 3, 
                 dropout: float = 0.7, bidirectional: bool = True, task: str = 'detection', domain_discriminator:bool=False):
        """
        Initializes the layers of the CNN-BiLSTM model.

        Args:
            num_features (int): The number of input features per time step (e.g., joint errors).
            hidden_size (int): The number of features in the LSTM hidden state.
            num_layers (int): The number of recurrent layers in the LSTM.
            dropout (float): The dropout rate for regularization.
            bidirectional (bool): If True, the LSTM will be bidirectional.
            task (str): The operational mode, either 'detection' or 'localization'.
        """
        super(cnnBiLSTM, self).__init__()
        
        self.task = task
        self.domain_discriminator = domain_discriminator
        if self.task not in ['detection', 'localization']:
            raise ValueError("task must be either 'detection' or 'localization'")

        # --- Shared Layers ---
        # Normalization layer to stabilize inputs
        self.normalization = nn.LayerNorm(num_features)
        
        # 1D CNN layers to extract features from the time series
        self.cnn1 = nn.Conv1d(in_channels=num_features, out_channels=256, kernel_size=1, padding='same', padding_mode='circular')
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, padding='same', padding_mode='circular')
        self.cnn3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=4, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Bidirectional LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=256 // 2,  # Input features from the CNN backbone
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.normalization_lstm = nn.LayerNorm(lstm_output_size)
        self.dropout = nn.Dropout(dropout)
        
        # --- Task-Specific Components ---
        # For 'detection', we need to aggregate the time series information
        if self.task == 'detection':
            # Adaptive pooling takes the LSTM outputs and reduces them to a single vector
            self.last_pooling = nn.AdaptiveAvgPool1d(1)

        # Final fully-connected layer to produce the output score(s)
        self.fc = nn.Linear(lstm_output_size, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        # Flatten LSTM parameters for performance optimization with CUDA
        if hasattr(self.lstm, 'flatten_parameters'):
            self.lstm.flatten_parameters()
        
        # 1. Initial normalization
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

        # 3. LSTM Layers
        # Input shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.normalization_lstm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # 4. Task-Specific Output Calculation
        if self.task == 'detection':
            # Pool the sequence to a single vector for classification
            # Reshape for pooling: (batch, seq_len, features) -> (batch, features, seq_len)
            pooled_out = self.last_pooling(lstm_out.permute(0, 2, 1)).squeeze(-1)
            # Final prediction from the pooled output
            output = self.fc(pooled_out).squeeze(-1)
        else: # localization
            # Apply the FC layer to every token (time step) in the sequence
            # Output shape: (batch_size, seq_len)
            output = self.fc(lstm_out).squeeze(-1)
        
        return output

    def prediction(self, input: torch.Tensor) -> torch.Tensor:
        """Performs inference and returns the final prediction."""
        output = self.forward(input)
        if self.task == 'detection':
            # For detection, apply sigmoid and threshold for binary classification
            probabilities = torch.sigmoid(output)
            return (probabilities > 0.5).int()
        else: # localization
            # For localization, find the link with the highest score
            return torch.argmax(output, dim=1) + 1

# --- Transformer Helper Class ---
class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This helps the Transformer understand the order of the sequence, as it
    lacks inherent recurrence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension and register it as a non-trainable buffer
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor."""
        # Add the positional encoding up to the length of the input sequence
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Unified Transformer Model ---
class TransformerModel(nn.Module):
    """
    A unified Transformer model for contact detection or localization.
    
    - task='detection': Uses a special [CLS] token for a single, sequence-level
      prediction.
    - task='localization': Applies the final layers to every output token for
      link-specific predictions.
    """
    def __init__(self, num_features: int = 28, d_model: int = 128, nhead: int = 8, 
                 num_encoder_layers: int = 3, dim_feedforward: int = 0, dropout: float = 0.5,
                 task: str = 'detection', domain_discriminator: bool=False):
        """
        Initializes the layers of the Transformer model.

        Args:
            num_features (int): The number of input features per time step (e.g., joint errors).
            d_model (int): The internal embedding dimension of the Transformer layers.
            nhead (int): The number of parallel attention heads. d_model must be divisible by nhead.
            num_encoder_layers (int): The number of stacked Transformer blocks for complex representations.
            dim_feedforward (int): The dimension of the feed-forward network inside each Transformer block.
            dropout (float): The dropout rate for regularization to prevent overfitting.
            task (str): The operational mode, either 'detection' or 'localization'.
            DomainDiscriminator: for finetuning.
        """
        super(TransformerModel, self).__init__()
        self.task = task
        self.domain_discriminator = domain_discriminator
        if self.task not in ['detection', 'localization']:
            raise ValueError("task must be either 'detection' or 'localization'")

        # --- Shared Layers ---
        # Projects input features into the Transformer's embedding space (d_model)
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        if dim_feedforward ==0:
            dim_feedforward = d_model*4
        # Standard Transformer Encoder Layer with modern defaults (norm_first, gelu)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.input_normalization = nn.LayerNorm(num_features)
        self.d_model = d_model
        
        # --- Task-Specific Components ---
        # For 'detection', create a learnable [CLS] token
        if self.task == 'detection':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Final decoder head (MLP) to produce the output scores
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        # Adversarial Components ---
        if domain_discriminator:
            self.grl = GradientReversalLayer(alpha=1.0)
            self.domain_discriminator = DomainDiscriminator(input_dim=d_model)
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layers."""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)

    def forward(self, src: torch.Tensor,  src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Normalize and project input
        src = self.input_normalization(src)
        # Scale the embedding by sqrt(d_model) as is common in Transformers
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # 2. Task-Specific Forward Logic
        if self.task == 'detection':
            # Prepend the [CLS] token to the beginning of each sequence in the batch
            batch_size = src.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            src = torch.cat((cls_tokens, src), dim=1)
            
            # Add positional encoding
            src = self.pos_encoder(src)
            
            # Pass through the encoder
            output = self.transformer_encoder(src)
            
            # Use only the output of the [CLS] token for the final prediction
            final_features = output[:, 0, :]
            final_output = self.decoder(final_features).squeeze(-1)
        else: # localization
            # Add positional encoding to the entire sequence of links
            src = self.pos_encoder(src)
            
            # 'output' will have shape [batch_size, seq_len, d_model]
            final_features = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)
            
            # Apply the decoder to every token in the output sequence
            final_output = self.decoder(final_features).squeeze(-1)
        
        if self.domain_discriminator:
            if self.task == 'detection':
                domain_features = final_features
            else: # localization
                domain_features = torch.mean(final_features, dim=1)
            reversed_features = self.grl(domain_features)
            domain_output = self.domain_discriminator(reversed_features)
            return final_output, domain_output
        else:
            return final_output

    def prediction(self, src: torch.Tensor) -> torch.Tensor:
        """Performs inference and returns the final prediction."""
        output = self.forward(src)
        if self.domain_discriminator:
            output, _ = output
            
        if self.task == 'detection':
            probabilities = torch.sigmoid(output)
            return (probabilities > 0.5).int()
        else: # localization
            return torch.argmax(output, dim=1) + 1