import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    
    This version is modified to be compatible with batch_first=True.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension and register it as a buffer
        # Shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor
        # self.pe is sliced to the sequence length of the input x
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    An enriched Transformer model for sequence classification/regression.
    
    This version incorporates several modern enhancements:
    1.  Input Normalization: LayerNorm is applied to the input features first.
    2.  CLS Token: A learnable [CLS] token is prepended to the sequence. The final
        representation of this token is used for classification, which is often
        more effective than averaging all token outputs.
    3.  Pre-Layer Normalization: LayerNorm is applied *before* the self-attention
        and feed-forward layers, which leads to more stable training.
    4.  GeLU Activation: Uses the GeLU activation function, which often outperforms ReLU.
    5.  MLP Head: A multi-layer perceptron (MLP) is used as the final classification
        head for increased expressive power.
    """
    def __init__(self, num_features=28, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        # --- Architecture Components ---
        
        # 1. Input feature projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. Learnable CLS token
        # This token will be prepended to the sequence and its final hidden state will be used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 3. Positional Encoding (using the batch_first-compatible version)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 4. Transformer Encoder Layer with Pre-LN and GeLU
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,   # Input format is (batch, seq, feature)
            norm_first=True,    # Pre-Layer Normalization
            activation='gelu'   # GeLU activation function
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 5. Input Normalization
        self.input_normalization = nn.LayerNorm(num_features)
        self.d_model = d_model
        
        # 6. Richer MLP Decoder Head
        # Takes the final representation of the [CLS] token for classification
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.init_weights()

    def init_weights(self):
        """Initializes weights for the model."""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        # Initialize decoder weights
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)

    def forward(self, src):
        """
        Forward pass of the model.
        Args:
            src: Tensor, shape [batch_size, seq_len, num_features]
        """
        # 1. Normalize input features
        src = self.input_normalization(src)
        
        # 2. Project input features to d_model and scale
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # 3. Prepend [CLS] token to each sequence in the batch
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        # 4. Add positional encoding
        src = self.pos_encoder(src)
        
        # 5. Pass through the transformer encoder
        output = self.transformer_encoder(src)
        
        # 6. Extract the output of the [CLS] token (it's the first token)
        cls_output = output[:, 0, :]
        
        # 7. Pass the [CLS] token's representation through the MLP head
        final_output = self.decoder(cls_output)
        
        return final_output.squeeze(-1)

    def prediction(self, src):
        """
        Takes the raw logit output from the forward pass, converts it to a
        probability using the sigmoid function, and then thresholds it at 0.5
        to return a binary prediction (0 or 1).
        """
        # Get the raw scores (logits) from the forward pass
        output = self.forward(src)
        # Apply the sigmoid function to convert logits to probabilities
        probabilities = torch.sigmoid(output)
        # Threshold the probabilities at 0.5 to get the final binary class
        return (probabilities > 0.5).int()
