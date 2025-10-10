import torch
import torch.nn as nn
import math

# PositionalEncoding class remains the same
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerForLinkLocalization(nn.Module):
    """
    Transformer model adapted for token-level classification to predict contact
    on each robot link (DOF).
    """
    def __init__(self, num_features=28, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.5):
        super(TransformerForLinkLocalization, self).__init__()
        
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.input_normalization = nn.LayerNorm(num_features)
        self.d_model = d_model
        
        # The structure can remain the same, but its application in the forward pass changes.
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Outputs one score per token
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, num_features] where seq_len is the DOF.
            src_padding_mask: Tensor, shape [batch_size, seq_len] to ignore padded tokens.
        """
        src = self.input_normalization(src)
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)
        
        # Pass the ENTIRE output sequence to the decoder.
        # 'output' has shape [batch_size, seq_len, d_model]
        final_output = self.decoder(output)
        
        # final_output will have shape [batch_size, seq_len, 1]
        # .squeeze(-1) changes it to [batch_size, seq_len]
        return final_output.squeeze(-1)