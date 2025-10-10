import torch
import torch.nn as nn
import math

class cnnLSTM_contactDetection(nn.Module):
    def __init__(self, num_features_joints=28, hidden_size=32, num_layers=3, dropout=0.7, bidirectional=True):
        super(cnnLSTM_contactDetection, self).__init__()
        self.normalization = nn.LayerNorm(num_features_joints)#ZScoreNormalization() #nn.LayerNorm(num_features_joints)#
        
         # Define the 1D CNN layers
        self.cnn1 = nn.Conv1d(in_channels=num_features_joints, out_channels=256, kernel_size=1, padding='same',padding_mode='circular')
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, padding='same',padding_mode='circular')
        self.cnn3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=4, padding='same',padding_mode='circular')
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU() #nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=256//2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # Dropout only if num_layers > 1
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.normalization_lstm = nn.LayerNorm(lstm_output_size)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, 1)

    
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        self.lstm_out_calculation = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        # Normalize input (batch_size, sequence_length, input_size)
        self.lstm.flatten_parameters()

        normalized_input = self.normalization(input)

        # Reshape for CNN: (batch_size, in_channels, sequence_length)
        cnn_input = normalized_input.permute(0, 2, 1)
        
        # Apply CNN layers
        cnn_out = self.relu(self.bn1(self.cnn1(cnn_input)))#, seq_pose = 1))
        cnn_out = self.relu(self.bn2(self.cnn2(cnn_out)))#, seq_pose = 1))
        cnn_out = self.relu(self.bn3(self.cnn3(cnn_out)))#, seq_pose = 1))
        
        cnn_out = cnn_out.permute(0,2,1)
        cnn_out = self.pool(cnn_out)

        # Reshape for LSTM: (batch_size, sequence_l98pength, input_size)

        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.normalization_lstm(lstm_out)
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.lstm_out_calculation(lstm_out.permute(0, 2, 1)).squeeze(-1)
        # Fully connected layer
        
        joint_step_outputs = self.fc(lstm_out)  # (batch_size, 1, 1)

        # Squeeze to remove the last dimension
        joint_step_outputs = joint_step_outputs.squeeze(-1)  # Shape: (batch_size, 1)
        return joint_step_outputs
    
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


class cnnLSTM_contactLocalization(nn.Module):
    def __init__(self, num_features_joints=28, hidden_size=32, num_layers=3, dropout=0.7, bidirectional=True):
        super(cnnLSTM_contactLocalization, self).__init__()
        self.normalization = nn.LayerNorm(num_features_joints)#ZScoreNormalization()
        
         # Define the 1D CNN layers
        self.cnn1 = nn.Conv1d(in_channels=num_features_joints, out_channels=256, kernel_size=1, padding='same',padding_mode='circular')
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, padding='same',padding_mode='circular')
        self.cnn3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=4, padding='same',padding_mode='circular')
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.relu = nn.LeakyReLU() #nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.3)
        
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=256//2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # Dropout only if num_layers > 1
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.normalization_lstm = nn.LayerNorm(lstm_output_size)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, 1)

        # Dropout for regularization
        self.dropout_fc = nn.Dropout(dropout)
        
        

    def forward(self, input, freeze_last_layer=False):
        # Normalize input (batch_size, sequence_length, input_size)
        self.lstm.flatten_parameters()

        normalized_input = self.normalization(input)

        # Reshape for CNN: (batch_size, in_channels, sequence_length)
        cnn_input = normalized_input.permute(0, 2, 1)
        
        # Apply CNN layers
        cnn_out = self.relu(self.bn1(self.cnn1(cnn_input)))#, seq_pose = 1))
        cnn_out = self.relu(self.bn2(self.cnn2(cnn_out)))#, seq_pose = 1))
        cnn_out = self.relu(self.bn3(self.cnn3(cnn_out)))#, seq_pose = 1))
        cnn_out = cnn_out.permute(0,2,1)
        cnn_out = self.pool(cnn_out)

        # Reshape for LSTM: (batch_size, sequence_l98pength, input_size)

        lstm_out, _ = self.lstm(cnn_out)
        self.normalization_lstm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout_fc(lstm_out)
        
        # Fully connected layer

        joint_step_outputs = self.fc(lstm_out).squeeze(-1)  # (batch_size, joint_dof, 1) -> (batch_size, joint_dof)
        
        return joint_step_outputs
    
    def prediction(self, input):
        device = input.device
        output = self.forward(input)
        
        # Find the index of the max score, which corresponds to the predicted link
        # Add 1 because link numbers are typically 1-based, not 0-based
        predicted_link = torch.argmax(output, dim=1) + 1
        return predicted_link


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

    
class TransformerModel_contactDetection(nn.Module):
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
        super(TransformerModel_contactDetection, self).__init__()
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
    
    def prediction(self, input):
        device = input.device
        output = self.forward(input)
        
        # Find the index of the max score, which corresponds to the predicted link
        # Add 1 because link numbers are typically 1-based, not 0-based
        predicted_link = torch.argmax(output, dim=1) + 1
        return predicted_link