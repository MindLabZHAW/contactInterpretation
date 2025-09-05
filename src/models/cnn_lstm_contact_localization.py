
import torch
import torch.nn as nn


class cnnLSTM(nn.Module):
    def __init__(self, num_features_joints=28, hidden_size=32, num_layers=3, dropout=0.5, bidirectional=False):
        super(cnnLSTM, self).__init__()
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
        
        # Check if all elements are masked (-inf) for each sample
        nocontact_masked = (output <= 0).all(dim=1)  # True if all entries are masked

        # Find the index of the max value (ignoring masked ones)
        predictions = torch.argmax(output, dim=1)+1  # Get index of max valid value
        predictions[nocontact_masked] = 0
        return predictions
