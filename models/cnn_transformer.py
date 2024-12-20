# models/cnn_transformer.py

import torch
import torch.nn as nn

class CNNTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, sample_input, target_height, target_width, mode='simple'):
        """
        Initializes the CNNTransformer model.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            sample_input (torch.Tensor): A sample input tensor to infer CNN output size.
            target_height (int): Height of the target output.
            target_width (int): Width of the target output.
            mode (str): 'simple' or 'complex' model architecture.
        """
        super(CNNTransformer, self).__init__()
        assert mode in ['simple', 'complex'], "Mode must be 'simple' or 'complex'"
        self.mode = mode
        
        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.MaxPool2d(2),
        )
        
        if self.mode == 'complex':
            self.cnn.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.cnn.add_module('gn3', nn.GroupNorm(8, 128))
            self.cnn.add_module('relu3', nn.ReLU())
            self.cnn.add_module('dropout3', nn.Dropout(0.3))
            self.cnn.add_module('maxpool2', nn.MaxPool2d(2))
        
        # Compute CNN output size
        with torch.no_grad():
            dummy_input = sample_input.unsqueeze(0)  # Add batch dimension
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        
        # Define fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.cnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Define Transformer Encoder
        if self.mode == 'simple':
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.1), num_layers=2
            )
        elif self.mode == 'complex':
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.2), num_layers=4
            )
            self.fc1 = nn.Sequential(
                nn.Linear(self.cnn_output_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
        
        # Define output layer
        transformer_output_dim = 128 if self.mode == 'simple' else 256
        self.fc2 = nn.Linear(transformer_output_dim, output_channels * target_height * target_width)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_length=1, feature_size)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc2(x)
        return x
