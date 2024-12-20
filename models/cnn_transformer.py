import torch
import torch.nn as nn

# flexible starter code for CNN-Transformer model
class CNNTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, sample_input, target_depth, target_height, target_width, mode='base'):
        """
        Initializes the CNNTransformer model, with cube/data dims as flexible arguments.
        """
        super(CNNTransformer, self).__init__()
        self.mode = mode
        self.target_depth = target_depth
        self.target_height = target_height
        self.target_width = target_width

        if self.mode == 'standard':
            # Define 3D CNN layers for cubes
            self.cnn = nn.Sequential(
                nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.MaxPool3d(2),  # Downsample by a factor of 2 in all dimensions
            )

            # Additional layers for 'standard' mode
            self.cnn.add_module('conv3', nn.Conv3d(64, 128, kernel_size=3, padding=1))
            self.cnn.add_module('bn3', nn.BatchNorm3d(128))
            self.cnn.add_module('relu3', nn.ReLU())
            self.cnn.add_module('dropout3', nn.Dropout(0.3))
            self.cnn.add_module('maxpool2', nn.MaxPool3d(2))

            self.cnn.add_module('conv4', nn.Conv3d(128, 256, kernel_size=3, padding=1))
            self.cnn.add_module('bn4', nn.BatchNorm3d(256))
            self.cnn.add_module('relu4', nn.ReLU())
            self.cnn.add_module('dropout4', nn.Dropout(0.4))

        else:
            # Define 2D CNN layers for images
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
            if self.mode == 'extended': # with extra compute
                self.cnn.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
                self.cnn.add_module('gn3', nn.GroupNorm(8, 128))
                self.cnn.add_module('relu3', nn.ReLU())
                self.cnn.add_module('dropout3', nn.Dropout(0.3))
                self.cnn.add_module('maxpool2', nn.MaxPool2d(2))

        # Compute CNN output size
        with torch.no_grad():
            assert sample_input.dim() in [4, 5], "flexible input must comply with architecture."
            sample_input = sample_input.unsqueeze(0) 
            cnn_output = self.cnn(sample_input)
            self.cnn_output_size = cnn_output.view(cnn_output.size(0), -1).size(1)

        # Define fully connected layers
        fc_size = 256 if self.mode == 'standard' else 128 if self.mode == 'base' else 256
        self.fc1 = nn.Sequential(
            nn.Linear(self.cnn_output_size, fc_size),
            nn.ReLU(),
            nn.Dropout(0.2 if self.mode == 'base' else 0.3 if self.mode == 'extended' else 0.4),
        )

        # Flexibility for ground-up testing
        if self.mode == 'base':
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.1), num_layers=2
            )
        elif self.mode == 'extended':
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.2), num_layers=6
            )
        elif self.mode == 'standard':
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.3), num_layers=8
            )

        if self.mode == 'standard':
            self.fc2 = nn.Linear(fc_size, output_channels * target_depth * target_height * target_width)
        else:
            self.fc2 = nn.Linear(fc_size, output_channels * target_height * target_width)

    def forward(self, x):
        batch_size = x.size(0)
        assert x.size(1) > 0, "Input tensor must have a positive channel dimension."
        assert batch_size > 0, "Batch size must be positive."

        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc2(x)

        if self.mode == 'standard':
            x = x.view(batch_size, -1, self.target_depth, self.target_height, self.target_width)
        else:
            x = x.view(batch_size, -1, self.target_height, self.target_width)

        return x
