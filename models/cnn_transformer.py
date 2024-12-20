import torch
import torch.nn as nn

# simple/flexible starter code for CNN-Transformer model
class CNNTransformer(nn.Module):
    def __init__(self, a, b, c, d, e, f, g='base'):
        """
        Initializes the CNNTransformer model, with cube/data dims as flexible arguments.
        Can be extended if extra compute available & modified as necessary.
        """
        super(CNNTransformer, self).__init__()
        self.g = g
        self.d = d
        self.e = e
        self.f = f

        if self.g in ['standard', 'extended']:
            self.a1 = nn.Sequential(
                nn.Conv3d(a, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool3d(2),
            )
            if self.g == 'standard':
                self.a1.add_module('c3', nn.Conv3d(64, 128, kernel_size=3, padding=1))
                self.a1.add_module('bn3', nn.BatchNorm3d(128))
                self.a1.add_module('r3', nn.ReLU())
                self.a1.add_module('d3', nn.Dropout(0.3))
                self.a1.add_module('mp2', nn.MaxPool3d(2))
                self.a1.add_module('c4', nn.Conv3d(128, 256, kernel_size=3, padding=1))
                self.a1.add_module('bn4', nn.BatchNorm3d(256))
                self.a1.add_module('r4', nn.ReLU())
                self.a1.add_module('d4', nn.Dropout(0.4))
                
            elif self.g == 'extended':
                self.a1.add_module('c3', nn.Conv3d(64, 128, kernel_size=3, padding=1))
                self.a1.add_module('bn3', nn.BatchNorm3d(128))
                self.a1.add_module('r3', nn.ReLU())
                self.a1.add_module('d3', nn.Dropout(0.3))
                self.a1.add_module('mp2', nn.MaxPool3d(2))
                self.a1.add_module('c4', nn.Conv3d(128, 256, kernel_size=3, padding=1))
                self.a1.add_module('bn4', nn.BatchNorm3d(256))
                self.a1.add_module('r4', nn.ReLU())
                self.a1.add_module('d4', nn.Dropout(0.4))
        else:
            self.a1 = nn.Sequential(
                nn.Conv2d(a, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.GroupNorm(4, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(2),
            )
            if self.g == 'extended':
                self.a1.add_module('c3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
                self.a1.add_module('gn3', nn.GroupNorm(8, 128))
                self.a1.add_module('r3', nn.ReLU())
                self.a1.add_module('d3', nn.Dropout(0.3))
                self.a1.add_module('mp2', nn.MaxPool2d(2))

        with torch.no_grad():
            assert c.dim() in [4, 5], "flexible input must comply with architecture."
            tmp = c.unsqueeze(0) 
            out = self.a1(tmp)
            self.s = out.view(out.size(0), -1).size(1)

        z = 256 if self.g in ['standard', 'extended'] else 128 if self.g == 'base' else 256
        self.fc1 = nn.Sequential(
            nn.Linear(self.s, z),
            nn.ReLU(),
            nn.Dropout(0.3 if self.g in ['standard', 'extended'] else 0.2),
        )

        if self.g == 'base':
            self.t = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.1), num_layers=2
            )
        elif self.g == 'extended':
            self.t = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.2), num_layers=6
            )
        elif self.g == 'standard':
            self.t = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.3), num_layers=8
            )

        if self.g in ['standard', 'extended']:
            self.fc2 = nn.Linear(z, b * d * e * f)
        else:
            self.fc2 = nn.Linear(z, b * e * f)

    def forward(self, x):
        if self.g in ['standard', 'extended']:
            bsz, sl, C, D, H, W = x.size()
            x = x.view(bsz * sl, C, D, H, W)
            x = self.a1(x)
            x = x.view(bsz, sl, -1)
            x = self.fc1(x)
            x = self.t(x)
            x = self.fc2(x[:, -1, :])
            x = x.view(bsz, -1, self.d, self.e, self.f)
        else:
            bsz = x.size(0)
            x = self.a1(x)
            x = x.view(bsz, -1)
            x = self.fc1(x)
            x = self.t(x.unsqueeze(1))
            x = self.fc2(x.squeeze(1))
            x = x.view(bsz, -1, self.e, self.f)
        return x
