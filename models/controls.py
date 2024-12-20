import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
import numpy as np

# =============================================================
# Model starter code; fine-tuned as needed. Simple models built for flexibility.
# =============================================================

class RNNModel(nn.Module):
    def __init__(self, input_channels, output_channels, sample_input, mode='base'):
        super(RNNModel, self).__init__()
        self.input_size = input_channels

        if mode == 'base':
            self.hidden_size = 64
            self.num_layers = 1
        elif mode == 'extended':
            self.hidden_size = 128
            self.num_layers = 2
        else:
            self.hidden_size = 256
            self.num_layers = 3

        if mode in ['standard', 'extended']:
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(
                self.hidden_size,
                output_channels
            )
        else:
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(
                self.hidden_size * sample_input.shape[1] * sample_input.shape[2],
                output_channels * sample_input.shape[1] * sample_input.shape[2]
            )

    def forward(self, x):
        batch_size = x.size(0)
        if self.mode in ['standard', 'extended']:
            # Shape: (batch_size, seq_length, channels, height, width)
            x = x.permute(0, 1, 3, 4, 2)
            # Shape: (batch_size, seq_length, input_size)
            x = x.reshape(batch_size, x.size(1), -1)
            out, _ = self.rnn(x)
            out = self.fc(out[:, -1, :])
        else:
            # Shape: (batch_size, height, width, channels)
            x = x.permute(0, 2, 3, 1)
            # Shape: (batch_size, seq_length, input_size)
            x = x.reshape(batch_size, -1, self.input_size)
            out, _ = self.rnn(x)
            # Flatten output
            out = out.reshape(batch_size, -1)
            out = self.fc(out)
        return out


class MLPModel(nn.Module):
    def __init__(self, input_channels, output_channels, sample_input, mode='base'):
        super(MLPModel, self).__init__()
        self.input_dim = input_channels * sample_input.shape[1] * sample_input.shape[2]

        if mode == 'base':
            hidden_layers = [128]
            self.flatten = True
        elif mode == 'extended':
            hidden_layers = [256, 128]
            self.flatten = True
        else:
            hidden_layers = [512, 256]
            self.flatten = True

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        if mode in ['standard', 'extended']:
            self.fc = nn.Sequential(*layers, nn.Linear(in_dim, output_channels))
        else:
            layers.append(
                nn.Linear(
                    in_dim,
                    output_channels * sample_input.shape[1] * sample_input.shape[2]
                )
            )
            self.fc = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        if self.mode in ['standard', 'extended']:
            # Shape: (batch_size, seq_length, channels, height, width)
            x = x.permute(0, 1, 3, 4, 2)
            # Flatten sequence and spatial dimensions
            x = x.reshape(batch_size, x.size(1), -1)
            # Take the last time step
            x = self.fc(x[:, -1, :])
        else:
            # Flatten input
            x = x.view(batch_size, -1)
            x = self.fc(x)
        return x


class PureTransformerModel(nn.Module):
    def __init__(self, input_channels, output_channels, sample_input, mode='base'):
        super(PureTransformerModel, self).__init__()
        self.input_size = input_channels * sample_input.shape[1] * sample_input.shape[2]

        if mode == 'base':
            hidden_dim = 128
            num_layers = 2
            num_heads = 4
        elif mode == 'extended':
            hidden_dim = 256
            num_layers = 4
            num_heads = 8
        else:
            hidden_dim = 512
            num_layers = 6
            num_heads = 16

        if mode in ['standard', 'extended']:
            self.fc1 = nn.Linear(self.input_size, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.1
                ),
                num_layers=num_layers
            )
            self.fc2 = nn.Linear(
                hidden_dim,
                output_channels
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(self.input_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.1
                ),
                num_layers=num_layers
            )
            self.fc2 = nn.Linear(
                hidden_dim,
                output_channels * sample_input.shape[1] * sample_input.shape[2]
            )

    def forward(self, x):
        batch_size = x.size(0)
        if self.mode in ['standard', 'extended']:
            # Shape: (batch_size, seq_length, channels, height, width)
            x = x.permute(0, 1, 3, 4, 2)
            # Flatten spatial dimensions
            x = x.reshape(batch_size, x.size(1), -1)
            x = self.fc1(x)
            x = self.transformer(x)
            x = self.fc2(x[:, -1, :])
        else:
            # Flatten input
            x = x.view(batch_size, -1)
            x = self.fc1(x)
            # Shape: (batch_size, 1, hidden_dim)
            x = x.unsqueeze(1)
            x = self.transformer(x)
            # Shape: (batch_size, hidden_dim)
            x = x.squeeze(1)
            x = self.fc2(x)
        return x


class LinearRegressionModel:
    def __init__(self, mode='base'):
        self.model = LinearRegression()
        self.mode = mode

    def fit(self, X, y):
        if self.mode in ['standard', 'extended']:
            X = X.reshape(X.size(0), -1)
        else:
            X = X.view(X.size(0), -1).numpy()
        y = y.view(y.size(0), -1).numpy()
        self.model.fit(X, y)

    def predict(self, X):
        if self.mode in ['standard', 'extended']:
            X = X.reshape(X.size(0), -1)
        else:
            X = X.view(X.size(0), -1).numpy()
        return self.model.predict(X)


# =========================
# Training and Evaluation Functions
# =========================

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.6f}")
    return epoch_loss


def evaluate_model(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss


def train_and_evaluate_multiple_times(
    model_constructor,
    num_trials,
    train_loader,
    test_loader,
    criterion,
    create_optimizer,
    num_epochs,
    device
):
    losses = []
    for trial in range(num_trials):
        model = model_constructor()
        optimizer = create_optimizer(model)
        train_loss = train_model(
            model, train_loader, criterion, optimizer, num_epochs, device
        )
        test_loss = evaluate_model(model, test_loader, criterion, device)
        losses.append(test_loss)
        print(f"Trial {trial + 1}/{num_trials}, Test Loss: {test_loss:.6f}")
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    print(f"Mean Test Loss: {mean_loss:.6f}, Std Dev: {std_loss:.6f}")
    return mean_loss, std_loss
