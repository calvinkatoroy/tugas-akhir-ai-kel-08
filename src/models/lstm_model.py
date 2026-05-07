import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take last timestep
        return self.classifier(last)


def build_lstm(cfg, n_features):
    return LSTMClassifier(
        n_features=n_features,
        hidden_size=cfg['lstm']['hidden_size'],
        num_layers=cfg['lstm']['num_layers'],
        dropout=cfg['lstm']['dropout'],
    )
