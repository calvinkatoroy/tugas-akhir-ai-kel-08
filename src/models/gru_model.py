import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(
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
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.classifier(last)


def build_gru(cfg, n_features):
    return GRUClassifier(
        n_features=n_features,
        hidden_size=cfg['gru']['hidden_size'],
        num_layers=cfg['gru']['num_layers'],
        dropout=cfg['gru']['dropout'],
    )
