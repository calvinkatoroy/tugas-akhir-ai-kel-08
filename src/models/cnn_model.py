import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, n_features, num_filters=64, kernel_size=3, dropout=0.3, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) → permute to (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_cnn(cfg, n_features):
    return CNNClassifier(
        n_features=n_features,
        num_filters=cfg['cnn']['num_filters'],
        kernel_size=cfg['cnn']['kernel_size'],
        dropout=cfg['cnn']['dropout'],
    )
