from torch import nn


class CNN(nn.Module):
    """Conv1D -> ReLU -> Dropout -> Flatten -> Dense(hidden, ReLU) -> Dense(1)."""

    def __init__(self, n_features, T, hidden, filters, dropout, kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, filters, kernel, padding="same"), nn.ReLU(), nn.Dropout(dropout),
            nn.Flatten(), nn.Linear(filters * T, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, T, F)
        return self.net(x.transpose(1, 2))
