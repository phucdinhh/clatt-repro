import torch
from torch import nn


class Attention(nn.Module):
    """Paper Eq. 8-9: s_i = v^T tanh(W h_i), a = softmax(s), c = sum_i a_i h_i."""

    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, h):  # h: (B, T, d)
        a = torch.softmax(self.v(torch.tanh(self.W(h))).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * h).sum(1), a


class ConvRNN(nn.Module):
    """[Conv1D -> ReLU -> Dropout] -> (Bi)LSTM x layers -> Dropout -> last state or attention -> Dense(1).

    Covers LSTM (filters=None), CNN-LSTM, CNN-BiLSTM and CLATT (bidirectional + attention).
    """

    def __init__(self, n_features, hidden, filters, dropout, bidirectional=False, attention=False,
                 layers=2, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, filters, kernel, padding="same"), nn.ReLU(),
                                  nn.Dropout(dropout)) if filters else None
        self.rnn = nn.LSTM(filters or n_features, hidden, layers, batch_first=True, bidirectional=bidirectional)
        self.dirs = 2 if bidirectional else 1
        self.drop = nn.Dropout(dropout)
        self.attn = Attention(hidden * self.dirs) if attention else None
        self.out = nn.Linear(hidden * self.dirs, 1)

    def forward(self, x, return_attention=False):  # x: (B, T, F)
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        seq, (hn, _) = self.rnn(x)
        if self.attn is None:
            # Keras return_sequences=False: final state of each direction (the backward one ends at t=0)
            return self.out(self.drop(torch.cat(list(hn[-self.dirs:]), dim=-1)))
        c, a = self.attn(self.drop(seq))
        y = self.out(c)
        return (y, a) if return_attention else y
