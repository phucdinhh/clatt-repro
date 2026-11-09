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
    """[Conv1D -> ReLU -> Dropout] -> (Bi)LSTM x layers -> Dropout -> head -> Dense(1).

    Covers LSTM (filters=None), CNN-LSTM, CNN-BiLSTM and CLATT (bidirectional + attention).
    head: "attention" (Eq. 8-9), "last" (final state of each direction) or "mean" (uniform mean over time).
    `attention=True` is the old spelling of head="attention". Ablations: A2 = CLATT with head="last",
    A3 = CLATT with filters=None, A4 = CLATT with head="mean".
    """

    HEADS = ("attention", "last", "mean")

    def __init__(self, n_features, hidden, filters, dropout, bidirectional=False, attention=False,
                 layers=2, kernel=3, head=None):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, filters, kernel, padding="same"), nn.ReLU(),
                                  nn.Dropout(dropout)) if filters else None
        self.rnn = nn.LSTM(filters or n_features, hidden, layers, batch_first=True, bidirectional=bidirectional)
        self.dirs = 2 if bidirectional else 1
        self.drop = nn.Dropout(dropout)
        self.head = head or ("attention" if attention else "last")
        if self.head not in self.HEADS:
            raise ValueError(f"unknown head {self.head!r}; choose from {self.HEADS}")
        self.attn = Attention(hidden * self.dirs) if self.head == "attention" else None
        self.out = nn.Linear(hidden * self.dirs, 1)

    def forward(self, x, return_attention=False):  # x: (B, T, F)
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        seq, (hn, _) = self.rnn(x)
        a = None
        if self.head == "last":
            # Keras return_sequences=False: final state of each direction (the backward one ends at t=0)
            y = self.out(self.drop(torch.cat(list(hn[-self.dirs:]), dim=-1)))
        elif self.head == "mean":
            y = self.out(self.drop(seq).mean(1))
        else:
            c, a = self.attn(self.drop(seq))
            y = self.out(c)
        return (y, a) if return_attention else y
