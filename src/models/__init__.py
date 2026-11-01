"""The five deep models of the paper, built from Table 1 hyperparameters."""
from src.models.cnn import CNN
from src.models.recurrent import ConvRNN

# [PAPER] Table 1: hidden units per direction [INFERENCE], conv filters, dropout, batch size, epochs
TABLE1 = {
    "CNN": dict(hidden=64, filters=31, dropout=0.1, batch=62, epochs=30),
    "LSTM": dict(hidden=32, filters=None, dropout=0.2, batch=64, epochs=30),
    "CNN-LSTM": dict(hidden=114, filters=32, dropout=0.2, batch=32, epochs=46),
    "CNN-BiLSTM": dict(hidden=126, filters=21, dropout=0.3, batch=23, epochs=50),
    "CLATT": dict(hidden=168, filters=77, dropout=0.3, batch=32, epochs=56),
}
MODELS = list(TABLE1)


def build(name, n_features=3, T=68):
    p = TABLE1[name]
    if name == "CNN":
        return CNN(n_features, T, p["hidden"], p["filters"], p["dropout"])
    return ConvRNN(n_features, p["hidden"], p["filters"], p["dropout"],
                   bidirectional=name in ("CNN-BiLSTM", "CLATT"), attention=name == "CLATT")


def n_params(model):
    return sum(p.numel() for p in model.parameters())
