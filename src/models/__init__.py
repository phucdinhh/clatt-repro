"""The five deep models of the paper, built from Table 1 hyperparameters."""
import yaml

from src.data import ROOT
from src.models.cnn import CNN
from src.models.recurrent import ConvRNN

CONFIGS = ROOT / "configs" / "models"

MODELS = ["CNN", "LSTM", "CNN-LSTM", "CNN-BiLSTM", "CLATT"]
# [PAPER] Table 1, one YAML per model: hidden, filters, dropout, batch_size, max_epochs
TABLE1 = {m: yaml.safe_load((CONFIGS / f"{m.lower().replace('-', '_')}.yaml").read_text()) for m in MODELS}


def build(name, n_features=3, T=68, **hp):
    """`hp` overrides Table 1 values (hidden, filters, dropout) and may set `layers` (LSTM depth, default 2)."""
    p = {**TABLE1[name], **hp}
    if name == "CNN":
        return CNN(n_features, T, p["hidden"], p["filters"], p["dropout"])
    return ConvRNN(n_features, p["hidden"], p["filters"], p["dropout"],
                   bidirectional=name in ("CNN-BiLSTM", "CLATT"), attention=name == "CLATT",
                   layers=p.get("layers", 2))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
