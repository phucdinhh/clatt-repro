"""Stage 7.1 (--tuned, reproducibility, freeze) and stage 8.1 (ablation switches)."""
import json

import pandas as pd
import pytest
import torch
import yaml

from src import freeze
from src.data import ROOT
from src.dataset import T
from src.features import pairs
from src.models import TABLE1, build, n_params
from src.run import ABLATIONS
from src.train import load_config, train_one, tuned_path

PAIR = "sz.300015_sz.300274"
HP = ("hidden", "filters", "dropout", "batch_size", "lr", "max_epochs")


@pytest.mark.parametrize("model", list(TABLE1))
@pytest.mark.parametrize("group", [1, 2])
def test_tuned_config_matches_yaml(model, group):
    tuned = yaml.safe_load(tuned_path(model, group).read_text())
    cfg = load_config(model, tuned_group=group)
    assert all(cfg[k] == tuned[k] for k in tuned if k != "name")
    assert cfg["tuned_from"] == f"configs/tuned/{model.lower().replace('-', '_')}_g{group}.yaml"
    off = load_config(model)
    assert "tuned_from" not in off
    assert all(off[k] == v for k, v in TABLE1[model].items())
    assert off["lr"] == yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())["lr"]


def test_run_cli_picks_group_of_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run.RESULTS", tmp_path)  # the CLI writes results/logs/<exp>.log
    seen = []
    monkeypatch.setattr("src.run.train_one", lambda m, pr, s, cfg, **kw: seen.append((pr, cfg)) or
                        {"val": {"mse": 0, "skill": 0}, "train_seconds": 0})
    from src.run import main
    g2 = next(f"{a}_{b}" for g, a, b in pairs() if g == 2)
    main(["--exp", "x", "--models", "CLATT", "--pairs", PAIR, g2, "--tuned"])
    assert [c["tuned_from"] for _, c in seen] == ["configs/tuned/clatt_g1.yaml", "configs/tuned/clatt_g2.yaml"]
    seen.clear()
    main(["--exp", "x", "--models", "CLATT", "--pairs", PAIR, "--ablation", "A1"])
    assert seen[0][1]["features"] == ["rho"] and seen[0][1]["hidden"] == TABLE1["CLATT"]["hidden"]


def test_train_one_deep_model_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RUNS", tmp_path)
    cfg = load_config("CNN-LSTM", ["max_epochs=2"])
    runs = []
    for _ in range(2):
        m = train_one("CNN-LSTM", PAIR, 0, cfg, exp="tmp", force=True)
        out = tmp_path / "tmp" / "CNN-LSTM" / PAIR / "seed0"
        runs.append((m["val"], m["best_epoch"], pd.read_csv(out / "preds_val.csv")))
    assert runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1]
    pd.testing.assert_frame_equal(runs[0][2], runs[1][2])


def test_final_force_cannot_reread_test(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RUNS", tmp_path)
    cfg = load_config("Persistence")
    train_one("Persistence", PAIR, 0, cfg, split_eval=("val", "test"), exp="tmp", final=True)
    assert train_one("Persistence", PAIR, 0, cfg, split_eval=("val", "test"), exp="tmp", final=True)["test"]
    with pytest.raises(ValueError, match="already read the test split"):
        train_one("Persistence", PAIR, 0, cfg, split_eval=("val", "test"), exp="tmp", final=True, force=True)


def test_freeze_detects_changes(tmp_path, monkeypatch):
    (tmp_path / "configs" / "models").mkdir(parents=True)
    a = tmp_path / "configs" / "base.yaml"
    a.write_text("T: 90\n")
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(freeze, "CONFIGS", tmp_path / "configs")
    monkeypatch.setattr(freeze, "MANIFEST", tmp_path / "configs" / "FROZEN.sha256")
    assert freeze.check()  # no manifest yet
    freeze.write()
    assert freeze.check() == []
    a.write_text("T: 68\n")
    (tmp_path / "configs" / "models" / "new.yaml").write_text("x: 1\n")
    assert freeze.check() == ["changed: configs/base.yaml", "added: configs/models/new.yaml"]


def test_repo_configs_match_freeze():
    """After stage 7.0 nothing under configs/ may change (the manifest is written once, before --final)."""
    if not freeze.MANIFEST.exists():
        pytest.skip("configs not frozen yet")
    assert freeze.check() == []


# ---- stage 8.1 ablation switches -------------------------------------------------------------------------

@pytest.mark.parametrize("head", ["attention", "last", "mean"])
def test_heads_shape(head):
    m = build("CLATT", head=head).eval()
    y, a = m(torch.randn(4, T, 3), return_attention=True)
    assert y.shape == (4, 1)
    assert (a is None) == (head != "attention")


def test_default_heads_unchanged():
    assert build("CLATT").head == "attention" and build("CNN-BiLSTM").head == "last"
    assert build("LSTM").head == "last" and build("CNN-LSTM").attn is None


def test_a2_is_clatt_without_attention_params():
    d = 2 * TABLE1["CLATT"]["hidden"]
    assert n_params(build("CLATT")) - n_params(build("CLATT", head="last")) == d * d + d


def test_a3_no_conv():
    m = build("CLATT", filters=None)
    assert m.conv is None and m.rnn.input_size == 3 and m(torch.randn(2, T, 3)).shape == (2, 1)


def test_a4_mean_is_uniform_pooling():
    m = build("CLATT", head="mean").eval()
    x = torch.randn(3, T, 3)
    with torch.no_grad():
        seq, _ = m.rnn(m.conv(x.transpose(1, 2)).transpose(1, 2))
        assert torch.allclose(m(x), m.out(seq.mean(1)))


def test_unknown_head_raises():
    with pytest.raises(ValueError):
        build("CLATT", head="max")


def test_ablation_file():
    assert set(ABLATIONS) == {"A1", "A2", "A3", "A4"}
    cfgs = {k: load_config("CLATT", v["overrides"]) for k, v in ABLATIONS.items()}
    assert cfgs["A1"]["features"] == ["rho"] and cfgs["A2"]["head"] == "last"
    assert cfgs["A3"]["filters"] is None and cfgs["A4"]["head"] == "mean"
    base = load_config("CLATT")
    for cfg in cfgs.values():  # exactly one switch per variant
        assert len({k for k in cfg if cfg[k] != base.get(k, "<missing>")}) == 1


def test_a1_trains_on_rho_only(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RUNS", tmp_path)
    cfg = load_config("CLATT", ABLATIONS["A1"]["overrides"] + ["max_epochs=1"])
    m = train_one("CLATT", PAIR, 0, cfg, exp="tmp")
    assert m["n_params"] == n_params(build("CLATT", n_features=1))
    saved = yaml.safe_load((tmp_path / "tmp" / "CLATT" / PAIR / "seed0" / "config.yaml").read_text())
    assert saved["features"] == ["rho"]
    assert json.loads((tmp_path / "tmp" / "CLATT" / PAIR / "seed0" / "metrics.json").read_text())["val"]["mse"] > 0
