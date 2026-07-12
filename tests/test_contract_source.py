from pathlib import Path


def test_model_has_no_target_derived_nominal_or_solver():
    text = (Path(__file__).parents[1] / "gridfm" / "model.py").read_text().lower()
    assert "v_nominal" not in text
    assert "linalg.solve" not in text
    assert "opendss" not in text


def test_config_uses_real_topology_holdout():
    import yaml

    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "pf_fraction.yaml").read_text())
    assert 0 < cfg["data"]["train_frac"] < 1
    assert cfg["data"]["val_frac"] > 0
    assert cfg["data"]["cast_float32"] is False

