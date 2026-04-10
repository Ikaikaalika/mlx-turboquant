import json

import mlx_lm
import pytest

import turboquant_mlx.mlx_lm_weights as weights_mod


def test_load_turboquant_mlx_lm_falls_back_when_model_is_not_turboquant(tmp_path):
    model_dir = tmp_path / "plain-model"
    model_dir.mkdir(parents=True)
    with open(model_dir / "config.json", "w") as f:
        json.dump({"model_type": "qwen2"}, f)

    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return "model", "tokenizer"

    result = weights_mod.load_turboquant_mlx_lm(
        str(model_dir),
        _fallback_load_fn=fallback,
    )

    assert result == ("model", "tokenizer")
    assert len(calls) == 1


def test_load_turboquant_mlx_lm_uses_turbo_loader_when_config_marked(tmp_path, monkeypatch):
    model_dir = tmp_path / "tq-model"
    model_dir.mkdir(parents=True)
    with open(model_dir / "config.json", "w") as f:
        json.dump(
            {
                "model_type": "qwen2",
                "turboquant": {
                    "format_version": 1,
                    "algorithm": "mse",
                    "bit_width": 4,
                    "pack": True,
                    "seed": 0,
                    "weight_index_filename": "turboquant-weights.index.json",
                    "passthrough_index_filename": "turboquant-passthrough.index.json",
                    "metadata_filename": "turboquant-metadata.json",
                },
            },
            f,
        )

    def fake_loader(*args, **kwargs):
        return "turbo-model", "turbo-tokenizer", {"ok": True}

    monkeypatch.setattr(weights_mod, "_load_turboquant_model_from_path", fake_loader)

    model, tokenizer, config = weights_mod.load_turboquant_mlx_lm(
        str(model_dir),
        return_config=True,
    )
    assert model == "turbo-model"
    assert tokenizer == "turbo-tokenizer"
    assert config == {"ok": True}


def test_mlx_lm_weight_patcher_apply_and_restore(monkeypatch):
    root_original = mlx_lm.load
    utils_module = __import__("mlx_lm.utils", fromlist=["load"])
    utils_original = utils_module.load

    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "patched-model", "patched-tokenizer"

    monkeypatch.setattr(weights_mod, "load_turboquant_mlx_lm", fake_load)

    patcher = weights_mod.MLXLMTurboQuantWeightPatcher().apply()
    try:
        model, tokenizer = mlx_lm.load("some-model")
        assert (model, tokenizer) == ("patched-model", "patched-tokenizer")
        assert calls
    finally:
        patcher.restore()

    assert mlx_lm.load is root_original
    assert utils_module.load is utils_original

