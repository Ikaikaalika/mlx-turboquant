from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib
from pathlib import Path
import shutil
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import mlx.core as mx
from mlx.utils import tree_flatten

from .core import TURBOQUANT_CONFIG_KEY, load_turboquant_weights, save_turboquant_weights

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.tuner.utils import dequantize as mlx_lm_dequantize
    from mlx_lm.tuner.utils import load_adapters as mlx_lm_load_adapters
    from mlx_lm.utils import _download as mlx_lm_download
    from mlx_lm.utils import _get_classes as mlx_lm_get_classes
    from mlx_lm.utils import load_config as mlx_lm_load_config
    from mlx_lm.utils import load_tokenizer as mlx_lm_load_tokenizer
except Exception:  # pragma: no cover - optional dependency in tests
    mlx_lm_load = None
    mlx_lm_dequantize = None
    mlx_lm_load_adapters = None
    mlx_lm_download = None
    mlx_lm_get_classes = None
    mlx_lm_load_config = None
    mlx_lm_load_tokenizer = None


def _ensure_mlx_lm_available() -> None:
    if (
        mlx_lm_load is None
        or mlx_lm_dequantize is None
        or mlx_lm_load_adapters is None
        or mlx_lm_download is None
        or mlx_lm_get_classes is None
        or mlx_lm_load_config is None
        or mlx_lm_load_tokenizer is None
    ):
        raise RuntimeError(
            "mlx_lm is not available. Install optional dependencies to use mlx_lm weight integration."
        )


def _resolve_model_path(path_or_hf_repo: str, revision: Optional[str]) -> Path:
    _ensure_mlx_lm_available()
    path = Path(path_or_hf_repo)
    if path.exists():
        return path
    return Path(mlx_lm_download(path_or_hf_repo, revision=revision))


def _copy_support_files(src_path: Path, dst_path: Path) -> None:
    for pattern in ("*.py", "generation_config.json"):
        for file in src_path.glob(pattern):
            shutil.copy(file, dst_path / file.name)


def convert_turboquant_mlx_lm_model(
    path_or_hf_repo: str,
    output_path: str | Path,
    *,
    bit_width: int = 4,
    algorithm: str = "mse",
    seed: int = 0,
    pack: bool = True,
    revision: Optional[str] = None,
    max_file_size_gb: int = 5,
    tokenizer_config: Optional[dict[str, Any]] = None,
    model_config: Optional[dict[str, Any]] = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """
    Convert any mlx-lm model into TurboQuant-compressed weight artifacts.
    """
    _ensure_mlx_lm_available()

    output_dir = Path(output_path)
    if output_dir.exists():
        raise ValueError(
            f"Cannot convert into {output_dir} because it already exists. "
            "Choose an empty destination path."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer_cfg = dict(tokenizer_config or {})
    if trust_remote_code:
        tokenizer_cfg["trust_remote_code"] = True
    model_cfg = dict(model_config or {})

    model, tokenizer, config = mlx_lm_load(
        path_or_hf_repo,
        tokenizer_config=tokenizer_cfg,
        model_config=model_cfg,
        lazy=True,
        return_config=True,
        revision=revision,
    )

    model = mlx_lm_dequantize(model)

    quantized = save_turboquant_weights(
        model.parameters(),
        output_dir,
        bit_width=bit_width,
        seed=seed,
        pack=pack,
        algorithm=algorithm,
        max_file_size_gb=max_file_size_gb,
        config=config,
    )

    tokenizer.save_pretrained(output_dir)
    source_path = _resolve_model_path(path_or_hf_repo, revision=revision)
    _copy_support_files(source_path, output_dir)

    return {
        "output_path": str(output_dir),
        "algorithm": algorithm,
        "bit_width": bit_width,
        "pack": bool(pack),
        "seed": int(seed),
        "quantized_tensors": int(quantized.stats.quantized_tensors),
        "skipped_tensors": int(quantized.stats.skipped_tensors),
        "original_bytes": int(quantized.stats.original_bytes),
        "quantized_bytes": int(quantized.stats.quantized_bytes),
        "compression_ratio": float(quantized.stats.compression_ratio),
    }


def _load_turboquant_model_from_path(
    model_path: Path,
    *,
    tokenizer_config: Optional[dict[str, Any]] = None,
    model_config: Optional[dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    _ensure_mlx_lm_available()

    config = mlx_lm_load_config(model_path)
    merged_config = copy.deepcopy(config)
    merged_config.update(model_config or {})

    model_class, model_args_class = mlx_lm_get_classes(config=merged_config)
    model_args = model_args_class.from_dict(merged_config)
    model = model_class(model_args)

    dense_weights = load_turboquant_weights(model_path)

    expected_paths = {path for path, _ in tree_flatten(model.parameters())}
    actual_paths = {path for path, _ in tree_flatten(dense_weights)}
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            "TurboQuant weight tree does not match model parameters. "
            f"Missing paths: {missing[:5]}, extra paths: {extra[:5]}"
        )

    model.update(dense_weights)

    if adapter_path is not None:
        model = mlx_lm_load_adapters(model, adapter_path)
    if not lazy:
        mx.eval(model.parameters())
    model.eval()

    tokenizer = mlx_lm_load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=merged_config.get("eos_token_id", None),
    )
    return model, tokenizer, merged_config


def load_turboquant_mlx_lm(
    path_or_hf_repo: str,
    tokenizer_config: Optional[dict[str, Any]] = None,
    model_config: Optional[dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: Optional[str] = None,
    _fallback_load_fn: Optional[Callable[..., Any]] = None,
) -> Union[
    Tuple[Any, Any],
    Tuple[Any, Any, Dict[str, Any]],
]:
    """
    Load an mlx-lm model, auto-detecting TurboQuant artifacts when present.
    """
    _ensure_mlx_lm_available()

    tokenizer_cfg = dict(tokenizer_config or {})
    model_cfg = dict(model_config or {})

    model_path = _resolve_model_path(path_or_hf_repo, revision=revision)
    config = mlx_lm_load_config(model_path)
    if TURBOQUANT_CONFIG_KEY not in config:
        fallback = _fallback_load_fn or mlx_lm_load
        return fallback(
            path_or_hf_repo,
            tokenizer_config=tokenizer_cfg,
            model_config=model_cfg,
            adapter_path=adapter_path,
            lazy=lazy,
            return_config=return_config,
            revision=revision,
        )

    model, tokenizer, merged_config = _load_turboquant_model_from_path(
        model_path,
        tokenizer_config=tokenizer_cfg,
        model_config=model_cfg,
        adapter_path=adapter_path,
        lazy=lazy,
    )

    if return_config:
        return model, tokenizer, merged_config
    return model, tokenizer


class MLXLMTurboQuantWeightPatcher:
    """
    Monkey patch mlx_lm load entry points to support TurboQuant artifacts.
    """

    def __init__(self):
        self._originals: dict[tuple[str, str], Any] = {}
        self._applied = False

    def _patch_attr(self, module: ModuleType, attr: str, value: Any) -> None:
        key = (module.__name__, attr)
        if key not in self._originals:
            self._originals[key] = getattr(module, attr)
        setattr(module, attr, value)

    def apply(self) -> "MLXLMTurboQuantWeightPatcher":
        if self._applied:
            return self

        _ensure_mlx_lm_available()
        root_module = importlib.import_module("mlx_lm")
        utils_module = importlib.import_module("mlx_lm.utils")

        root_load = root_module.load
        utils_load = utils_module.load

        def patched_root_load(
            path_or_hf_repo: str,
            tokenizer_config: Optional[dict[str, Any]] = None,
            model_config: Optional[dict[str, Any]] = None,
            adapter_path: Optional[str] = None,
            lazy: bool = False,
            return_config: bool = False,
            revision: Optional[str] = None,
        ):
            return load_turboquant_mlx_lm(
                path_or_hf_repo,
                tokenizer_config=tokenizer_config,
                model_config=model_config,
                adapter_path=adapter_path,
                lazy=lazy,
                return_config=return_config,
                revision=revision,
                _fallback_load_fn=root_load,
            )

        def patched_utils_load(
            path_or_hf_repo: str,
            tokenizer_config: Optional[dict[str, Any]] = None,
            model_config: Optional[dict[str, Any]] = None,
            adapter_path: Optional[str] = None,
            lazy: bool = False,
            return_config: bool = False,
            revision: Optional[str] = None,
        ):
            return load_turboquant_mlx_lm(
                path_or_hf_repo,
                tokenizer_config=tokenizer_config,
                model_config=model_config,
                adapter_path=adapter_path,
                lazy=lazy,
                return_config=return_config,
                revision=revision,
                _fallback_load_fn=utils_load,
            )

        self._patch_attr(root_module, "load", patched_root_load)
        self._patch_attr(utils_module, "load", patched_utils_load)

        self._applied = True
        return self

    def restore(self) -> None:
        if not self._applied:
            return

        for (module_name, attr), original in self._originals.items():
            module = importlib.import_module(module_name)
            setattr(module, attr, original)
        self._originals.clear()
        self._applied = False


@contextmanager
def turboquantize_mlx_lm_weights() -> Iterator[MLXLMTurboQuantWeightPatcher]:
    patcher = MLXLMTurboQuantWeightPatcher().apply()
    try:
        yield patcher
    finally:
        patcher.restore()


def patch_mlx_lm_weights() -> MLXLMTurboQuantWeightPatcher:
    return MLXLMTurboQuantWeightPatcher().apply()


__all__ = [
    "convert_turboquant_mlx_lm_model",
    "load_turboquant_mlx_lm",
    "MLXLMTurboQuantWeightPatcher",
    "patch_mlx_lm_weights",
    "turboquantize_mlx_lm_weights",
]

