#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models import cache as mlx_cache
from mlx_lm.sample_utils import make_sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_mlx.mlx_lm_integration import (
    TurboQuantBatchKVCache,
    TurboQuantChunkedKVCache,
    TurboQuantKVCache,
    make_turbo_prompt_cache,
)


COMPRESSED_CACHE_TYPES = (
    TurboQuantKVCache,
    TurboQuantBatchKVCache,
    TurboQuantChunkedKVCache,
)


@dataclass
class LayerMetrics:
    layer: int
    key_serialization_rel_l2: float
    value_serialization_rel_l2: float
    key_model_rel_l2: float
    value_model_rel_l2: float
    compression_ratio: float
    original_bytes: int
    quantized_bytes: int


def _to_int_token(token: Any) -> int:
    return int(np.asarray(token).reshape(()).item())


def _iter_leaf_caches(prompt_cache: Iterable[Any]) -> Iterable[Any]:
    for cache_obj in prompt_cache:
        if isinstance(cache_obj, mlx_cache.CacheList):
            for sub in cache_obj.caches:
                yield sub
        else:
            yield cache_obj


def _cache_kv_state(cache_obj: Any) -> Tuple[mx.array, mx.array]:
    state = cache_obj.state
    if not isinstance(state, tuple) or len(state) < 2:
        raise TypeError(f"Expected tuple state with keys/values, got: {type(state)!r}")
    return state[0], state[1]


def _relative_l2(a: mx.array, b: mx.array) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(aa)) + 1e-12
    return float(np.linalg.norm(aa - bb) / denom)


def _run_tokens(
    model: Any,
    prompt_tokens: mx.array,
    prompt_cache: List[Any],
    max_tokens: int,
) -> List[int]:
    sampler = make_sampler(temp=0.0)
    step = generate_step(
        prompt_tokens,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    )
    out: List[int] = []
    for token, _ in step:
        out.append(_to_int_token(token))
        if len(out) >= max_tokens:
            break
    return out


def _analyze_layers(
    baseline_cache: List[Any],
    turbo_cache: List[Any],
) -> tuple[List[LayerMetrics], bool]:
    baseline_layers = list(_iter_leaf_caches(baseline_cache))
    turbo_layers = list(_iter_leaf_caches(turbo_cache))
    if len(baseline_layers) != len(turbo_layers):
        raise ValueError(
            f"Layer count mismatch: baseline={len(baseline_layers)} turbo={len(turbo_layers)}"
        )

    dense_released = True
    metrics: List[LayerMetrics] = []
    for i, (base_layer, turbo_layer) in enumerate(zip(baseline_layers, turbo_layers)):
        if not isinstance(turbo_layer, COMPRESSED_CACHE_TYPES):
            continue
        if getattr(turbo_layer, "keys", None) is not None or getattr(turbo_layer, "values", None) is not None:
            dense_released = False

        base_keys, base_values = _cache_kv_state(base_layer)
        tq_keys, tq_values = turbo_layer._tq_decode_to_dense()
        restored_layer = turbo_layer.__class__.from_state(turbo_layer.state, turbo_layer.meta_state)
        restored_keys, restored_values = restored_layer._tq_decode_to_dense()
        key_serialization_rel_l2 = _relative_l2(restored_keys, tq_keys)
        value_serialization_rel_l2 = _relative_l2(restored_values, tq_values)
        key_model_rel_l2 = _relative_l2(base_keys, tq_keys)
        value_model_rel_l2 = _relative_l2(base_values, tq_values)
        stats = turbo_layer.last_turboquant_stats
        if stats is None:
            raise RuntimeError(f"No turboquant stats found for layer {i}.")

        metrics.append(
            LayerMetrics(
                layer=i,
                key_serialization_rel_l2=key_serialization_rel_l2,
                value_serialization_rel_l2=value_serialization_rel_l2,
                key_model_rel_l2=key_model_rel_l2,
                value_model_rel_l2=value_model_rel_l2,
                compression_ratio=stats.compression_ratio,
                original_bytes=stats.original_bytes,
                quantized_bytes=stats.quantized_bytes,
            )
        )

    return metrics, dense_released


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a production smoke test on a small Qwen mlx-lm model using TurboQuant cache wrappers."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Hugging Face model repo (1B params or less recommended).",
    )
    parser.add_argument("--prompt", default="Explain quantization in one short paragraph.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--key-bit-width", type=int, default=3)
    parser.add_argument("--value-bit-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-impl-rel-l2",
        type=float,
        default=1e-5,
        help="Maximum allowed per-layer relative L2 error after state serialization round-trip.",
    )
    parser.add_argument(
        "--require-exact-output",
        action="store_true",
        help="Fail if greedy-decoded token stream differs from baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load(args.model)

    add_special_tokens = tokenizer.bos_token is None or not args.prompt.startswith(tokenizer.bos_token)
    prompt_tokens = mx.array(tokenizer.encode(args.prompt, add_special_tokens=add_special_tokens))

    baseline_cache = mlx_cache.make_prompt_cache(model)
    baseline_tokens = _run_tokens(model, prompt_tokens, baseline_cache, max_tokens=args.max_tokens)

    turbo_cache = make_turbo_prompt_cache(
        model,
        key_bit_width=args.key_bit_width,
        value_bit_width=args.value_bit_width,
        seed=args.seed,
    )
    turbo_tokens = _run_tokens(model, prompt_tokens, turbo_cache, max_tokens=args.max_tokens)

    layer_metrics, dense_released = _analyze_layers(baseline_cache, turbo_cache)
    if not layer_metrics:
        raise RuntimeError("No TurboQuant-compressed cache layers were detected.")

    max_key_impl_rel = max(m.key_serialization_rel_l2 for m in layer_metrics)
    max_value_impl_rel = max(m.value_serialization_rel_l2 for m in layer_metrics)
    max_key_model_rel = max(m.key_model_rel_l2 for m in layer_metrics)
    max_value_model_rel = max(m.value_model_rel_l2 for m in layer_metrics)
    min_ratio = min(m.compression_ratio for m in layer_metrics)
    exact_output_match = baseline_tokens == turbo_tokens

    passed = (
        dense_released
        and min_ratio > 1.0
        and max_key_impl_rel <= args.max_impl_rel_l2
        and max_value_impl_rel <= args.max_impl_rel_l2
        and (exact_output_match or not args.require_exact_output)
    )

    result = {
        "model": args.model,
        "prompt_tokens": int(prompt_tokens.shape[0]),
        "generated_tokens": len(turbo_tokens),
        "dense_released": dense_released,
        "exact_output_match": exact_output_match,
        "max_key_impl_rel_l2": max_key_impl_rel,
        "max_value_impl_rel_l2": max_value_impl_rel,
        "max_key_model_rel_l2": max_key_model_rel,
        "max_value_model_rel_l2": max_value_model_rel,
        "min_compression_ratio": min_ratio,
        "layers": [asdict(m) for m in layer_metrics],
        "baseline_text": tokenizer.decode(baseline_tokens),
        "turbo_text": tokenizer.decode(turbo_tokens),
        "passed": passed,
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
