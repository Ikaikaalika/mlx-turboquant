#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_mlx import dequantize_kv_cache, quantize_kv_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark TurboQuant KV compression quality and latency."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--key-bit-width", type=int, default=3)
    parser.add_argument("--value-bit-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def _mse(a: mx.array, b: mx.array) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.mean((aa - bb) ** 2))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    key_shape = (args.batch, args.heads, args.tokens, args.key_dim)
    value_shape = (args.batch, args.heads, args.tokens, args.value_dim)
    keys = mx.array(rng.standard_normal(key_shape, dtype=np.float32))
    values = mx.array(rng.standard_normal(value_shape, dtype=np.float32))

    quant_times = []
    dequant_times = []
    key_mse = []
    value_mse = []
    key_bytes = int(np.asarray(keys).nbytes)
    value_bytes = int(np.asarray(values).nbytes)

    for _ in range(args.trials):
        t0 = time.perf_counter()
        qcache = quantize_kv_cache(
            keys=keys,
            values=values,
            key_bit_width=args.key_bit_width,
            value_bit_width=args.value_bit_width,
            seed=args.seed,
            pack=True,
        )
        quant_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        keys_hat, values_hat = dequantize_kv_cache(qcache)
        dequant_times.append(time.perf_counter() - t1)

        key_mse.append(_mse(keys, keys_hat))
        value_mse.append(_mse(values, values_hat))

    key_quant_bytes = qcache.key_codes.storage_bytes()
    value_quant_bytes = qcache.value_codes.storage_bytes()

    result: Dict[str, Any] = {
        "config": {
            "shape": {
                "keys": key_shape,
                "values": value_shape,
            },
            "key_bit_width": args.key_bit_width,
            "value_bit_width": args.value_bit_width,
            "trials": args.trials,
        },
        "quality": {
            "key_mse_mean": float(np.mean(key_mse)),
            "value_mse_mean": float(np.mean(value_mse)),
            "key_mse_std": float(np.std(key_mse)),
            "value_mse_std": float(np.std(value_mse)),
        },
        "compression": {
            "key_original_bytes": key_bytes,
            "value_original_bytes": value_bytes,
            "key_quantized_bytes": key_quant_bytes,
            "value_quantized_bytes": value_quant_bytes,
            "key_ratio": float(key_bytes / max(1, key_quant_bytes)),
            "value_ratio": float(value_bytes / max(1, value_quant_bytes)),
            "overall_ratio": float((key_bytes + value_bytes) / max(1, key_quant_bytes + value_quant_bytes)),
        },
        "latency_seconds": {
            "quantize_mean": float(np.mean(quant_times)),
            "dequantize_mean": float(np.mean(dequant_times)),
            "quantize_std": float(np.std(quant_times)),
            "dequantize_std": float(np.std(dequant_times)),
        },
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
