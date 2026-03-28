# turboquant-mlx

TurboQuant implementation for Apple MLX, based on the ICLR 2026 TurboQuant paper.

- `TurboQuantMSE` (Algorithm 1): low-distortion vector quantization.
- `TurboQuantProd` (Algorithm 2): inner-product-oriented quantization using residual QJL.
- `mlx_lm` integration: TurboQuant-backed prompt caches for generation/chat/evaluate flows.

## Install

```bash
pip install -e ".[dev,integration]"
```

## Quick Start

```python
import mlx.core as mx
import numpy as np

from turboquant_mlx import TurboQuantMSE, TurboQuantProd

rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((128, 64), dtype=np.float32))
q = mx.array(rng.standard_normal((8, 64), dtype=np.float32))

mse_quant = TurboQuantMSE(dimension=64, bit_width=4, seed=0)
mse_codes = mse_quant.quantize(x)
x_hat = mse_quant.dequantize(mse_codes)

prod_quant = TurboQuantProd(dimension=64, bit_width=3, seed=0)
prod_codes = prod_quant.quantize(x)
scores_hat = prod_quant.estimate_inner_products(q, prod_codes)
```

## KV Cache Helpers

```python
from turboquant_mlx import quantize_kv_cache, dequantize_kv_cache

cache = quantize_kv_cache(keys, values, key_bit_width=3, value_bit_width=4, seed=42)
keys_hat, values_hat = dequantize_kv_cache(cache)
```

## MLX-LM Integration

Use the patcher to force prompt-cache construction to TurboQuant wrappers across `mlx_lm` entry points:

```python
from turboquant_mlx import patch_mlx_lm
from mlx_lm import generate, load

patcher = patch_mlx_lm(key_bit_width=3, value_bit_width=3, seed=0)
try:
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    text = generate(model, tokenizer, prompt="Hello", max_tokens=64)
finally:
    patcher.restore()
```

Or use the context manager:

```python
from turboquant_mlx import turboquantize_mlx_lm

with turboquantize_mlx_lm(key_bit_width=3, value_bit_width=3):
    ...
```

## Cache Coverage

TurboQuant wrappers now support the major `mlx_lm` cache types used in generation pipelines:

- `KVCache`: compressed canonical state.
- `BatchKVCache`: compressed canonical state.
- `ChunkedKVCache`: compressed canonical state.
- `RotatingKVCache`: compatibility mode (dense cache kept as canonical).
- `BatchRotatingKVCache`: compatibility mode (dense cache kept as canonical).
- `ConcatenateKVCache`: compatibility mode (dense cache kept as canonical).

Unsupported cache types are passed through unchanged with a runtime warning.

## Validate With Qwen (<=1B)

Run an end-to-end smoke test against a small Qwen model from Hugging Face:

```bash
python scripts/smoke_qwen_turboquant.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --max-tokens 24
```

This checks:

- TurboQuant cache compression ratio per layer.
- Dense KV tensors are released for compressed canonical wrappers.
- Wrapper state matches direct TurboQuant round-trip (implementation correctness).
- Baseline/Turbo text match is reported as telemetry only (not a hard gate).

## Benchmarks

```bash
python scripts/benchmark_kv_cache.py --tokens 1024 --key-bit-width 3 --value-bit-width 3
```

## Production Checklist

See [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md) for release gates and operational steps.

## Tests

```bash
pytest -q
```
