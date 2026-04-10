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

## Model Weight Helpers

TurboQuant can also quantize model parameter trees (not only KV cache tensors):

```python
from turboquant_mlx import (
    dequantize_model_weights,
    quantize_model_weights,
    turboquantize_model_weights,
)

# Quantize a parameter pytree and reconstruct dequantized weights later.
packed = quantize_model_weights(model.parameters(), bit_width=4, algorithm="mse", seed=0)
restored_params = dequantize_model_weights(packed)
model.update(restored_params)

# Convenience in-place helper (quantize + write back dequantized weights).
packed_inplace = turboquantize_model_weights(model, bit_width=4, algorithm="mse", seed=0)
print(packed_inplace.stats.compression_ratio)
```

Notes:
- Floating-point tensor leaves are quantized.
- Non-floating/scalar leaves are passed through unchanged.
- `algorithm="mse"` is the default; `algorithm="prod"` is also supported.

### Save/Load TurboQuant Artifacts

```python
from turboquant_mlx import load_turboquant_weights, save_turboquant_weights

# Saves:
# - config.json with a `turboquant` block
# - turboquant-weights*.safetensors + index
# - turboquant-passthrough*.safetensors + index
# - turboquant-metadata.json
save_turboquant_weights(model.parameters(), "artifacts/my-model-tq", bit_width=4, seed=0)

restored = load_turboquant_weights("artifacts/my-model-tq")
model.update(restored)
```

## MLX-LM Model Weight Integration

You can convert and reload any `mlx_lm` model using TurboQuant-compressed on-disk
weights (runtime remains load-dequantized MLX tensors for broad compatibility).

```python
from turboquant_mlx import (
    convert_turboquant_mlx_lm_model,
    load_turboquant_mlx_lm,
    patch_mlx_lm_weights,
)

summary = convert_turboquant_mlx_lm_model(
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "artifacts/qwen-tq",
    bit_width=4,
    algorithm="mse",
    seed=0,
)

model, tokenizer = load_turboquant_mlx_lm("artifacts/qwen-tq")

# Optional transparent load() patching:
weight_patcher = patch_mlx_lm_weights()
try:
    from mlx_lm import load
    model2, tokenizer2 = load("artifacts/qwen-tq")
finally:
    weight_patcher.restore()
```

CLI conversion:

```bash
python scripts/convert_mlx_lm_turboquant_weights.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --output artifacts/qwen-tq \
  --bit-width 4
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

## Matrix Smoke (Manual)

Run model-weight conversion/load checks across multiple model families:

```bash
python scripts/smoke_mlx_lm_turboquant_weights_matrix.py \
  --models \
    mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    mlx-community/Llama-3.2-1B-Instruct-4bit \
    mlx-community/SmolLM2-360M-Instruct-4bit
```

The script reports per-model conversion success, load success, sample generation,
and storage reduction metrics.

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
