# turboquant-mlx

TurboQuant implementation for Apple MLX, based on the ICLR 2026 TurboQuant paper:

- `TurboQuantMSE` (Algorithm 1): low-distortion vector quantization.
- `TurboQuantProd` (Algorithm 2): inner-product-oriented quantization using residual QJL.

## Install

```bash
pip install -e .
```

## Quick Start

```python
import mlx.core as mx
import numpy as np

from turboquant_mlx import TurboQuantMSE, TurboQuantProd

rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((128, 64), dtype=np.float32))
q = mx.array(rng.standard_normal((8, 64), dtype=np.float32))

# MSE-oriented quantization
mse_quant = TurboQuantMSE(dimension=64, bit_width=4, seed=0)
mse_codes = mse_quant.quantize(x)
x_hat = mse_quant.dequantize(mse_codes)

# Inner-product-oriented quantization
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

## Run Tests

```bash
pytest
```
