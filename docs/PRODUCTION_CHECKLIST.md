# Production Checklist

Use this checklist before publishing a release of `turboquant-mlx`.

## 1. Correctness Gates

1. Run unit tests.
   ```bash
   pytest -q
   ```
2. Run KV benchmark and confirm compression ratio is stable for your target shapes.
   ```bash
   python scripts/benchmark_kv_cache.py --tokens 1024
   ```
3. Run Qwen smoke test (<=1B model) and confirm it passes.
   ```bash
   python scripts/smoke_qwen_turboquant.py \
     --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
     --max-tokens 24
   ```

## 2. Compatibility Gates

1. Confirm your target `mlx_lm` workload uses one of the supported cache wrappers:
   - `TurboQuantKVCache`
   - `TurboQuantBatchKVCache`
   - `TurboQuantChunkedKVCache`
2. If your model uses rotating/concatenate wrappers, note these currently run in compatibility mode (dense canonical state).
3. Fail deployment if runtime warnings indicate unsupported cache types in critical paths.

## 3. Packaging Gates

1. Build sdist and wheel.
   ```bash
   python -m build
   ```
2. Validate install in a clean env.
   ```bash
   pip install dist/*.whl
   ```
3. Verify import + quick API smoke.
   ```bash
   python -c "import turboquant_mlx; print('ok')"
   ```

## 4. CI Gates

1. Ensure `.github/workflows/ci.yml` is passing on `main`.
2. Trigger `.github/workflows/smoke-qwen.yml` before tagged releases.
3. Keep test runtime under CI timeout budgets.

## 5. Release Gates

1. Update `pyproject.toml` version.
2. Tag release.
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
3. Publish release notes:
   - cache coverage changes
   - compression/quality benchmark deltas
   - known limitations

## 6. Post-Release Monitoring

1. Track issues for:
   - cache serialization/deserialization failures
   - unsupported cache warnings
   - model-specific regressions
2. Add regression tests for every production incident before closing it.
