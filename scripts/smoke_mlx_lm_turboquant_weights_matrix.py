#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from mlx_lm import generate

from turboquant_mlx.mlx_lm_weights import (
    convert_turboquant_mlx_lm_model,
    load_turboquant_mlx_lm,
)


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-model smoke test for TurboQuant mlx-lm model-weight conversion/load."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more mlx-lm model ids/paths. Provide at least three families for matrix validation.",
    )
    parser.add_argument(
        "--output-root",
        default="tmp/turboquant-weight-matrix",
        help="Directory where converted artifacts are written.",
    )
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--algorithm", choices=("mse", "prod"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="Describe what model quantization is in one short paragraph.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for model_id in args.models:
        row = {
            "model": model_id,
            "status": "ok",
            "error": None,
        }
        try:
            output_dir = output_root / _slug(model_id)
            convert_summary = convert_turboquant_mlx_lm_model(
                path_or_hf_repo=model_id,
                output_path=output_dir,
                bit_width=args.bit_width,
                algorithm=args.algorithm,
                seed=args.seed,
            )

            model, tokenizer = load_turboquant_mlx_lm(str(output_dir))
            generated = generate(
                model,
                tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )

            row.update(
                {
                    "output_dir": str(output_dir),
                    "compression_ratio": convert_summary["compression_ratio"],
                    "original_bytes": convert_summary["original_bytes"],
                    "quantized_bytes": convert_summary["quantized_bytes"],
                    "artifact_size_bytes": _dir_size_bytes(output_dir),
                    "generated_chars": len(generated),
                    "generated_preview": generated[:120],
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc(limit=4)
        results.append(row)

    print(json.dumps({"results": results}, indent=2))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

