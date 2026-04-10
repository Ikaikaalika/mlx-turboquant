#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_mlx.mlx_lm_weights import convert_turboquant_mlx_lm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an mlx-lm model into TurboQuant-compressed weight artifacts."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or Hugging Face repo id loadable by mlx-lm.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for TurboQuant artifacts.",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        default=4,
        help="TurboQuant bit width.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("mse", "prod"),
        default="mse",
        help="TurboQuant algorithm variant for model weights.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-pack",
        action="store_true",
        help="Disable bit packing for debug-oriented artifacts.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--max-file-size-gb",
        type=int,
        default=5,
        help="Maximum shard size in gigabytes.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code while loading tokenizer/model assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = convert_turboquant_mlx_lm_model(
        path_or_hf_repo=args.model,
        output_path=args.output,
        bit_width=args.bit_width,
        algorithm=args.algorithm,
        seed=args.seed,
        pack=not args.no_pack,
        revision=args.revision,
        max_file_size_gb=args.max_file_size_gb,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

