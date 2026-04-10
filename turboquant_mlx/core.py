from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

TURBOQUANT_CONFIG_KEY = "turboquant"
TURBOQUANT_FORMAT_VERSION = 1
TURBOQUANT_METADATA_FILENAME = "turboquant-metadata.json"
TURBOQUANT_WEIGHT_INDEX_FILENAME = "turboquant-weights.index.json"
TURBOQUANT_PASSTHROUGH_INDEX_FILENAME = "turboquant-passthrough.index.json"


def _num_vectors(leading_shape: Tuple[int, ...]) -> int:
    return int(np.prod(leading_shape)) if leading_shape else 1


def _flatten_rows(x: mx.array, dimension: int) -> tuple[mx.array, tuple[int, ...]]:
    if x.shape[-1] != dimension:
        raise ValueError(f"Expected trailing dimension {dimension}, got {x.shape[-1]}")
    leading_shape = tuple(int(v) for v in x.shape[:-1])
    return x.reshape((-1, dimension)).astype(mx.float32), leading_shape


def _pack_bits(indices: mx.array, bits: int) -> mx.array:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    flat = np.asarray(indices, dtype=np.uint16).reshape(-1)
    count = int(flat.size)
    if count == 0:
        return mx.array(np.zeros((0,), dtype=np.uint8))

    if bits == 8:
        return mx.array(flat.astype(np.uint8))

    total_bits = count * bits
    out = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    mask = (1 << bits) - 1
    offset = 0

    for value in flat:
        v = int(value) & mask
        byte_idx = offset >> 3
        bit_idx = offset & 7

        out[byte_idx] |= (v << bit_idx) & 0xFF
        spill = bit_idx + bits - 8
        if spill > 0:
            out[byte_idx + 1] |= (v >> (bits - spill)) & 0xFF

        offset += bits

    return mx.array(out)


def _unpack_bits(packed: mx.array, bits: int, count: int) -> mx.array:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    if count == 0:
        return mx.array(np.zeros((0,), dtype=np.int32))

    raw = np.asarray(packed, dtype=np.uint8).reshape(-1)

    if bits == 8:
        if raw.size < count:
            raise ValueError("Packed array is too short for requested count")
        return mx.array(raw[:count].astype(np.int32))

    out = np.zeros((count,), dtype=np.uint16)
    mask = (1 << bits) - 1
    offset = 0

    for i in range(count):
        byte_idx = offset >> 3
        bit_idx = offset & 7

        if byte_idx >= raw.size:
            raise ValueError("Packed array is too short for requested count")

        value = (int(raw[byte_idx]) >> bit_idx) & mask
        if bit_idx + bits > 8:
            if byte_idx + 1 >= raw.size:
                raise ValueError("Packed array is too short for requested count")
            value |= (int(raw[byte_idx + 1]) << (8 - bit_idx)) & mask

        out[i] = value
        offset += bits

    return mx.array(out.astype(np.int32))


def _pack_signs(signs: mx.array) -> mx.array:
    bits = mx.where(signs > 0, 1, 0).astype(mx.int32)
    return _pack_bits(bits, bits=1)


def _unpack_signs(packed: mx.array, count: int) -> mx.array:
    bits = _unpack_bits(packed, bits=1, count=count)
    return mx.where(bits > 0, 1, -1).astype(mx.int8)


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _truncated_normal_mean(a: float, b: float, sigma: float) -> float:
    za = a / sigma
    zb = b / sigma

    phi_a = _normal_pdf(za) if math.isfinite(za) else 0.0
    phi_b = _normal_pdf(zb) if math.isfinite(zb) else 0.0

    cdf_a = _normal_cdf(za) if math.isfinite(za) else (0.0 if za < 0 else 1.0)
    cdf_b = _normal_cdf(zb) if math.isfinite(zb) else (0.0 if zb < 0 else 1.0)

    denom = cdf_b - cdf_a
    if denom <= 1e-12:
        if math.isfinite(a) and math.isfinite(b):
            return 0.5 * (a + b)
        return 0.0

    return sigma * (phi_a - phi_b) / denom


def _lloyd_max_normal_codebook(
    dimension: int,
    bit_width: int,
    max_iters: int = 128,
    tol: float = 1e-7,
) -> np.ndarray:
    if bit_width < 1:
        raise ValueError(f"bit_width must be >= 1, got {bit_width}")

    k = 1 << bit_width
    sigma = 1.0 / math.sqrt(float(dimension))

    span = 3.5 * sigma
    centroids = np.linspace(-span, span, k, dtype=np.float64)

    for _ in range(max_iters):
        boundaries = np.empty((k + 1,), dtype=np.float64)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        updated = np.empty_like(centroids)
        for i in range(k):
            updated[i] = _truncated_normal_mean(boundaries[i], boundaries[i + 1], sigma)

        if np.max(np.abs(updated - centroids)) < tol:
            centroids = updated
            break

        centroids = updated

    centroids.sort()
    return centroids.astype(np.float32)


def _random_orthogonal_matrix(dimension: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension), dtype=np.float64)
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    return mx.array(q.astype(np.float32))


def _random_gaussian_projection(dimension: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension), dtype=np.float32)
    return mx.array(gaussian)


@dataclass(frozen=True)
class PackedMSECodes:
    packed_indices: mx.array
    norms: mx.array
    leading_shape: tuple[int, ...]
    dimension: int
    bit_width: int
    packed: bool = True

    @property
    def num_vectors(self) -> int:
        return _num_vectors(self.leading_shape)

    def storage_bytes(self) -> int:
        return int(np.asarray(self.packed_indices).nbytes + np.asarray(self.norms).nbytes)


@dataclass(frozen=True)
class PackedProdCodes:
    packed_indices: Optional[mx.array]
    packed_qjl: mx.array
    residual_norms: mx.array
    norms: mx.array
    leading_shape: tuple[int, ...]
    dimension: int
    bit_width: int
    packed: bool = True

    @property
    def num_vectors(self) -> int:
        return _num_vectors(self.leading_shape)

    def storage_bytes(self) -> int:
        indices_bytes = 0 if self.packed_indices is None else int(np.asarray(self.packed_indices).nbytes)
        return indices_bytes + int(np.asarray(self.packed_qjl).nbytes) + int(np.asarray(self.residual_norms).nbytes) + int(np.asarray(self.norms).nbytes)


class TurboQuantMSE:
    """TurboQuant MSE-oriented quantizer (Algorithm 1 in the paper)."""

    def __init__(self, dimension: int, bit_width: int, seed: int = 0, epsilon: float = 1e-8):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if bit_width < 1 or bit_width > 8:
            raise ValueError(f"bit_width must be in [1, 8], got {bit_width}")

        self.dimension = int(dimension)
        self.bit_width = int(bit_width)
        self.seed = int(seed)
        self.epsilon = float(epsilon)

        self.rotation = _random_orthogonal_matrix(self.dimension, self.seed)
        self.codebook = mx.array(_lloyd_max_normal_codebook(self.dimension, self.bit_width))

    def _quantize_unit_rows(self, unit_rows: mx.array) -> mx.array:
        rotated = mx.matmul(unit_rows, self.rotation)
        distances = mx.abs(rotated[..., None] - self.codebook)
        return mx.argmin(distances, axis=-1).astype(mx.int32)

    def _dequantize_unit_rows(self, indices: mx.array) -> mx.array:
        rotated_hat = self.codebook[indices]
        return mx.matmul(rotated_hat, self.rotation.T)

    def _decode_indices(self, codes: PackedMSECodes) -> mx.array:
        count = codes.num_vectors * self.dimension
        if codes.packed:
            idx = _unpack_bits(codes.packed_indices, bits=codes.bit_width, count=count)
        else:
            idx = codes.packed_indices.astype(mx.int32).reshape((-1,))
        return idx.reshape((codes.num_vectors, self.dimension))

    def quantize(self, x: mx.array, pack: bool = True) -> PackedMSECodes:
        rows, leading_shape = _flatten_rows(x, self.dimension)

        norms = mx.linalg.norm(rows, axis=-1)
        safe_norms = mx.maximum(norms, self.epsilon)
        unit_rows = rows / safe_norms[:, None]

        indices = self._quantize_unit_rows(unit_rows)
        packed_indices = _pack_bits(indices, self.bit_width) if pack else indices

        return PackedMSECodes(
            packed_indices=packed_indices,
            norms=norms.astype(mx.float32),
            leading_shape=leading_shape,
            dimension=self.dimension,
            bit_width=self.bit_width,
            packed=pack,
        )

    def dequantize(self, codes: PackedMSECodes) -> mx.array:
        if codes.dimension != self.dimension:
            raise ValueError(f"Code dimension {codes.dimension} does not match quantizer dimension {self.dimension}")
        if codes.bit_width != self.bit_width:
            raise ValueError(f"Code bit_width {codes.bit_width} does not match quantizer bit_width {self.bit_width}")

        indices = self._decode_indices(codes)
        unit_hat = self._dequantize_unit_rows(indices)
        rows = unit_hat * codes.norms[:, None]

        return rows.reshape(codes.leading_shape + (self.dimension,))


class TurboQuantProd:
    """TurboQuant inner-product quantizer (Algorithm 2 in the paper)."""

    def __init__(self, dimension: int, bit_width: int, seed: int = 0, epsilon: float = 1e-8):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if bit_width < 1 or bit_width > 8:
            raise ValueError(f"bit_width must be in [1, 8], got {bit_width}")

        self.dimension = int(dimension)
        self.bit_width = int(bit_width)
        self.seed = int(seed)
        self.epsilon = float(epsilon)

        self.mse_quantizer: Optional[TurboQuantMSE]
        if self.bit_width > 1:
            self.mse_quantizer = TurboQuantMSE(
                dimension=self.dimension,
                bit_width=self.bit_width - 1,
                seed=self.seed,
                epsilon=self.epsilon,
            )
        else:
            self.mse_quantizer = None

        self.qjl_projection = _random_gaussian_projection(self.dimension, self.seed + 1)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / float(self.dimension)

    def _decode_indices(self, codes: PackedProdCodes) -> Optional[mx.array]:
        if self.mse_quantizer is None:
            return None
        if codes.packed_indices is None:
            raise ValueError("Expected packed_indices for bit_width > 1")

        count = codes.num_vectors * self.dimension
        mse_bits = self.bit_width - 1
        if codes.packed:
            idx = _unpack_bits(codes.packed_indices, bits=mse_bits, count=count)
        else:
            idx = codes.packed_indices.astype(mx.int32).reshape((-1,))
        return idx.reshape((codes.num_vectors, self.dimension))

    def _decode_qjl(self, codes: PackedProdCodes) -> mx.array:
        count = codes.num_vectors * self.dimension
        if codes.packed:
            signs = _unpack_signs(codes.packed_qjl, count=count)
        else:
            signs = codes.packed_qjl.astype(mx.int8).reshape((-1,))
        return signs.reshape((codes.num_vectors, self.dimension)).astype(mx.float32)

    def quantize(self, x: mx.array, pack: bool = True) -> PackedProdCodes:
        rows, leading_shape = _flatten_rows(x, self.dimension)

        norms = mx.linalg.norm(rows, axis=-1)
        safe_norms = mx.maximum(norms, self.epsilon)
        unit_rows = rows / safe_norms[:, None]

        if self.mse_quantizer is None:
            indices = None
            mse_hat = mx.zeros_like(unit_rows)
        else:
            indices = self.mse_quantizer._quantize_unit_rows(unit_rows)
            mse_hat = self.mse_quantizer._dequantize_unit_rows(indices)

        residual = unit_rows - mse_hat
        residual_norms = mx.linalg.norm(residual, axis=-1)

        projected = mx.matmul(residual, self.qjl_projection.T)
        qjl_signs = mx.where(projected >= 0.0, 1, -1).astype(mx.int8)

        packed_indices = None
        if indices is not None:
            packed_indices = _pack_bits(indices, self.bit_width - 1) if pack else indices

        packed_qjl = _pack_signs(qjl_signs) if pack else qjl_signs

        return PackedProdCodes(
            packed_indices=packed_indices,
            packed_qjl=packed_qjl,
            residual_norms=residual_norms.astype(mx.float32),
            norms=norms.astype(mx.float32),
            leading_shape=leading_shape,
            dimension=self.dimension,
            bit_width=self.bit_width,
            packed=pack,
        )

    def dequantize(self, codes: PackedProdCodes) -> mx.array:
        if codes.dimension != self.dimension:
            raise ValueError(f"Code dimension {codes.dimension} does not match quantizer dimension {self.dimension}")
        if codes.bit_width != self.bit_width:
            raise ValueError(f"Code bit_width {codes.bit_width} does not match quantizer bit_width {self.bit_width}")

        if self.mse_quantizer is None:
            mse_hat = mx.zeros((codes.num_vectors, self.dimension), dtype=mx.float32)
        else:
            indices = self._decode_indices(codes)
            assert indices is not None
            mse_hat = self.mse_quantizer._dequantize_unit_rows(indices)

        qjl_signs = self._decode_qjl(codes)
        qjl_hat = self.qjl_scale * codes.residual_norms[:, None] * mx.matmul(qjl_signs, self.qjl_projection)

        unit_hat = mse_hat + qjl_hat
        rows = unit_hat * codes.norms[:, None]

        return rows.reshape(codes.leading_shape + (self.dimension,))

    def estimate_inner_products(self, query: mx.array, codes: PackedProdCodes) -> mx.array:
        query_rows, query_leading_shape = _flatten_rows(query, self.dimension)
        dequantized = self.dequantize(codes).reshape((codes.num_vectors, self.dimension))
        scores = mx.matmul(query_rows, dequantized.T)
        return scores.reshape(query_leading_shape + codes.leading_shape)


@dataclass(frozen=True)
class QuantizedKVCache:
    key_quantizer: TurboQuantProd
    value_quantizer: TurboQuantMSE
    key_codes: PackedProdCodes
    value_codes: PackedMSECodes


@dataclass(frozen=True)
class QuantizedWeightsStats:
    total_tensors: int
    quantized_tensors: int
    skipped_tensors: int
    original_bytes: int
    quantized_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes <= 0:
            return float("inf")
        return float(self.original_bytes / self.quantized_bytes)


@dataclass(frozen=True)
class QuantizedWeightTensor:
    path: str
    quantizer_kind: str
    bit_width: int
    seed: int
    dtype: Any
    original_bytes: int
    codes: PackedMSECodes | PackedProdCodes

    def storage_bytes(self) -> int:
        return int(self.codes.storage_bytes())


@dataclass(frozen=True)
class QuantizedModelWeights:
    quantized_tensors: tuple[QuantizedWeightTensor, ...]
    passthrough_leaves: tuple[tuple[str, Any], ...]
    stats: QuantizedWeightsStats


def _is_floating_tensor(x: mx.array) -> bool:
    return bool(mx.issubdtype(x.dtype, mx.floating))


def _array_nbytes(x: mx.array) -> int:
    return int(x.nbytes)


def _dtype_to_name(dtype: Any) -> str:
    raw = str(dtype)
    if raw.startswith("mlx.core."):
        raw = raw[len("mlx.core.") :]
    if raw == "bool":
        return "bool_"
    return raw


def _dtype_from_name(name: str) -> Any:
    if hasattr(mx, name):
        return getattr(mx, name)
    if name == "bool":
        return mx.bool_
    raise ValueError(f"Unsupported dtype name {name!r}")


def _make_shards(tensors: dict[str, mx.array], max_file_size_bytes: int) -> list[dict[str, mx.array]]:
    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive")

    shards: list[dict[str, mx.array]] = []
    shard: dict[str, mx.array] = {}
    shard_size = 0

    for key, value in tensors.items():
        value_nbytes = _array_nbytes(value)
        if shard and shard_size + value_nbytes > max_file_size_bytes:
            shards.append(shard)
            shard = {}
            shard_size = 0

        shard[key] = value
        shard_size += value_nbytes

    if shard:
        shards.append(shard)
    return shards


def _save_sharded_tensors(
    tensors: dict[str, mx.array],
    output_path: Path,
    *,
    prefix: str,
    index_filename: str,
    max_file_size_bytes: int,
) -> None:
    shards = _make_shards(tensors, max_file_size_bytes) if tensors else []
    shard_count = len(shards)

    weight_map: dict[str, str] = {}
    total_size = 0

    for i, shard in enumerate(shards):
        if shard_count == 1:
            shard_name = f"{prefix}.safetensors"
        else:
            shard_name = f"{prefix}-{i + 1:05d}-of-{shard_count:05d}.safetensors"
        shard_path = output_path / shard_name
        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})
        for key, value in shard.items():
            weight_map[key] = shard_name
            total_size += _array_nbytes(value)

    with open(output_path / index_filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "format": "mlx",
                    "total_size": int(total_size),
                    "tensor_count": len(weight_map),
                },
                "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
            },
            f,
            indent=2,
        )


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing index file: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid index format at {path}")
    if "weight_map" not in payload or not isinstance(payload["weight_map"], dict):
        raise ValueError(f"Invalid index file (weight_map missing): {path}")
    return payload


def _load_index_tensors(index_path: Path) -> dict[str, mx.array]:
    index_payload = _load_index(index_path)
    weight_map: dict[str, str] = index_payload["weight_map"]
    if not weight_map:
        return {}

    tensors: dict[str, mx.array] = {}
    loaded_files: dict[str, dict[str, mx.array]] = {}
    base_dir = index_path.parent

    for key, filename in weight_map.items():
        if filename not in loaded_files:
            shard_path = base_dir / filename
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Missing shard file {shard_path} referenced by {index_path.name}"
                )
            loaded_files[filename] = mx.load(str(shard_path))
        shard = loaded_files[filename]
        if key not in shard:
            raise ValueError(
                f"Tensor key {key!r} missing in shard {filename} referenced by {index_path.name}"
            )
        tensors[key] = shard[key]

    return tensors


def _require_tensor(store: dict[str, mx.array], key: str, *, source: str) -> mx.array:
    if key not in store:
        raise ValueError(f"Missing tensor key {key!r} in {source}")
    return store[key]


def quantize_model_weights(
    weights: Any,
    *,
    bit_width: int = 4,
    seed: int = 0,
    pack: bool = True,
    algorithm: str = "mse",
) -> QuantizedModelWeights:
    """
    Quantize MLX model weights (or any MLX pytree) with TurboQuant.

    ``weights`` may be a pytree of tensors (for example ``model.parameters()``)
    or an object exposing ``parameters()``. Floating-point tensor leaves are
    quantized, while non-floating/scalar leaves are passed through unchanged.
    """
    if hasattr(weights, "parameters") and callable(weights.parameters):
        weights = weights.parameters()

    if bit_width < 1 or bit_width > 8:
        raise ValueError(f"bit_width must be in [1, 8], got {bit_width}")

    algo = str(algorithm).strip().lower()
    if algo not in {"mse", "prod"}:
        raise ValueError(f"algorithm must be 'mse' or 'prod', got {algorithm!r}")

    leaves = tree_flatten(weights)
    quantized_tensors: list[QuantizedWeightTensor] = []
    passthrough_leaves: list[tuple[str, Any]] = []

    total_tensors = 0
    quantized_count = 0
    skipped_count = 0
    original_bytes = 0
    quantized_bytes = 0

    for i, (path, leaf) in enumerate(leaves):
        if isinstance(leaf, mx.array):
            total_tensors += 1
            leaf_nbytes = _array_nbytes(leaf)
            original_bytes += leaf_nbytes

            if leaf.ndim > 0 and int(leaf.shape[-1]) > 0 and _is_floating_tensor(leaf):
                tensor_seed = int(seed) + i
                dimension = int(leaf.shape[-1])
                if algo == "mse":
                    quantizer = TurboQuantMSE(
                        dimension=dimension,
                        bit_width=bit_width,
                        seed=tensor_seed,
                    )
                else:
                    quantizer = TurboQuantProd(
                        dimension=dimension,
                        bit_width=bit_width,
                        seed=tensor_seed,
                    )
                codes = quantizer.quantize(leaf, pack=pack)
                quantized_tensors.append(
                    QuantizedWeightTensor(
                        path=str(path),
                        quantizer_kind=algo,
                        bit_width=int(bit_width),
                        seed=tensor_seed,
                        dtype=leaf.dtype,
                        original_bytes=leaf_nbytes,
                        codes=codes,
                    )
                )
                quantized_count += 1
                quantized_bytes += int(codes.storage_bytes())
                continue

            skipped_count += 1
            quantized_bytes += leaf_nbytes
            passthrough_leaves.append((str(path), leaf))
            continue

        passthrough_leaves.append((str(path), leaf))

    stats = QuantizedWeightsStats(
        total_tensors=total_tensors,
        quantized_tensors=quantized_count,
        skipped_tensors=skipped_count,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
    )
    return QuantizedModelWeights(
        quantized_tensors=tuple(quantized_tensors),
        passthrough_leaves=tuple(passthrough_leaves),
        stats=stats,
    )


def dequantize_model_weights(quantized: QuantizedModelWeights) -> Any:
    """Reconstruct a weight pytree from ``QuantizedModelWeights``."""
    leaves: list[tuple[str, Any]] = list(quantized.passthrough_leaves)

    for leaf in quantized.quantized_tensors:
        if leaf.quantizer_kind == "mse":
            if not isinstance(leaf.codes, PackedMSECodes):
                raise TypeError(f"Expected PackedMSECodes for path {leaf.path!r}")
            quantizer = TurboQuantMSE(
                dimension=leaf.codes.dimension,
                bit_width=leaf.bit_width,
                seed=leaf.seed,
            )
            value = quantizer.dequantize(leaf.codes)
        elif leaf.quantizer_kind == "prod":
            if not isinstance(leaf.codes, PackedProdCodes):
                raise TypeError(f"Expected PackedProdCodes for path {leaf.path!r}")
            quantizer = TurboQuantProd(
                dimension=leaf.codes.dimension,
                bit_width=leaf.bit_width,
                seed=leaf.seed,
            )
            value = quantizer.dequantize(leaf.codes)
        else:
            raise ValueError(f"Unknown quantizer kind {leaf.quantizer_kind!r}")

        leaves.append((leaf.path, value.astype(leaf.dtype)))

    return tree_unflatten(leaves)


def turboquantize_model_weights(
    model: Any,
    *,
    bit_width: int = 4,
    seed: int = 0,
    pack: bool = True,
    algorithm: str = "mse",
) -> QuantizedModelWeights:
    """
    Quantize model parameters with TurboQuant and write dequantized weights back.

    The returned object keeps the compressed representation and aggregate stats.
    """
    if not hasattr(model, "parameters") or not callable(model.parameters):
        raise TypeError("model must expose a callable parameters() method")
    if not hasattr(model, "update") or not callable(model.update):
        raise TypeError("model must expose a callable update(...) method")

    quantized = quantize_model_weights(
        model.parameters(),
        bit_width=bit_width,
        seed=seed,
        pack=pack,
        algorithm=algorithm,
    )
    model.update(dequantize_model_weights(quantized))
    return quantized


def save_turboquant_weights(
    weights: Any,
    output_path: str | Path,
    *,
    bit_width: int = 4,
    seed: int = 0,
    pack: bool = True,
    algorithm: str = "mse",
    max_file_size_gb: int = 5,
    config: Optional[dict[str, Any]] = None,
) -> QuantizedModelWeights:
    """
    Quantize and persist model weights as TurboQuant sharded artifacts.

    The destination directory will contain:
    - ``config.json`` with a ``turboquant`` metadata block.
    - Quantized shard index: ``turboquant-weights.index.json``.
    - Passthrough shard index: ``turboquant-passthrough.index.json``.
    - Metadata file: ``turboquant-metadata.json``.
    """
    if max_file_size_gb <= 0:
        raise ValueError(f"max_file_size_gb must be positive, got {max_file_size_gb}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantized = quantize_model_weights(
        weights,
        bit_width=bit_width,
        seed=seed,
        pack=pack,
        algorithm=algorithm,
    )
    max_file_size_bytes = int(max_file_size_gb * (1 << 30))

    quantized_tensors: dict[str, mx.array] = {}
    quantized_meta: list[dict[str, Any]] = []
    for i, leaf in enumerate(quantized.quantized_tensors):
        entry: dict[str, Any] = {
            "path": leaf.path,
            "quantizer_kind": leaf.quantizer_kind,
            "bit_width": int(leaf.bit_width),
            "seed": int(leaf.seed),
            "dtype": _dtype_to_name(leaf.dtype),
            "original_bytes": int(leaf.original_bytes),
        }
        base_key = f"q.{i}"

        if isinstance(leaf.codes, PackedMSECodes):
            indices_key = f"{base_key}.indices"
            norms_key = f"{base_key}.norms"
            quantized_tensors[indices_key] = leaf.codes.packed_indices
            quantized_tensors[norms_key] = leaf.codes.norms
            entry["codes"] = {
                "kind": "mse",
                "packed": bool(leaf.codes.packed),
                "dimension": int(leaf.codes.dimension),
                "leading_shape": [int(v) for v in leaf.codes.leading_shape],
                "packed_indices_key": indices_key,
                "norms_key": norms_key,
            }
        elif isinstance(leaf.codes, PackedProdCodes):
            qjl_key = f"{base_key}.qjl"
            residual_norms_key = f"{base_key}.residual_norms"
            norms_key = f"{base_key}.norms"

            packed_indices_key: Optional[str] = None
            if leaf.codes.packed_indices is not None:
                packed_indices_key = f"{base_key}.indices"
                quantized_tensors[packed_indices_key] = leaf.codes.packed_indices

            quantized_tensors[qjl_key] = leaf.codes.packed_qjl
            quantized_tensors[residual_norms_key] = leaf.codes.residual_norms
            quantized_tensors[norms_key] = leaf.codes.norms

            entry["codes"] = {
                "kind": "prod",
                "packed": bool(leaf.codes.packed),
                "dimension": int(leaf.codes.dimension),
                "leading_shape": [int(v) for v in leaf.codes.leading_shape],
                "packed_indices_key": packed_indices_key,
                "packed_qjl_key": qjl_key,
                "residual_norms_key": residual_norms_key,
                "norms_key": norms_key,
            }
        else:
            raise TypeError(f"Unsupported quantized leaf codes for path {leaf.path!r}")

        quantized_meta.append(entry)

    passthrough_tensors: dict[str, mx.array] = {}
    passthrough_meta: list[dict[str, Any]] = []
    for i, (path, leaf) in enumerate(quantized.passthrough_leaves):
        if not isinstance(leaf, mx.array):
            raise TypeError(
                "save_turboquant_weights only supports passthrough leaves that are mx.array"
            )
        storage_key = f"p.{i}"
        passthrough_tensors[storage_key] = leaf
        passthrough_meta.append(
            {
                "path": str(path),
                "dtype": _dtype_to_name(leaf.dtype),
                "shape": [int(v) for v in leaf.shape],
                "storage_key": storage_key,
            }
        )

    _save_sharded_tensors(
        quantized_tensors,
        output_dir,
        prefix="turboquant-weights",
        index_filename=TURBOQUANT_WEIGHT_INDEX_FILENAME,
        max_file_size_bytes=max_file_size_bytes,
    )
    _save_sharded_tensors(
        passthrough_tensors,
        output_dir,
        prefix="turboquant-passthrough",
        index_filename=TURBOQUANT_PASSTHROUGH_INDEX_FILENAME,
        max_file_size_bytes=max_file_size_bytes,
    )

    metadata = {
        "format_version": TURBOQUANT_FORMAT_VERSION,
        "algorithm": str(algorithm),
        "bit_width": int(bit_width),
        "pack": bool(pack),
        "seed": int(seed),
        "stats": {
            "total_tensors": int(quantized.stats.total_tensors),
            "quantized_tensors": int(quantized.stats.quantized_tensors),
            "skipped_tensors": int(quantized.stats.skipped_tensors),
            "original_bytes": int(quantized.stats.original_bytes),
            "quantized_bytes": int(quantized.stats.quantized_bytes),
        },
        "quantized_tensors": quantized_meta,
        "passthrough_tensors": passthrough_meta,
    }
    with open(output_dir / TURBOQUANT_METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)

    if config is None:
        config_path = output_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                output_config = json.load(f)
        else:
            output_config = {}
    else:
        output_config = copy.deepcopy(config)

    output_config[TURBOQUANT_CONFIG_KEY] = {
        "format_version": TURBOQUANT_FORMAT_VERSION,
        "algorithm": str(algorithm),
        "bit_width": int(bit_width),
        "pack": bool(pack),
        "seed": int(seed),
        "weight_index_filename": TURBOQUANT_WEIGHT_INDEX_FILENAME,
        "passthrough_index_filename": TURBOQUANT_PASSTHROUGH_INDEX_FILENAME,
        "metadata_filename": TURBOQUANT_METADATA_FILENAME,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(output_config, f, indent=2)

    return quantized


def load_turboquant_weights(
    input_path: str | Path,
    *,
    return_quantized: bool = False,
) -> Any | tuple[Any, QuantizedModelWeights]:
    """
    Load TurboQuant artifacts from disk and return dequantized weight pytree.
    """
    input_dir = Path(input_path)
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {input_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    tq_cfg = config.get(TURBOQUANT_CONFIG_KEY)
    if not isinstance(tq_cfg, dict):
        raise ValueError(
            f"config.json at {input_dir} does not contain a {TURBOQUANT_CONFIG_KEY!r} block"
        )

    for required_key in (
        "format_version",
        "algorithm",
        "bit_width",
        "pack",
        "seed",
        "weight_index_filename",
        "passthrough_index_filename",
        "metadata_filename",
    ):
        if required_key not in tq_cfg:
            raise ValueError(
                f"TurboQuant config in {config_path} is missing required key {required_key!r}"
            )

    format_version = int(tq_cfg["format_version"])
    if format_version != TURBOQUANT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported TurboQuant format_version={format_version}. "
            f"Expected {TURBOQUANT_FORMAT_VERSION}."
        )

    metadata_path = input_dir / str(tq_cfg["metadata_filename"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing TurboQuant metadata file: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if int(metadata.get("format_version", -1)) != TURBOQUANT_FORMAT_VERSION:
        raise ValueError(
            f"Invalid TurboQuant metadata format_version in {metadata_path.name}"
        )

    quantized_store = _load_index_tensors(input_dir / str(tq_cfg["weight_index_filename"]))
    passthrough_store = _load_index_tensors(
        input_dir / str(tq_cfg["passthrough_index_filename"])
    )

    quantized_leaves: list[QuantizedWeightTensor] = []
    for entry in metadata.get("quantized_tensors", []):
        if not isinstance(entry, dict):
            raise ValueError("Invalid quantized_tensors metadata entry")
        for required_key in ("path", "quantizer_kind", "bit_width", "seed", "dtype", "codes"):
            if required_key not in entry:
                raise ValueError(
                    f"Quantized tensor metadata missing required key {required_key!r}"
                )

        codes_meta = entry["codes"]
        if not isinstance(codes_meta, dict):
            raise ValueError("Invalid codes metadata entry")
        for required_key in ("kind", "packed", "dimension", "leading_shape"):
            if required_key not in codes_meta:
                raise ValueError(
                    f"Codes metadata missing required key {required_key!r}"
                )

        kind = str(codes_meta["kind"])
        leading_shape = tuple(int(v) for v in codes_meta["leading_shape"])
        bit_width = int(entry["bit_width"])
        seed = int(entry["seed"])
        dtype = _dtype_from_name(str(entry["dtype"]))

        if kind == "mse":
            indices_key = str(codes_meta["packed_indices_key"])
            norms_key = str(codes_meta["norms_key"])
            codes = PackedMSECodes(
                packed_indices=_require_tensor(
                    quantized_store, indices_key, source="TurboQuant quantized index"
                ),
                norms=_require_tensor(
                    quantized_store, norms_key, source="TurboQuant quantized index"
                ),
                leading_shape=leading_shape,
                dimension=int(codes_meta["dimension"]),
                bit_width=bit_width,
                packed=bool(codes_meta["packed"]),
            )
        elif kind == "prod":
            indices_key = codes_meta.get("packed_indices_key")
            packed_indices = (
                None
                if indices_key is None
                else _require_tensor(
                    quantized_store,
                    str(indices_key),
                    source="TurboQuant quantized index",
                )
            )
            qjl_key = str(codes_meta["packed_qjl_key"])
            residual_norms_key = str(codes_meta["residual_norms_key"])
            norms_key = str(codes_meta["norms_key"])

            codes = PackedProdCodes(
                packed_indices=packed_indices,
                packed_qjl=_require_tensor(
                    quantized_store, qjl_key, source="TurboQuant quantized index"
                ),
                residual_norms=_require_tensor(
                    quantized_store,
                    residual_norms_key,
                    source="TurboQuant quantized index",
                ),
                norms=_require_tensor(
                    quantized_store, norms_key, source="TurboQuant quantized index"
                ),
                leading_shape=leading_shape,
                dimension=int(codes_meta["dimension"]),
                bit_width=bit_width,
                packed=bool(codes_meta["packed"]),
            )
        else:
            raise ValueError(f"Unknown code kind {kind!r}")

        quantized_leaves.append(
            QuantizedWeightTensor(
                path=str(entry["path"]),
                quantizer_kind=str(entry["quantizer_kind"]),
                bit_width=bit_width,
                seed=seed,
                dtype=dtype,
                original_bytes=int(entry.get("original_bytes", 0)),
                codes=codes,
            )
        )

    passthrough_leaves: list[tuple[str, Any]] = []
    for entry in metadata.get("passthrough_tensors", []):
        if not isinstance(entry, dict):
            raise ValueError("Invalid passthrough_tensors metadata entry")
        for required_key in ("path", "dtype", "shape", "storage_key"):
            if required_key not in entry:
                raise ValueError(
                    f"Passthrough tensor metadata missing required key {required_key!r}"
                )

        storage_key = str(entry["storage_key"])
        tensor = _require_tensor(
            passthrough_store,
            storage_key,
            source="TurboQuant passthrough index",
        )
        dtype = _dtype_from_name(str(entry["dtype"]))
        expected_shape = tuple(int(v) for v in entry["shape"])
        casted = tensor.astype(dtype)
        if tuple(int(v) for v in casted.shape) != expected_shape:
            raise ValueError(
                f"Passthrough tensor {entry['path']!r} shape mismatch. "
                f"Expected {expected_shape}, got {tuple(int(v) for v in casted.shape)}."
            )
        passthrough_leaves.append((str(entry["path"]), casted))

    stats_payload = metadata.get("stats", {})
    quantized_obj = QuantizedModelWeights(
        quantized_tensors=tuple(quantized_leaves),
        passthrough_leaves=tuple(passthrough_leaves),
        stats=QuantizedWeightsStats(
            total_tensors=int(stats_payload.get("total_tensors", len(quantized_leaves) + len(passthrough_leaves))),
            quantized_tensors=int(stats_payload.get("quantized_tensors", len(quantized_leaves))),
            skipped_tensors=int(stats_payload.get("skipped_tensors", len(passthrough_leaves))),
            original_bytes=int(stats_payload.get("original_bytes", 0)),
            quantized_bytes=int(stats_payload.get("quantized_bytes", 0)),
        ),
    )

    dense_weights = dequantize_model_weights(quantized_obj)
    if return_quantized:
        return dense_weights, quantized_obj
    return dense_weights


def quantize_kv_cache(
    keys: mx.array,
    values: mx.array,
    key_bit_width: int = 3,
    value_bit_width: int = 3,
    seed: int = 0,
    pack: bool = True,
) -> QuantizedKVCache:
    key_dim = int(keys.shape[-1])
    value_dim = int(values.shape[-1])

    key_quantizer = TurboQuantProd(dimension=key_dim, bit_width=key_bit_width, seed=seed)
    value_quantizer = TurboQuantMSE(dimension=value_dim, bit_width=value_bit_width, seed=seed + 13)

    key_codes = key_quantizer.quantize(keys, pack=pack)
    value_codes = value_quantizer.quantize(values, pack=pack)

    return QuantizedKVCache(
        key_quantizer=key_quantizer,
        value_quantizer=value_quantizer,
        key_codes=key_codes,
        value_codes=value_codes,
    )


def dequantize_kv_cache(cache: QuantizedKVCache) -> tuple[mx.array, mx.array]:
    keys = cache.key_quantizer.dequantize(cache.key_codes)
    values = cache.value_quantizer.dequantize(cache.value_codes)
    return keys, values


__all__ = [
    "TURBOQUANT_CONFIG_KEY",
    "TURBOQUANT_FORMAT_VERSION",
    "TURBOQUANT_METADATA_FILENAME",
    "TURBOQUANT_WEIGHT_INDEX_FILENAME",
    "TURBOQUANT_PASSTHROUGH_INDEX_FILENAME",
    "PackedMSECodes",
    "PackedProdCodes",
    "TurboQuantMSE",
    "TurboQuantProd",
    "QuantizedKVCache",
    "QuantizedWeightsStats",
    "QuantizedWeightTensor",
    "QuantizedModelWeights",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "quantize_model_weights",
    "dequantize_model_weights",
    "turboquantize_model_weights",
    "save_turboquant_weights",
    "load_turboquant_weights",
]
