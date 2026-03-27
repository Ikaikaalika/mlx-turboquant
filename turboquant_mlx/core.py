from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np


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
    "PackedMSECodes",
    "PackedProdCodes",
    "TurboQuantMSE",
    "TurboQuantProd",
    "QuantizedKVCache",
    "quantize_kv_cache",
    "dequantize_kv_cache",
]
