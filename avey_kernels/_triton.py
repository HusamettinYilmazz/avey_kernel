"""Shared forward kernel; flags specialize away architecture-specific work.

Normalization is a separate linear-size pass. Each program computes a tile of
output rows/features and streams over key tiles, never storing a C x C tensor.
Feature tiling bounds register usage, at the cost of recomputing similarity tiles
for each output-feature block. This initial implementation prioritizes FP32
numerics; it is not a tuned tensor-core implementation for low-precision input.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _stable_sigmoid(x):
    e = tl.exp(-tl.abs(x))
    return tl.where(x >= 0., 1. / (1. + e), e / (1. + e))


@triton.jit
def _mix_kernel(
    Z_TR, Z_TR_NORM, V, Z_TL, Bias, Valid, Out,
    C: tl.constexpr, D_PRIME: tl.constexpr, EPS: tl.constexpr,
    HAS_V: tl.constexpr, HAS_Z_TL: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
    CAUSAL: tl.constexpr, NORMALIZE: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
):
    batch = tl.program_id(1)
    n_feature_tiles = tl.cdiv(D_PRIME, BD)
    query_tile = tl.program_id(0) // n_feature_tiles
    feature_tile = tl.program_id(0) % n_feature_tiles
    rows = query_tile * BM + tl.arange(0, BM)
    features = feature_tile * BD + tl.arange(0, BD)
    row_valid = rows < C
    if HAS_VALID:
        row_valid = row_valid & tl.load(Valid + batch * C + rows, rows < C, other=False)
    accumulator = tl.full((BM, BD), 0, tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    for key_start in range(tl.cdiv(C, BN)):
        keys = key_start * BN + tl.arange(0, BN)
        key_valid = keys < C
        if HAS_VALID:
            key_valid = key_valid & tl.load(Valid + batch * C + keys, keys < C, other=False)
        s_tile = tl.full((BM, BN), 0, tl.float32)
        for feature_start in range(tl.cdiv(D_PRIME, BK)):
            inner = feature_start * BK + tl.arange(0, BK)
            z_tr_norm_rows = tl.load(
                Z_TR_NORM + batch * C * D_PRIME + rows[:, None] * D_PRIME + inner[None, :],
                row_valid[:, None] & (inner[None, :] < D_PRIME), other=0)
            z_tr_norm_keys = tl.load(
                Z_TR_NORM + batch * C * D_PRIME + keys[None, :] * D_PRIME + inner[:, None],
                key_valid[None, :] & (inner[:, None] < D_PRIME), other=0)
            s_tile += tl.dot(z_tr_norm_rows, z_tr_norm_keys, input_precision='tf32x3')
        if HAS_V:
            v_tile = tl.load(V + rows[:, None] * C + keys[None, :],
                             (rows[:, None] < C) & (keys[None, :] < C), other=0).to(tl.float32)
            s_tile *= v_tile
        allowed = row_valid[:, None] & key_valid[None, :]
        if CAUSAL:
            allowed = allowed & (keys[None, :] <= rows[:, None])
        s_tile = tl.where(allowed, s_tile, 0.)
        if NORMALIZE:
            denominator += tl.sum(s_tile, axis=1)
        z_tr_tile = tl.load(Z_TR + batch * C * D_PRIME + keys[:, None] * D_PRIME + features[None, :],
                           key_valid[:, None] & (features[None, :] < D_PRIME), other=0).to(tl.float32)
        accumulator += tl.dot(s_tile, z_tr_tile, input_precision='tf32x3')
    if NORMALIZE:
        accumulator /= denominator[:, None] + EPS
    if HAS_BIAS:
        bias = tl.load(Bias + features, features < D_PRIME, other=0).to(tl.float32)
        accumulator += bias[None, :]
    if ACT == 1:
        accumulator = tl.maximum(accumulator, 0.)
    elif ACT == 2:
        accumulator = tl.maximum(accumulator, 0.)
        accumulator *= accumulator
    elif ACT == 3:
        accumulator *= _stable_sigmoid(accumulator)
    elif ACT == 4:
        x = accumulator
        t = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        accumulator = x * _stable_sigmoid(2. * t)
    if HAS_Z_TL:
        z_tl = tl.load(Z_TL + batch * C * D_PRIME + rows[:, None] * D_PRIME + features[None, :],
                       row_valid[:, None] & (features[None, :] < D_PRIME), other=0).to(tl.float32)
        accumulator *= z_tl
    accumulator = tl.where(row_valid[:, None], accumulator, 0.)
    tl.store(Out + batch * C * D_PRIME + rows[:, None] * D_PRIME + features[None, :],
             accumulator, (rows[:, None] < C) & (features[None, :] < D_PRIME))


@triton_op('avey_kernels::mix', mutates_args={})
def mix(
    z_tr: torch.Tensor, z_tr_norm: torch.Tensor, v: Optional[torch.Tensor],
    z_tl: Optional[torch.Tensor], bias: Optional[torch.Tensor],
    valid: Optional[torch.Tensor], eps: float, causal: bool,
    normalize: bool, activation: str, output_tile: int = 32,
) -> torch.Tensor:
    # Dynamo may generalize a changing scalar argument to a backed SymInt.
    # BD and the launch grid must remain compile-time constants: converting
    # it to int installs a value guard and specializes this kernel per width.
    if isinstance(output_tile, torch.SymInt):
        output_tile = int(output_tile)
    if type(output_tile) is not int or output_tile not in (32, 64, 128):
        raise ValueError('output_tile must be 32, 64, or 128')
    out = torch.empty(z_tr.shape, device=z_tr.device, dtype=z_tr.dtype)
    batch, c, d_prime = z_tr.shape
    activation_id = {'identity': 0, 'relu': 1, 'relu2': 2, 'silu': 3, 'gelu': 4}[activation]
    # Dummy pointers have dead loads: constexpr flags remove unused arguments.
    wrap_triton(_mix_kernel)[(triton.cdiv(c, 16) * triton.cdiv(d_prime, output_tile), batch)](
        z_tr, z_tr_norm, v if v is not None else z_tr,
        z_tl if z_tl is not None else z_tr, bias if bias is not None else z_tr,
        valid if valid is not None else z_tr, out,
        c, d_prime, eps, v is not None, z_tl is not None,
        bias is not None, valid is not None, causal, normalize, activation_id,
        16, 32, 32, output_tile, num_warps=4,
    )
    return out
