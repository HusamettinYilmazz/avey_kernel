"""Shared forward kernel; flags specialize away architecture-specific work.

Normalization is a separate linear-size pass. Each program computes a tile of
output rows/features and streams over key tiles, never storing a C x C tensor.
Feature tiling bounds register usage, at the cost of recomputing similarity tiles
for each output-feature block. Arithmetic remains FP32 for low-precision
inputs; autotuning changes tiling and launch settings, not arithmetic precision.
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
def _tile_ids(pid, n_query_tiles, n_feature_tiles, GROUP_M: tl.constexpr):
    if GROUP_M == 1:
        return pid // n_feature_tiles, pid % n_feature_tiles
    programs_per_group = GROUP_M * n_feature_tiles
    group_id = pid // programs_per_group
    first_query = group_id * GROUP_M
    group_size = tl.minimum(n_query_tiles - first_query, GROUP_M)
    pid_in_group = pid % programs_per_group
    query_tile = first_query + pid_in_group % group_size
    feature_tile = pid_in_group // group_size
    return query_tile, feature_tile


@triton.jit
def _mix_kernel(
    Z_TR, Z_TR_NORM, V, Z_TL, Bias, Valid, Out,
    C: tl.constexpr, D_PRIME: tl.constexpr, EPS: tl.constexpr,
    HAS_V: tl.constexpr, HAS_Z_TL: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
    CAUSAL: tl.constexpr, NORMALIZE: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
    BATCH: tl.constexpr = 1, DEVICE: tl.constexpr = -1, GROUP_M: tl.constexpr = 1,
):
    batch = tl.program_id(1)
    n_feature_tiles = tl.cdiv(D_PRIME, BD)
    query_tile, feature_tile = _tile_ids(
        tl.program_id(0), tl.cdiv(C, BM), n_feature_tiles, GROUP_M)
    rows = query_tile * BM + tl.arange(0, BM)
    features = feature_tile * BD + tl.arange(0, BD)
    row_valid = rows < C
    if HAS_VALID:
        # Access analysis may represent torch.bool storage as *i8. Convert
        # loaded bytes to predicates before combining them with load masks.
        row_valid = row_valid & (tl.load(Valid + batch * C + rows, rows < C, other=0) != 0)
    accumulator = tl.full((BM, BD), 0, tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    for key_start in range(tl.cdiv(C, BN)):
        keys = key_start * BN + tl.arange(0, BN)
        key_valid = keys < C
        if HAS_VALID:
            key_valid = key_valid & (tl.load(Valid + batch * C + keys, keys < C, other=0) != 0)
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


# Keep the raw JIT kernel available for fixed launches and offline compilation.
# Group 1 retains query-major order; groups 4/8 encourage input cache reuse.
# This changes the mapping, not execution order or similarity recomputation.
# All candidates overwrite Out completely; no reset hooks or input mutation.
_AUTOTUNE_CONFIGS = [
    triton.Config(dict(BM=bm, BN=bn, BK=32, BD=bd, GROUP_M=group), num_warps=4, num_stages=3)
    for bm in (16, 32) for bn in (32, 64) for bd in (32, 64, 128)
    for group in (1, 4, 8)
] + [
    triton.Config(dict(BM=bm, BN=bn, BK=32, BD=128, GROUP_M=group), num_warps=8, num_stages=2)
    for bm in (16, 32) for bn in (32, 64) for group in (1, 4, 8)
]
_mix_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    # Triton also appends tensor dtypes to its in-memory key. DEVICE separates
    # devices in a multi-GPU process; BATCH matters for launch occupancy.
    key=['BATCH', 'DEVICE', 'C', 'D_PRIME', 'EPS', 'HAS_V', 'HAS_Z_TL',
         'HAS_BIAS', 'HAS_VALID', 'CAUSAL', 'NORMALIZE', 'ACT'],
)(_mix_kernel)


@triton_op('avey_kernels::mix', mutates_args={})
def mix(
    z_tr: torch.Tensor, z_tr_norm: torch.Tensor, v: Optional[torch.Tensor],
    z_tl: Optional[torch.Tensor], bias: Optional[torch.Tensor],
    valid: Optional[torch.Tensor], eps: float, causal: bool,
    normalize: bool, activation: str, output_tile: int = 32, autotune: bool = False,
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
    args = (z_tr, z_tr_norm, v if v is not None else z_tr,
            z_tl if z_tl is not None else z_tr, bias if bias is not None else z_tr,
            valid if valid is not None else z_tr, out,
            c, d_prime, eps, v is not None, z_tl is not None,
            bias is not None, valid is not None, causal, normalize, activation_id)
    if autotune:
        def grid(meta):
            return (triton.cdiv(c, meta['BM']) * triton.cdiv(d_prime, meta['BD']), batch)
        wrap_triton(_mix_autotuned)[grid](
            *args, BATCH=batch, DEVICE=z_tr.device.index if z_tr.device.index is not None else -1)
    else:
        wrap_triton(_mix_kernel)[(triton.cdiv(c, 16) * triton.cdiv(d_prime, output_tile), batch)](
            # Inductor's argument analysis does not fill in JIT defaults.
            *args, 16, 32, 32, output_tile, BATCH=batch,
            DEVICE=z_tr.device.index if z_tr.device.index is not None else -1,
            GROUP_M=1, num_warps=4)
    return out
