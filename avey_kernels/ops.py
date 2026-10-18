"""Public operations and differentiable PyTorch reference implementations.

Paper notation: z_tr = Z_tr, z_tl = Z_tl, v = V, z_tr_norm = N(Z_tr),
and s = S. Features have shape [batch, C, d_prime], with d_prime = m_t / 2
in original Avey. V is [query position, key position], shared across the
batch. Avey-B's static layers are ordinary matrix multiplications and are
outside this module.
"""

import math

import torch
import torch.nn.functional as F

try:
    from ._triton import mix as _mix
except ModuleNotFoundError as exc:
    if exc.name != 'triton':
        raise
    _mix = None


def _validate(z_tr, v, z_tl, bias, valid, eps, norm_eps, activation, backend, output_tile, autotune):
    if z_tr.ndim != 3 or any(size == 0 for size in z_tr.shape):
        raise ValueError('z_tr must have nonempty shape [batch, context, features]')
    if z_tr.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError('z_tr must be floating point')
    if backend not in ('auto', 'torch', 'triton'):
        raise ValueError('backend must be auto, torch, or triton')
    if type(output_tile) is not int or output_tile not in (32, 64, 128):
        raise ValueError('output_tile must be 32, 64, or 128')
    if type(autotune) is not bool:
        raise ValueError('autotune must be boolean')
    if activation not in ('identity', 'relu', 'relu2', 'silu', 'gelu'):
        raise ValueError('unsupported activation')
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('eps and norm_eps must be finite and positive')
    c, d_prime = z_tr.shape[1:]
    for name, tensor, shape in (('v', v, (c, c)),
                                ('z_tl', z_tl, z_tr.shape), ('bias', bias, (d_prime,))):
        if tensor is not None and (tensor.shape != shape or tensor.device != z_tr.device
                                   or tensor.dtype != z_tr.dtype):
            raise ValueError(f'{name} must have shape {shape}, z_tr.dtype, and z_tr.device')
    if valid is not None and (valid.shape != z_tr.shape[:2] or valid.dtype != torch.bool
                              or valid.device != z_tr.device):
        raise ValueError('valid must be boolean [batch, context] on z_tr.device')


def _activate(x, activation):
    if activation == 'relu':
        return F.relu(x)
    if activation == 'relu2':
        return F.relu(x).square()
    if activation == 'silu':
        return F.silu(x)
    if activation == 'gelu':
        return F.gelu(x, approximate='tanh')
    return x


def _reference(z_tr, v, z_tl, bias, valid, eps, norm_eps, causal, normalize, activation):
    # Accumulate low-precision input in FP32, preserving FP64 for gradcheck.
    z_tr_acc = z_tr if z_tr.dtype == torch.float64 else z_tr.float()
    if valid is not None:
        # Mask before arithmetic: masking NaNs after matmul is too late for
        # autograd, whose zero upstream gradients can still multiply NaNs.
        z_tr_acc = z_tr_acc.masked_fill(~valid[:, :, None], 0.)
    z_tr_norm = F.normalize(z_tr_acc, p=2, dim=-1, eps=norm_eps)
    s = z_tr_norm @ z_tr_norm.transpose(-1, -2)
    if v is not None:
        s = s * v.to(z_tr_acc.dtype)
    if causal:
        c = z_tr.shape[1]
        allowed = torch.ones(c, c, device=z_tr.device, dtype=torch.bool).tril()
        s = s.masked_fill(~allowed, 0.)
    if valid is not None:
        s = s.masked_fill(~valid[:, None, :], 0.)
    numerator = s @ z_tr_acc
    if normalize:
        numerator = numerator / (s.sum(dim=-1, keepdim=True) + eps)
    if bias is not None:
        numerator = numerator + bias.to(z_tr_acc.dtype)
    out = _activate(numerator, activation)
    if z_tl is not None:
        clean_z_tl = z_tl.to(z_tr_acc.dtype)
        if valid is not None:
            clean_z_tl = clean_z_tl.masked_fill(~valid[:, :, None], 0.)
        out = out * clean_z_tl
    if valid is not None:
        out = out.masked_fill(~valid[:, :, None], 0.)
    return out.to(z_tr.dtype)


def _dispatch(z_tr, v, z_tl, bias, valid, eps, norm_eps, causal, normalize, activation, backend, output_tile, autotune):
    _validate(z_tr, v, z_tl, bias, valid, eps, norm_eps, activation, backend, output_tile, autotune)
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (z_tr, v, z_tl, bias))
    supported = (_mix is not None and z_tr.is_cuda and torch.version.hip is None
                 and z_tr.dtype != torch.float64 and not needs_grad
                 and torch.cuda.get_device_capability(z_tr.device)[0] >= 8)
    if backend == 'triton' and not supported:
        raise ValueError('Triton requires Triton installed, NVIDIA SM80+, and CUDA '
                         'FP16/BF16/FP32 inference inputs; use backend="torch" for autograd')
    if backend == 'torch' or not supported:
        return _reference(z_tr, v, z_tl, bias, valid, eps, norm_eps,
                          causal, normalize, activation)
    # Keep normalization separate: linear-size storage, no pairwise intermediates.
    z_tr_norm = F.normalize(z_tr.float(), p=2, dim=-1, eps=norm_eps)
    return _mix(z_tr.contiguous(), z_tr_norm.contiguous(),
               v.contiguous() if v is not None else None,
               z_tl.contiguous() if z_tl is not None else None,
               bias.contiguous() if bias is not None else None,
               valid.contiguous() if valid is not None else None,
               eps, causal, normalize, activation, output_tile, autotune)


def avey(z_tr: torch.Tensor, v: torch.Tensor, *, z_tl=None, bias=None,
         valid=None, causal=True, norm_eps=1e-12, activation='identity', backend='auto',
         output_tile=32, autotune=False):
    """z_tl * activation(((v * cosine(z_tr, z_tr)) masked) @ z_tr + bias).

    Causal masking includes the diagonal. `valid` masks both query and key
    positions; invalid output rows are zero, including after bias/gating.
    `bias` has shape [features]; `z_tl`, if present, has the same shape as z_tr.
    `output_tile` selects an experimental Triton feature tile (32/64/128).
    With `autotune=True`, Triton benchmarks 16 configurations on first use and
    reuses the winner in this process; `output_tile` is then ignored. Warm up
    outside CUDA-graph capture. CPU/autograd fallback keeps reference math.
    """
    if v is None:
        raise ValueError('Avey requires the learned V matrix (v)')
    return _dispatch(z_tr, v, z_tl, bias, valid, 1e-6, norm_eps,
                     causal, False, activation, backend, output_tile, autotune)


def avey_b(z_tr: torch.Tensor, *, z_tl=None, bias=None, valid=None, eps=1e-6,
           norm_eps=1e-12, activation='identity', backend='auto', output_tile=32, autotune=False):
    """Avey-B dynamic mixing: z_tl * activation((S @ z_tr)/(sum(S)+eps) + bias).

    S is cosine similarity with invalid keys zeroed. Signed similarities are
    preserved; denominators are not clamped or replaced with softmax.
    GELU, when requested, uses PyTorch's tanh approximation.
    `output_tile` selects the Triton output-feature tile: 32 (default), 64, or 128.
    `autotune=True` overrides it with a measured configuration, as in `avey`.
    """
    return _dispatch(z_tr, None, z_tl, bias, valid, eps, norm_eps,
                     False, True, activation, backend, output_tile, autotune)
