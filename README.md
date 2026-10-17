# Avey contextualizer kernels

Experimental standalone forward implementations of the two contextualizers in
`papers/Dontpayattention.pdf` (equation 2) and `papers/Avey-B.pdf` (equations 5–7).
These are the token-mixing operations, not full model implementations or an
Inductor graph-rewrite pass. No authors' source code or checkpoints are included.

Names follow the papers:

| Code | Paper | Meaning |
|---|---|---|
| `z_tr` / `Z_TR` | Z_tr | Right half of the enriched tail |
| `z_tl` / `Z_TL` | Z_tl | Left half of the tail, used as the gate |
| `v` / `V` | V | Learned cross-token mixing matrix |
| `z_tr_norm` / `Z_TR_NORM` | N(Z_tr) | Row-wise L2-normalized right-tail features |
| `s` / `s_tile` | S | Cosine similarities (or a local tile) before weighting/masking |
| `d_prime` / `D_PRIME` | d′ = m_t / 2 | Right-tail feature dimension |

`C` is context width; an additional batch axis groups independent contexts.
Public keyword arguments use `z_tr`, `v`, and `z_tl`; the earlier names
`r`, `weight`, and `gate` have been replaced.

For right-tail features `z_tr` of shape `[batch, context, features]`, define
`z_tr_norm = z_tr / max(norm(z_tr, dim=-1), norm_eps)` and `s = z_tr_norm @ z_tr_norm.transpose(-1, -2)`.

- **Avey:** `z_tl * activation(((v * s) masked) @ z_tr + bias)`.
- **Avey-B dynamic:** `z_tl * activation((s @ z_tr) / (s.sum(-1, keepdim=True) + eps) + bias)`.

`v[i, j]` weights key position `j` for query position `i`, shared across the
batch. Avey defaults to lower-triangular masking including the diagonal. Avey-B
is bidirectional. `valid` masks query and key positions; padded outputs are zero
even after bias/gating. Bias is a shared feature vector `[features]`. `z_tl` is an
optional tensor shaped like `z_tr`. Activations: `identity` (default), `relu`,
`relu2`, `silu`, `gelu` (tanh approximation). The default norm epsilon is `1e-12`;
Avey-B's row-denominator epsilon is `1e-6`. Signed scores are retained: the sum
can be small or negative, and is deliberately not clamped or softmax-normalized.
For signed inputs, `eps` alone does not guarantee numerical conditioning.

The ranker, compressor, enricher, fuser, full-model integration, and Avey-B's
static GEMM layers are outside this package. Optional choices above must be
matched to the actual model configuration before using a checkpoint.

## Use

Python 3.10+, PyTorch 2.6+, and (for the GPU path) Triton 3.2+ on NVIDIA SM80+
are required. Development verification uses PyTorch 2.13.0 / Triton 3.7.1;
the declared minimum versions have not been independently tested.

```bash
python3 -m pip install -e '.[cuda]'
```

```python
import torch
from avey_kernels import avey, avey_b

z_tr = torch.rand(8, 128, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(128, 128, device='cuda', dtype=z_tr.dtype)

with torch.inference_mode():
    y = avey(z_tr, v)       # learned weighting + causal mixing
    z = avey_b(z_tr)             # bidirectional, sum-normalized mixing
    compiled = torch.compile(avey_b, fullgraph=True)
    z_compiled = compiled(z_tr)
```

`backend='auto'` selects Triton for supported CUDA inference inputs and the
differentiable PyTorch reference otherwise. `backend='torch'` forces the
reference. `backend='triton'` rejects unsupported devices, dtypes, or gradient
requests. FP16, BF16, and FP32 GPU inputs are supported; accumulation and L2
normalization use FP32. CPU/FP64 and training use the reference. Noncontiguous
inputs are copied to contiguous storage for the GPU kernel.

## How fusion works

1. Normalize right-tail features in a separate linear-size pass.
2. A shared Triton template streams over key tiles to compute score tiles and
   immediately accumulate their contributions to the output.
3. Compile-time flags select Avey's learned weighting/causality or Avey-B's
   denominator accumulation. Optional bias, activation, and z_tl run before the
   output store. No pairwise similarity or mixing matrix is written to memory.

The two public entry points share one `torch.library.triton_op`, so
`torch.compile` can capture the kernel via `wrap_triton`. This explicitly
supplies the implementation; ordinary PyTorch expressions are not automatically
rewritten to these operators.

The initial fixed tiling is 16 query rows × 32 keys × 32 output features, with
32-feature score reduction chunks. Score tiles are recomputed for each output
feature tile to bound register use. All dot products use FP32 operands with
`tf32x3` precision. This favors correctness over low-precision throughput;
autotuning, tensor-core dtype specialization, associative Avey-B, and fused
backward kernels are future work. No speedup is claimed without GPU benchmarks.

## Verify and benchmark

Run from this directory; no test framework installation is needed:

```bash
python3 -m unittest discover -s tests -v
TRITON_INTERPRET=1 python3 -m unittest discover -s tests -v
AVEY_TEST_CUDA_COMPILE=1 TRITON_CACHE_DIR=/tmp/avey-triton-cache python3 -m unittest discover -s tests -p test_cuda_compile.py -v
python3 -m benchmarks.contextualizer --batch 8 --context 128 --features 128
```

The regular suite runs reference checks and, with CUDA available, GPU parity and
full-graph compilation tests. The interpreter suite executes the real tiled
kernel on CPU, covering multiple tiles, irregular dimensions, signed/zero
inputs, padding (including NaN padding), weights, gates, biases, and activations.
Interpreter success is not evidence of GPU runtime correctness or performance.
Triton 3.7's interpreter truncates BF16 output conversions, so that specific
comparison permits one BF16 ULP; CUDA tests retain their stricter tolerances.
The optional offline compilation test generates SM80 CUDA binaries for both
variants in FP32, FP16, and BF16 without launching them or requiring a driver.
The benchmark compares eager and compiled references with eager and compiled
fused implementations, checking results before timing. Compilation is excluded
from timings. Peak extra allocation is also reported; compiler-managed buffer
reuse can affect that measurement.
