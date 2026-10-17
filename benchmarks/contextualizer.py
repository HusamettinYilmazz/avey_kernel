"""Compare the fused forward operators against eager and compiled references.

Run from an installed checkout: python3 -m benchmarks.contextualizer
"""

import argparse

import torch

from avey_kernels import avey, avey_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('a working NVIDIA GPU/driver is required for timings')
    if min(args.batch, args.context, args.features) < 1:
        parser.error('dimensions must be positive')
    import triton

    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    dtype = getattr(torch, args.dtype)
    # Nonnegative right-tail features model a ReLU-family enricher and avoid
    # ill-conditioned row denominators. Signed behavior has separate tests.
    z_tr = torch.rand(args.batch, args.context, args.features, device='cuda', dtype=dtype)
    v = torch.randn(args.context, args.context, device='cuda', dtype=dtype)
    z_tl = torch.rand_like(z_tr)
    bias = torch.rand(args.features, device='cuda', dtype=dtype)
    print(f'{torch.cuda.get_device_name()}; torch={torch.__version__}; '
          f'triton={triton.__version__}; shape={tuple(z_tr.shape)}; dtype={dtype}')
    for name, op, extra in [('Avey', avey, (v,)), ('Avey-B dynamic', avey_b, ())]:
        def reference(x):
            return op(x, *extra, z_tl=z_tl, bias=bias, backend='torch')

        def fused(x):
            return op(x, *extra, z_tl=z_tl, bias=bias, backend='triton')

        variants = [('eager reference', reference),
                    ('compiled reference', torch.compile(reference, fullgraph=True, mode='max-autotune')),
                    ('fused', fused),
                    ('compiled fused', torch.compile(fused, fullgraph=True, mode='max-autotune'))]
        with torch.inference_mode():
            expected = reference(z_tr)
            for label, fn in variants:
                # Compile, launch, and check correctness before timing.
                torch.testing.assert_close(fn(z_tr), expected, atol=5e-3, rtol=5e-3)
                ms = triton.testing.do_bench(lambda: fn(z_tr), warmup=25, rep=100)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = fn(z_tr)
                torch.cuda.synchronize()
                extra_mib = (torch.cuda.max_memory_allocated() - baseline) / 2**20
                del result
                print(f'{name:15s} {label:20s} {ms:9.4f} ms  peak-extra={extra_mib:.2f} MiB')


if __name__ == '__main__':
    main()
