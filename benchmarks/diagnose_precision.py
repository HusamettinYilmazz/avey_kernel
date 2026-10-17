"""Inspect rounding differences for the CUDA integration test's exact inputs.

Run on the GPU host: python3 -m benchmarks.diagnose_precision
CPU mode checks the reporting/reference path only: --device cpu.
This deliberately does not change test tolerances or kernel arithmetic.
"""

import argparse

import torch

from avey_kernels import avey, avey_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is required; --device cpu only checks reference reporting')
    print(f'torch={torch.__version__}, device={args.device}, '
          f'matmul_precision={torch.get_float32_matmul_precision()}')
    if args.device == 'cuda':
        import triton
        print(f'GPU={torch.cuda.get_device_name()}, triton={triton.__version__}')

    torch.manual_seed(7)
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            z_tr = torch.rand(2, 35, 37, device=args.device, dtype=dtype)
            v = torch.randn(35, 35, device=args.device, dtype=dtype)
            # Consume the same random draws as the integration test, even
            # though its failing comparison omits gate and bias.
            torch.rand_like(z_tr)
            torch.randn(37, device=args.device, dtype=dtype)
            for name, op, extra in (('Avey', avey, (v,)), ('Avey-B', avey_b, ())):
                reference = op(z_tr, *extra, backend='torch')
                reference_fp32 = op(z_tr.float(), *(x.float() for x in extra), backend='torch')
                oracle = op(z_tr.double(), *(x.double() for x in extra), backend='torch')
                print(f'\n{name}, input/output dtype={dtype}')
                print('  reference FP32 max abs error vs FP64:',
                      (reference_fp32.double() - oracle).abs().max().item())
                if args.device != 'cuda':
                    continue
                actual = op(z_tr, *extra, backend='triton')
                actual_fp32 = op(z_tr.float(), *(x.float() for x in extra), backend='triton')
                print('  fused FP32 max abs error vs FP64:',
                      (actual_fp32.double() - oracle).abs().max().item())
                print('  FP32 fused/reference max abs difference:',
                      (actual_fp32 - reference_fp32).abs().max().item())
                print('  native output equals FP32 fused cast to input dtype:',
                      torch.equal(actual, actual_fp32.to(dtype)))
                mismatch = ~torch.isclose(actual, reference, atol=2e-3, rtol=2e-3)
                print('  elements failing current tolerance:', mismatch.sum().item())
                for location in mismatch.nonzero()[:5].tolist():
                    index = tuple(location)
                    print(f'  index={index}: fused={actual[index].item()}, '
                          f'reference={reference[index].item()}, '
                          f'fused_fp32={actual_fp32[index].item()}, '
                          f'reference_fp32={reference_fp32[index].item()}, '
                          f'oracle_fp64={oracle[index].item()}')


if __name__ == '__main__':
    main()
