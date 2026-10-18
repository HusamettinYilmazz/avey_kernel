"""Warm and check one benchmark case, then expose one launch to Nsight Compute.

Use ncu --profile-from-start off --kernel-name 'regex:.*mix_kernel.*'
with --launch-count 1. Compiler/autotuner activity happens before profilerStart.
Nsight instrumentation timings are diagnostic, not benchmark measurements.
"""
import argparse
import json

import torch

from benchmarks.profile_contextualizer import build_cases, _check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('avey', 'avey_b'), default='avey')
    parser.add_argument('--variant', choices=('autotuned', 'fixed', 'reference'), default='autotuned')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    if min(args.batch, args.context, args.features, args.warmup) < 1:
        parser.error('dimensions and warmup must be positive')
    if (not torch.cuda.is_available() or torch.version.hip is not None
            or torch.cuda.get_device_capability()[0] < 8):
        parser.error('a working NVIDIA SM80+ GPU is required')
    import triton

    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    with torch.inference_mode():
        x = torch.rand(args.batch, args.context, args.features,
                       device='cuda', dtype=getattr(torch, args.dtype))
        v = torch.randn(args.context, args.context, device='cuda', dtype=x.dtype)
        gate = torch.rand_like(x)
        bias = torch.rand(args.features, device='cuda', dtype=x.dtype)
        suffix = {'autotuned': 'fused/autotuned/compiled',
                  'fixed': 'fused/bd128/compiled',
                  'reference': 'reference_compiled'}[args.variant]
        name = f'{args.operation}/{suffix}'
        cases = build_cases(x, v, gate, bias, output_tiles=(128,),
                            autotune=args.variant == 'autotuned')
        case = next(case for case in cases if case.name == name)
        print(json.dumps(dict(case=name, shape=list(x.shape), dtype=args.dtype,
                              gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                              triton=triton.__version__, cuda=torch.version.cuda)), flush=True)
        _check(case.fn(), case.expected)  # Compile/autotune and validate before profiling.
        for _ in range(args.warmup):
            output = case.fn()
        torch.cuda.synchronize()
        print('Warmup complete; profiling one invocation.', flush=True)
        runtime = torch.cuda.cudart()
        torch.cuda.check_error(runtime.cudaProfilerStart())
        try:
            output = case.fn()
            torch.cuda.synchronize()
        finally:
            torch.cuda.check_error(runtime.cudaProfilerStop())
        _check(output, case.expected)
        print('Profiled invocation passed correctness check.', flush=True)


if __name__ == '__main__':
    main()
