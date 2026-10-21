"""Compare only GROUP_M=1/4/8; fixed tiling, no autotuning or timed normalization.

python3 -m benchmarks.compare_ordering --output-dir gpu-profile-ordering-fixed
"""
import argparse
import csv
import json
from pathlib import Path
import platform
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from torch.library import triton_op, wrap_triton

from avey_kernels import avey, avey_b
from avey_kernels._triton import _mix_kernel
from benchmarks.profile_contextualizer import ProfileCase, _check, measure_case


FIXED_CONFIG = dict(BM=32, BN=64, BK=32, BD=128, num_warps=4, num_stages=3)


@triton_op('avey_bench::mix_fixed_order', mutates_args={})
def mix_fixed_order(z_tr: torch.Tensor, z_tr_norm: torch.Tensor,
                    v: Optional[torch.Tensor], z_tl: torch.Tensor,
                    bias: torch.Tensor, group_m: int) -> torch.Tensor:
    # Specialize a changing backed SymInt rather than forwarding it as a
    # runtime argument to a constexpr parameter.
    if isinstance(group_m, torch.SymInt):
        group_m = int(group_m)
    if type(group_m) is not int or group_m not in (1, 4, 8):
        raise ValueError('group_m must be 1, 4, or 8')
    batch, c, d_prime = z_tr.shape
    out = torch.empty_like(z_tr)
    wrap_triton(_mix_kernel)[(triton.cdiv(c, 32) * triton.cdiv(d_prime, 128), batch)](
        z_tr, z_tr_norm, v if v is not None else z_tr, z_tl, bias, z_tr, out,
        C=c, D_PRIME=d_prime, EPS=1e-6, HAS_V=v is not None,
        HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=False,
        CAUSAL=v is not None, NORMALIZE=v is None, ACT=0,
        BATCH=batch, DEVICE=z_tr.device.index if z_tr.device.index is not None else -1,
        GROUP_M=group_m, **FIXED_CONFIG)
    return out


def build_ordering_cases(z_tr, v, z_tl, bias, *, compile_functions=True):
    """All six cases share one precomputed N(Z_tr) and identical input buffers."""
    z_tr_norm = F.normalize(z_tr.float(), dim=-1, eps=1e-12).contiguous()
    cases = []
    for name, op, extra in (('avey', avey, (v,)), ('avey_b', avey_b, ())):
        expected = op(z_tr, *extra, z_tl=z_tl, bias=bias, backend='torch')
        for group in (1, 4, 8):
            def fn(group=group, weighted=(name == 'avey')):
                return mix_fixed_order(z_tr, z_tr_norm, v if weighted else None,
                                       z_tl, bias, group)
            suffix = 'compiled' if compile_functions else 'eager'
            if compile_functions:
                fn = torch.compile(fn, fullgraph=True, options={
                    'max_autotune': False, 'coordinate_descent_tuning': False,
                    'triton.cudagraphs': False,
                })
            cases.append(ProfileCase(f'{name}/mix_only/group{group}/{suffix}', fn, expected, 128))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, default=Path('gpu-profile-ordering-fixed'))
    args = parser.parse_args()
    if min(args.batch, args.context, args.features, args.warmup,
           args.iterations, args.repeats) < 1:
        parser.error('dimensions and iteration counts must be positive')
    if (not torch.cuda.is_available() or torch.version.hip is not None
            or torch.cuda.get_device_capability()[0] < 8):
        parser.error('a working NVIDIA SM80+ GPU is required')
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(python=platform.python_version(), torch=torch.__version__,
                    triton=triton.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                    shape=[args.batch, args.context, args.features], dtype=args.dtype,
                    seed=42, fixed_config=FIXED_CONFIG, group_sizes=[1, 4, 8],
                    autotune=False, normalization_timed=False, mode='cuda_graph',
                    warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
                    matmul_precision='highest')
    results = []
    report = dict(metadata=metadata, results=results, complete=False)
    report_path = args.output_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
    with torch.inference_mode():
        x = torch.rand(*metadata['shape'], device='cuda', dtype=getattr(torch, args.dtype))
        v = torch.randn(args.context, args.context, device='cuda', dtype=x.dtype)
        gate, bias = torch.rand_like(x), torch.rand(args.features, device='cuda', dtype=x.dtype)
        cases = build_ordering_cases(x, v, gate, bias)
        for case in cases:
            _check(case.fn(), case.expected)  # Compilation and validation outside timing.
        baselines = {}
        for case in cases:
            result = measure_case(case, 'cuda_graph', warmup=args.warmup,
                                  iterations=args.iterations, repeats=args.repeats)
            architecture, _, group, _ = case.name.split('/')
            if group == 'group1':
                baselines[architecture] = result['gpu_span_us']
            result['group_m'] = int(group.removeprefix('group'))
            result['speedup_vs_group1'] = baselines[architecture] / result['gpu_span_us']
            results.append(result)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{case.name:40s} {result["gpu_span_us"]:9.3f} us '
                  f'range=[{result["gpu_span_min_us"]:.3f}, {result["gpu_span_max_us"]:.3f}] '
                  f'speedup={result["speedup_vs_group1"]:.3f}x', flush=True)
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'timings.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'Results: {args.output_dir}/ (report.json, timings.csv)', flush=True)


if __name__ == '__main__':
    main()
