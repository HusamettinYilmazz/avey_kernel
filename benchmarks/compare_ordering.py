"""Controlled ordering, causal-skip, or tile comparison; no timed normalization.

Our Triton launch configurations are fixed. The causal and tile comparisons include
an optimized PyTorch mixing baseline, with GEMM tuning completed before timing.

Add --comparison causal to compare full key traversal against skipping future tiles.
Add --comparison tiles to sweep BM=16/32 and BD=64/128/256 with causal skipping.
Add --comparison launch to sweep warps=4/8 and stages=1/2/3 at BM32/BD128.

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
                    bias: torch.Tensor, group_m: int, skip_causal: bool = True,
                    bm: int = 32, bd: int = 128,
                    num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    # Specialize a changing backed SymInt rather than forwarding it as a
    # runtime argument to a constexpr parameter.
    if isinstance(group_m, torch.SymInt):
        group_m = int(group_m)
    if type(group_m) is not int or group_m not in (1, 4, 8):
        raise ValueError('group_m must be 1, 4, or 8')
    bm = int(bm) if isinstance(bm, torch.SymInt) else bm
    bd = int(bd) if isinstance(bd, torch.SymInt) else bd
    if type(bm) is not int or bm not in (16, 32):
        raise ValueError('bm must be 16 or 32')
    if type(bd) is not int or bd not in (64, 128, 256):
        raise ValueError('bd must be 64, 128, or 256')
    num_warps = int(num_warps) if isinstance(num_warps, torch.SymInt) else num_warps
    num_stages = int(num_stages) if isinstance(num_stages, torch.SymInt) else num_stages
    if type(num_warps) is not int or num_warps not in (4, 8):
        raise ValueError('num_warps must be 4 or 8')
    if type(num_stages) is not int or num_stages not in (1, 2, 3):
        raise ValueError('num_stages must be 1, 2, or 3')
    config = {**FIXED_CONFIG, 'BM': bm, 'BD': bd,
              'num_warps': num_warps, 'num_stages': num_stages}
    batch, c, d_prime = z_tr.shape
    out = torch.empty_like(z_tr)
    wrap_triton(_mix_kernel)[(triton.cdiv(c, bm) * triton.cdiv(d_prime, bd), batch)](
        z_tr, z_tr_norm, v if v is not None else z_tr, z_tl, bias, z_tr, out,
        C=c, D_PRIME=d_prime, EPS=1e-6, HAS_V=v is not None,
        HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=False,
        CAUSAL=v is not None, NORMALIZE=v is None, ACT=0,
        BATCH=batch, DEVICE=z_tr.device.index if z_tr.device.index is not None else -1,
        GROUP_M=group_m, SKIP_CAUSAL=skip_causal, **config)
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


def torch_causal_mix(z_tr, z_tr_norm, v, z_tl, bias):
    """Same Avey mixing math and FP32 accumulation, consuming precomputed Q."""
    scores = (z_tr_norm @ z_tr_norm.transpose(-1, -2)) * v.float()
    c = z_tr.shape[1]
    allowed = torch.ones(c, c, device=z_tr.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~allowed, 0.)
    output = ((scores @ z_tr.float()) + bias.float()) * z_tl.float()
    return output.to(z_tr.dtype)


def build_causal_cases(z_tr, v, z_tl, bias, *, compile_functions=True):
    """Compare fixed full/shortened key loops and an optimized PyTorch mixing baseline."""
    z_tr_norm = F.normalize(z_tr.float(), dim=-1, eps=1e-12).contiguous()
    expected = avey(z_tr, v, z_tl=z_tl, bias=bias, backend='torch')
    cases = []
    for skip, label in ((False, 'full_keys'), (True, 'skip_future')):
        def fn(skip=skip):
            return mix_fixed_order(z_tr, z_tr_norm, v, z_tl, bias, 1, skip)
        suffix = 'compiled' if compile_functions else 'eager'
        if compile_functions:
            fn = torch.compile(fn, fullgraph=True, options={
                'max_autotune': False, 'coordinate_descent_tuning': False,
                'triton.cudagraphs': False,
            })
        cases.append(ProfileCase(f'avey/mix_only/{label}/{suffix}', fn, expected, 128))
    cases.append(_pytorch_case(z_tr, z_tr_norm, v, z_tl, bias, expected, compile_functions))
    return cases


def _pytorch_case(z_tr, z_tr_norm, v, z_tl, bias, expected, compile_functions):
    def reference():
        return torch_causal_mix(z_tr, z_tr_norm, v, z_tl, bias)

    if compile_functions:
        reference = torch.compile(reference, fullgraph=True, options={
            'max_autotune': True, 'coordinate_descent_tuning': True,
            'triton.cudagraphs': False,
        })
    suffix = 'compiled' if compile_functions else 'eager'
    return ProfileCase(f'avey/mix_only/pytorch/{suffix}', reference, expected)


# Put the previous configuration first so each result has a measured baseline.
TILE_CONFIGS = ((32, 128), (16, 64), (16, 128), (16, 256), (32, 64), (32, 256))


def build_tile_cases(z_tr, v, z_tl, bias, *, compile_functions=True):
    """Vary accumulator dimensions while fixing ordering and causal skipping."""
    z_tr_norm = F.normalize(z_tr.float(), dim=-1, eps=1e-12).contiguous()
    expected = avey(z_tr, v, z_tl=z_tl, bias=bias, backend='torch')
    cases = []
    suffix = 'compiled' if compile_functions else 'eager'
    for bm, bd in TILE_CONFIGS:
        def fn(bm=bm, bd=bd):
            return mix_fixed_order(z_tr, z_tr_norm, v, z_tl, bias, 1, True, bm, bd)
        if compile_functions:
            fn = torch.compile(fn, fullgraph=True, options={
                'max_autotune': False, 'coordinate_descent_tuning': False,
                'triton.cudagraphs': False,
            })
        cases.append(ProfileCase(f'avey/mix_only/bm{bm}_bd{bd}/{suffix}', fn, expected, bd))
    cases.append(_pytorch_case(z_tr, z_tr_norm, v, z_tl, bias, expected, compile_functions))
    return cases


LAUNCH_CONFIGS = ((4, 3), (4, 1), (4, 2), (8, 1), (8, 2), (8, 3))


def build_launch_cases(z_tr, v, z_tl, bias, *, compile_functions=True):
    """Fix BM32/BD128 and vary only warps and pipeline stages."""
    z_tr_norm = F.normalize(z_tr.float(), dim=-1, eps=1e-12).contiguous()
    expected = avey(z_tr, v, z_tl=z_tl, bias=bias, backend='torch')
    cases = []
    suffix = 'compiled' if compile_functions else 'eager'
    for warps, stages in LAUNCH_CONFIGS:
        def fn(warps=warps, stages=stages):
            return mix_fixed_order(z_tr, z_tr_norm, v, z_tl, bias,
                                   1, True, 32, 128, warps, stages)
        if compile_functions:
            fn = torch.compile(fn, fullgraph=True, options={
                'max_autotune': False, 'coordinate_descent_tuning': False,
                'triton.cudagraphs': False,
            })
        cases.append(ProfileCase(f'avey/mix_only/w{warps}_s{stages}/{suffix}', fn, expected, 128))
    cases.append(_pytorch_case(z_tr, z_tr_norm, v, z_tl, bias, expected, compile_functions))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', choices=('ordering', 'causal', 'tiles', 'launch'), default='ordering')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if min(args.batch, args.context, args.features, args.warmup,
           args.iterations, args.repeats) < 1:
        parser.error('dimensions and iteration counts must be positive')
    if (not torch.cuda.is_available() or torch.version.hip is not None
            or torch.cuda.get_device_capability()[0] < 8):
        parser.error('a working NVIDIA SM80+ GPU is required')
    if args.output_dir is None:
        args.output_dir = Path(f'gpu-profile-{args.comparison}-fixed')
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_config = dict(FIXED_CONFIG)
    varying = {'tiles': ('BM', 'BD'), 'launch': ('num_warps', 'num_stages')}
    for key in varying.get(args.comparison, ()):
        del fixed_config[key]
    metadata = dict(python=platform.python_version(), torch=torch.__version__,
                    triton=triton.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                    shape=[args.batch, args.context, args.features], dtype=args.dtype,
                    seed=42, fixed_config=fixed_config,
                    tile_configs=list(TILE_CONFIGS) if args.comparison == 'tiles' else None,
                    launch_configs=list(LAUNCH_CONFIGS) if args.comparison == 'launch' else None,
                    comparison=args.comparison,
                    group_sizes=[1, 4, 8] if args.comparison == 'ordering' else [1],
                    skip_causal_options=[False, True] if args.comparison == 'causal' else [True],
                    autotune=False, pytorch_max_autotune=args.comparison != 'ordering',
                    normalization_timed=False, mode='cuda_graph',
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
        builder = {'ordering': build_ordering_cases, 'causal': build_causal_cases,
                   'tiles': build_tile_cases, 'launch': build_launch_cases}[args.comparison]
        cases = builder(x, v, gate, bias)
        for case in cases:
            _check(case.fn(), case.expected)  # Compilation and validation outside timing.
        baselines = {}
        for case in cases:
            result = measure_case(case, 'cuda_graph', warmup=args.warmup,
                                  iterations=args.iterations, repeats=args.repeats)
            architecture, _, group, _ = case.name.split('/')
            if group in ('group1', 'full_keys', 'bm32_bd128', 'w4_s3'):
                baselines[architecture] = result['gpu_span_us']
            if args.comparison == 'ordering':
                result['group_m'] = int(group.removeprefix('group'))
                speedup_key = 'speedup_vs_group1'
            elif args.comparison == 'launch':
                result['group_m'] = None if group == 'pytorch' else 1
                result['skip_causal'] = None if group == 'pytorch' else True
                warps, stages = (None, None) if group == 'pytorch' else map(
                    int, group.removeprefix('w').split('_s'))
                result['num_warps'], result['num_stages'] = warps, stages
                speedup_key = 'speedup_vs_w4_s3'
            elif args.comparison == 'tiles':
                result['group_m'] = None if group == 'pytorch' else 1
                result['skip_causal'] = None if group == 'pytorch' else True
                bm, bd = (None, None) if group == 'pytorch' else map(
                    int, group.removeprefix('bm').split('_bd'))
                result['bm'], result['bd'] = bm, bd
                speedup_key = 'speedup_vs_bm32_bd128'
            else:
                result['group_m'] = None if group == 'pytorch' else 1
                result['skip_causal'] = None if group == 'pytorch' else group == 'skip_future'
                speedup_key = 'speedup_vs_full_keys'
            result[speedup_key] = baselines[architecture] / result['gpu_span_us']
            results.append(result)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{case.name:40s} {result["gpu_span_us"]:9.3f} us '
                  f'range=[{result["gpu_span_min_us"]:.3f}, {result["gpu_span_max_us"]:.3f}] '
                  f'speedup={result[speedup_key]:.3f}x', flush=True)
    if args.comparison != 'ordering':
        pytorch_us = next(r['gpu_span_us'] for r in results if '/pytorch/' in r['case'])
        print('Speedup relative to compiled PyTorch mixing:', flush=True)
        for result in results:
            result['speedup_vs_pytorch'] = pytorch_us / result['gpu_span_us']
            print(f'{result["case"]:40s} {result["speedup_vs_pytorch"]:.3f}x', flush=True)
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'timings.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'Results: {args.output_dir}/ (report.json, timings.csv)', flush=True)


if __name__ == '__main__':
    main()
