"""Investigate Avey-B [8,256,256] BF16 configuration selection.

Record Inductor's original tuning scores, replay both known configurations and
its actual chosen path in all six orders, and preserve the installed timer source.
Run separate processes with existing/fresh cache directories for comparison.
"""
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton

from avey_kernels import avey_b
from benchmarks.compare_ordering import mix_fixed_order, measure_stability
from benchmarks.profile_contextualizer import ProfileCase, _check
from benchmarks.selected_configs import capture_autotuning_choices, capture_selected_configs


def build_investigation_cases(x, gate, bias, *, compile_functions=True):
    from avey_kernels._triton import mix
    q = F.normalize(x.float(), dim=-1, eps=1e-12).contiguous()
    expected = avey_b(x, z_tl=gate, bias=bias, backend='torch')
    functions = []
    for bm, group in ((32, 1), (16, 8)):
        def fn(bm=bm, group=group):
            return mix_fixed_order(x, q, None, gate, bias, group, True, bm, 128, 4, 1)
        functions.append((f'bm{bm}_g{group}', fn))

    def tuned():
        return mix(x, q, None, gate, bias, None, 1e-6, False, True, 'identity', 128, True)
    functions.append(('autotuned', tuned))
    cases = []
    for label, fn in functions:
        if compile_functions:
            fn = torch.compile(fn, fullgraph=True, options={
                'max_autotune': label == 'autotuned',
                'coordinate_descent_tuning': False, 'triton.cudagraphs': False,
            })
        suffix = 'compiled' if compile_functions else 'eager'
        cases.append(ProfileCase(f'avey_b/mix_only/{label}/{suffix}', fn, expected))
    return cases


def save_timer_source(path):
    from torch._inductor.runtime import benchmarking, triton_heuristics as runtime
    timer = runtime.benchmarker
    objects = [runtime.CachingAutotuner.bench, type(timer).benchmark_gpu]
    clone = getattr(runtime.CachingAutotuner, 'maybe_clone_args', None)
    if clone is not None:
        objects.append(clone)
    dispatch = getattr(benchmarking, '_BENCHMARK_DISPATCH', {}).get('cuda')
    if dispatch is not None:
        objects.append(dispatch)
    sections = []
    for obj in objects:
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError) as error:
            source = f'Source unavailable: {error}'
        sections.append(f'# {obj.__module__}.{obj.__qualname__}\n{source}')
    path.write_text('\n\n'.join(sections))
    return f'{type(timer).__module__}.{type(timer).__name__}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error('iteration counts must be positive')
    if (not torch.cuda.is_available() or torch.version.hip is not None
            or torch.cuda.get_device_capability()[0] < 8):
        parser.error('requires NVIDIA SM80+')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    from torch._inductor import config as inductor_config
    report = dict(complete=False, metadata=dict(
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), shape=[8, 256, 256], dtype='bfloat16', seed=42,
        normalization_timed=False, mode='cuda_graph', rounds=6,
        warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
        matmul_precision='highest',
        inductor_settings={k: getattr(inductor_config, k, None) for k in
                           ('inductor_default_autotune_warmup', 'inductor_default_autotune_rep',
                            'autotune_local_cache', 'autotune_remote_cache', 'fx_graph_cache',
                            'force_disable_caches')},
        cache_directories={k: os.environ.get(k) for k in
                           ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')},
        benchmarker=save_timer_source(args.output_dir / 'tuning-method.txt')),
        autotuning_events=[], selected_configs={}, results=[])
    path = args.output_dir / 'report.json'
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    print(json.dumps(report['metadata']), flush=True)
    with torch.inference_mode():
        x = torch.rand(8, 256, 256, device='cuda', dtype=torch.bfloat16)
        # Preserve the previous sweep's RNG sequence for gate/bias inputs.
        torch.randn(256, 256, device='cuda', dtype=x.dtype)
        gate, bias = torch.rand_like(x), torch.rand(256, device='cuda', dtype=x.dtype)
        cases = build_investigation_cases(x, gate, bias)
        with capture_autotuning_choices() as events:
            for case in cases:
                _check(case.fn(), case.expected)
        report['autotuning_events'] = events
        report['selection_observation'] = ('candidate_scores_observed' if events
                                           else 'no_in_process_candidate_benchmark_observed')
        save()
        for case in cases:
            output, selected = capture_selected_configs(case.fn)
            _check(output, case.expected)
            report['selected_configs'][case.name] = selected
            save()
            if not any(r['status'] == 'selected' and '_mix_kernel' in r['kernel'] for r in selected):
                raise RuntimeError('Mixing launcher not captured; inspect partial report.json')
            print(f'Selected {case.name}: {json.dumps(selected)}', flush=True)
        print(f'Candidate benchmark events observed: {len(events)}', flush=True)
        for result in measure_stability(cases, warmup=args.warmup, iterations=args.iterations,
                                        repeats=args.repeats, baseline='bm32_g1', reference='bm32_g1'):
            report['results'].append(result)
            save()
            print(f'round={result["round"]} position={result["position"]} {result["case"]} '
                  f'{result["gpu_span_us"]:.3f} us vs_bm32_g1={result["speedup_vs_bm32_g1"]:.3f}x',
                  flush=True)
    report['summary'] = []
    for case in cases:
        rows = [r for r in report['results'] if r['case'] == case.name]
        summary = dict(case=case.name, median_gpu_span_us=median(r['gpu_span_us'] for r in rows),
                       median_paired_speedup_vs_bm32_g1=median(r['speedup_vs_bm32_g1'] for r in rows))
        report['summary'].append(summary)
        print(json.dumps(summary), flush=True)
    with (args.output_dir / 'timings.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(report['results'][0]))
        writer.writeheader()
        writer.writerows(report['results'])
    report['complete'] = True
    save()
    print(f'Results: {args.output_dir}/', flush=True)


if __name__ == '__main__':
    main()
