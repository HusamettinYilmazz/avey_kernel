"""Profile normalization, mixing, and full contextualizers on NVIDIA SM80+.

python3 -m benchmarks.profile_contextualizer --output-dir gpu-profile
All compilation/warmup precedes timing. Inductor's automatic CUDA graphs are
disabled; every case is measured with ordinary launches and explicit replay.

Outputs: report.json (environment and timing samples summarized by median),
timings.csv, trace.json (Chrome/Perfetto trace), operators.txt (profiler table).
Use the trace for actual kernel durations/counts. CUDA-event spans include GPU
idle gaps while Python dispatches work; host dispatch also includes allocations
and wrapper work, not just the CUDA launch API. Never subtract these columns to
derive a kernel duration. Isolated component times need not add to full-call
times because allocation, caching, and compiler fusion differ.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Callable

import torch
import torch.nn.functional as F

from avey_kernels import avey, avey_b


@dataclass
class ProfileCase:
    name: str
    fn: Callable[[], torch.Tensor]
    expected: torch.Tensor


def build_cases(z_tr, v, z_tl, bias, *, include_triton=True, compile_functions=True):
    """Mix-only cases consume a precomputed normalized buffer, excluding its cost."""
    cases = []

    def add_pair(name, fn, expected):
        cases.append(ProfileCase(name + '/eager', fn, expected))
        if compile_functions:
            compiled = torch.compile(fn, fullgraph=True, options={
                'max_autotune': True, 'coordinate_descent_tuning': True,
                'triton.cudagraphs': False,
            })
            cases.append(ProfileCase(name + '/compiled', compiled, expected))

    def normalize():
        return F.normalize(z_tr.float(), dim=-1, eps=1e-12)

    z_tr_norm = normalize().contiguous()
    add_pair('normalization', normalize, z_tr_norm)
    for name, op, extra in (('avey', avey, (v,)), ('avey_b', avey_b, ())):
        def reference(op=op, extra=extra):
            return op(z_tr, *extra, z_tl=z_tl, bias=bias, backend='torch')

        expected = reference()
        # Keep architecture and implementation together in the trace labels.
        cases.append(ProfileCase(name + '/reference_eager', reference, expected))
        if compile_functions:
            compiled = torch.compile(reference, fullgraph=True, options={
                'max_autotune': True, 'coordinate_descent_tuning': True,
                'triton.cudagraphs': False,
            })
            cases.append(ProfileCase(name + '/reference_compiled', compiled, expected))
        if include_triton:
            from avey_kernels._triton import mix

            def fused(op=op, extra=extra):
                return op(z_tr, *extra, z_tl=z_tl, bias=bias, backend='triton')

            def mixing_only(is_avey=(name == 'avey')):
                return mix(z_tr, z_tr_norm, v if is_avey else None, z_tl, bias,
                           None, 1e-6, is_avey, not is_avey, 'identity')

            add_pair(name + '/fused', fused, expected)
            add_pair(name + '/mix_only', mixing_only, expected)
    return cases


def _check(actual, expected):
    torch.testing.assert_close(actual, expected, atol=5e-3,
                               rtol=max(5e-3, torch.finfo(expected.dtype).eps))


@torch.inference_mode()
def measure_case(case, mode, *, warmup, iterations, repeats):
    """GPU span includes idle gaps between launches; host dispatch excludes sync.

    These measurements overlap and must not be subtracted as independent costs.
    Kernel execution durations and launch counts should be read from the trace.
    """
    if mode not in ('launch', 'cuda_graph'):
        raise ValueError('mode must be launch or cuda_graph')
    if min(warmup, iterations, repeats) < 1:
        raise ValueError('warmup, iterations, and repeats must be positive')
    _check(case.fn(), case.expected)  # Compile and check before capture/timing.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            output = case.fn()
    stream.synchronize()
    graph = None
    if mode == 'cuda_graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = case.fn()
        call = graph.replay
        call()
        torch.cuda.synchronize()
        _check(output, case.expected)
    else:
        call = case.fn
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        wall_start = time.perf_counter()
        for _ in range(iterations):
            # Retain captured output for the graph's lifetime. Ordinary outputs
            # are released after each call, as in the existing microbenchmark.
            call()
        dispatch_end = time.perf_counter()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter()
        samples.append((start.elapsed_time(end) * 1000 / iterations,
                        (dispatch_end - wall_start) * 1e6 / iterations,
                        (wall_end - wall_start) * 1e6 / iterations))
    spans, dispatch, wall = zip(*samples)
    return dict(case=case.name, mode=mode, iterations=iterations, repeats=repeats,
                gpu_span_us=statistics.median(spans),
                gpu_span_min_us=min(spans), gpu_span_max_us=max(spans),
                host_dispatch_us=statistics.median(dispatch),
                synchronized_wall_us=statistics.median(wall))


@torch.inference_mode()
def export_trace(cases, output_dir, *, steps, cuda=True):
    """Trace normal launches only, separately from uninstrumented timings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(steps):
            for case in cases:
                with torch.profiler.record_function(case.name):
                    output = case.fn()
                    if cuda:
                        # Isolate each label for inspection, not latency timing.
                        torch.cuda.synchronize()
        del output
    prof.export_chrome_trace(str(output_dir / 'trace.json'))
    sort_key = 'self_device_time_total' if cuda else 'self_cpu_time_total'
    table = prof.key_averages().table(sort_by=sort_key, row_limit=100)
    (output_dir / 'operators.txt').write_text(table + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--trace-steps', type=int, default=3)
    parser.add_argument('--no-compile', action='store_true', help='omit compiled cases for a quick smoke run')
    parser.add_argument('--output-dir', type=Path, default=Path('gpu-profile'))
    args = parser.parse_args()
    if min(args.batch, args.context, args.features, args.warmup,
           args.iterations, args.repeats, args.trace_steps) < 1:
        parser.error('dimensions and iteration counts must be positive')
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error('a working NVIDIA GPU is required')
    if torch.cuda.get_device_capability()[0] < 8:
        parser.error('the current kernel requires NVIDIA SM80+')
    import triton

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    dtype = getattr(torch, args.dtype)
    z_tr = torch.rand(args.batch, args.context, args.features, device='cuda', dtype=dtype)
    v = torch.randn(args.context, args.context, device='cuda', dtype=dtype)
    z_tl = torch.rand_like(z_tr)
    bias = torch.rand(args.features, device='cuda', dtype=dtype)
    metadata = dict(python=platform.python_version(), torch=torch.__version__,
                    triton=triton.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                    shape=list(z_tr.shape), dtype=args.dtype, seed=42,
                    automatic_cudagraphs=False, compiled_cases=not args.no_compile,
                    warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
                    trace_steps=args.trace_steps, matmul_precision='highest')
    results = []
    report = dict(metadata=metadata, results=results,
                  notes=['GPU span includes launch gaps; it is not pure kernel execution time.',
                         'Host dispatch and GPU span overlap; do not subtract them.',
                         'Graph capture/replay uses static input buffers.',
                         'Mix-only uses precomputed normalization; component times are not additive.',
                         'Timing is uninstrumented; trace is a separate normal-launch run.'])
    report_path = args.output_dir / 'report.json'
    print(json.dumps(metadata), flush=True)
    with torch.inference_mode():
        cases = build_cases(z_tr, v, z_tl, bias, compile_functions=not args.no_compile)
        # Warm every case before any measurement. Capture, compilation, and
        # checking results must never enter the timed repetition loop.
        for case in cases:
            _check(case.fn(), case.expected)
        for case in cases:
            for mode in ('launch', 'cuda_graph'):
                result = measure_case(case, mode, warmup=args.warmup,
                                      iterations=args.iterations, repeats=args.repeats)
                results.append(result)
                # Preserve completed measurements if a later capture fails.
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                print(f'{case.name:28s} {mode:10s} '
                      f'GPU-span={result["gpu_span_us"]:9.3f} us  '
                      f'host-dispatch={result["host_dispatch_us"]:9.3f} us  '
                      f'wall={result["synchronized_wall_us"]:9.3f} us', flush=True)
        with (args.output_dir / 'timings.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        export_trace(cases, args.output_dir, steps=args.trace_steps)
    print(f'Results: {args.output_dir}/ (report.json, timings.csv, trace.json, operators.txt)')


if __name__ == '__main__':
    main()
