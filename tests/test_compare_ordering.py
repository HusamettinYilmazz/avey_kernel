"""Fixed ordering comparison must preserve launch settings and kernel math."""
import inspect
import os
import unittest
import torch


class OrderingComparisonTests(unittest.TestCase):
    def test_pytorch_baseline_consumes_supplied_normalized_input(self):
        from benchmarks.compare_ordering import torch_causal_mix
        # Deliberately use Q != N(x): recomputing normalization would give a
        # different answer and invalidate the isolated mixing comparison.
        x = torch.tensor([[[1., 0.], [3., 0.]]])
        q = torch.tensor([[[2., 0.], [4., 0.]]])
        v = torch.ones(2, 2)
        result = torch_causal_mix(x, q, v, torch.ones_like(x), torch.tensor([1., 2.]))
        torch.testing.assert_close(result, torch.tensor([[[5., 2.], [57., 2.]]]))

    def test_avey_b_baseline_uses_supplied_q_and_signed_row_sum(self):
        from benchmarks.compare_ordering import torch_dynamic_mix
        x = torch.tensor([[[1., 0.], [3., 0.]]])
        q = torch.tensor([[[2., 0.], [-4., 0.]]])
        gate = torch.tensor([[[2., 3.], [4., 5.]]])
        bias = torch.tensor([1., 2.])
        expected = torch.tensor([[[(-20. / (-4. + 1e-6) + 1.) * 2., 6.],
                                  [(40. / (8. + 1e-6) + 1.) * 4., 10.]]])
        torch.testing.assert_close(torch_dynamic_mix(x, q, gate, bias), expected)

    def test_avey_b_stability_uses_matching_reference_and_labels(self):
        from benchmarks.compare_ordering import build_stability_cases
        x = torch.rand(1, 3, 5)
        cases = build_stability_cases(x, None, x, x[0, 0], compile_functions=False)
        self.assertEqual([c.name for c in cases],
                         [f'avey_b/mix_only/{label}/eager'
                          for label in ('w4_s3', 'w4_s1', 'pytorch')])
        torch.testing.assert_close(cases[-1].fn(), cases[-1].expected)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1'
                         or (torch.cuda.is_available() and torch.version.hip is None
                             and torch.cuda.get_device_capability()[0] >= 8),
                         'requires interpreter or NVIDIA SM80+')
    def test_avey_b_stability_matches_reference_across_tails(self):
        from benchmarks.compare_ordering import build_stability_cases
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        torch.manual_seed(46)
        for c, d in ((35, 133), (67, 257)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = torch.rand(1, c, d, device=device, dtype=dtype)
                for case in build_stability_cases(x, None, x, x[0, 0], compile_functions=False):
                    with self.subTest(shape=x.shape, dtype=dtype, case=case.name):
                        torch.testing.assert_close(case.fn(), case.expected, atol=1e-5,
                                                   rtol=max(1e-5, torch.finfo(dtype).eps))

    def test_stability_balances_order_and_pairs_each_round(self):
        from unittest.mock import patch
        from benchmarks.compare_ordering import measure_stability
        from benchmarks.profile_contextualizer import ProfileCase
        labels = ('w4_s3', 'w4_s1', 'pytorch')
        cases = [ProfileCase(f'avey/mix_only/{label}/compiled', None, None) for label in labels]
        calls = []

        def timing(case, mode, **kwargs):
            round_number = len(calls) // 3 + 1
            calls.append(case.name.split('/')[2])
            base = {'w4_s3': 400., 'w4_s1': 300., 'pytorch': 200.}[calls[-1]]
            return dict(case=case.name, gpu_span_us=round_number * base,
                        gpu_span_min_us=round_number * base, gpu_span_max_us=round_number * base)

        with patch('benchmarks.compare_ordering.measure_case', side_effect=timing) as measured:
            rows = list(measure_stability(cases, warmup=2, iterations=3, repeats=4))
        self.assertEqual(len(rows), 18)
        orders = [tuple(calls[i:i+3]) for i in range(0, 18, 3)]
        self.assertEqual(len(set(orders)), 6)
        for label in labels:
            for position in range(3):
                self.assertEqual(sum(order[position] == label for order in orders), 2)
        for row in rows:
            label = row['case'].split('/')[2]
            self.assertEqual(row['speedup_vs_pytorch'],
                             {'w4_s3': .5, 'w4_s1': 2./3., 'pytorch': 1.}[label])
            self.assertEqual(row['round'], rows.index(row) // 3 + 1)
            self.assertEqual(row['position'], rows.index(row) % 3 + 1)
        for call in measured.call_args_list:
            self.assertEqual(call.args[1], 'cuda_graph')
            self.assertEqual(call.kwargs, dict(warmup=2, iterations=3, repeats=4))

    def test_stability_cases_select_only_the_two_fixed_launches_and_pytorch(self):
        from benchmarks.compare_ordering import build_stability_cases
        x = torch.rand(1, 3, 5)
        cases = build_stability_cases(x, torch.ones(3, 3), x, x[0, 0],
                                      compile_functions=False)
        self.assertEqual([case.name for case in cases],
                         [f'avey/mix_only/{label}/eager'
                          for label in ('w4_s3', 'w4_s1', 'pytorch')])

    def test_stability_summary_uses_median_of_paired_ratios(self):
        from benchmarks.compare_ordering import summarize_stability
        rows = [dict(case='candidate', gpu_span_us=time, speedup_vs_pytorch=ratio)
                for time, ratio in ((10., 10.), (20., 1.), (100., 2.))]
        summary, = summarize_stability(rows)
        self.assertEqual(summary['rounds'], 3)
        self.assertEqual(summary['median_gpu_span_us'], 20.)
        self.assertEqual(summary['median_paired_speedup_vs_pytorch'], 2.)
        self.assertEqual(summary['min_paired_speedup_vs_pytorch'], 1.)
        self.assertEqual(summary['max_paired_speedup_vs_pytorch'], 10.)

    @unittest.skipIf(os.environ.get('TRITON_INTERPRET') == '1', 'requires JIT wrapping')
    def test_launch_sweep_keeps_math_and_grid_fixed(self):
        from benchmarks.compare_ordering import mix_fixed_order, LAUNCH_CONFIGS
        from avey_kernels._triton import _mix_kernel
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        implementation = inspect.getclosurevars(mix_fixed_order._init_fn).nonlocals['fn']
        self.assertEqual(set(LAUNCH_CONFIGS), {(w, s) for w in (4, 8) for s in (1, 2, 3)})
        for warps, stages in LAUNCH_CONFIGS:
            def fn(x):
                return implementation(x, x, x[0, :, :67].contiguous(), x,
                                      x[0, 0], 1, True, 32, 128, warps, stages)
            graph = make_fx(fn, tracing_mode='fake')(torch.empty(2, 67, 257))
            launches = [n for n in graph.graph.nodes
                        if 'triton_kernel_wrapper_mutation' in str(n.target)]
            self.assertEqual(len(launches), 1)
            args = launches[0].kwargs['kwargs']
            self.assertEqual(set(args), set(_mix_kernel.arg_names))
            self.assertEqual({k: args[k] for k in
                              ('BM', 'BN', 'BK', 'BD', 'GROUP_M', 'SKIP_CAUSAL')},
                             dict(BM=32, BN=64, BK=32, BD=128, GROUP_M=1, SKIP_CAUSAL=True))
            kernel = kernel_side_table.get_kernel(launches[0].kwargs['kernel_idx'])
            self.assertEqual(len(kernel.configs), 1)
            self.assertEqual(kernel.configs[0].num_warps, warps)
            self.assertEqual(kernel.configs[0].num_stages, stages)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1'
                         or (torch.cuda.is_available() and torch.version.hip is None
                             and torch.cuda.get_device_capability()[0] >= 8),
                         'requires interpreter or NVIDIA SM80+')
    def test_launch_sweep_matches_reference(self):
        from benchmarks.compare_ordering import build_launch_cases
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        torch.manual_seed(45)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.rand(1, 35, 133, device=device, dtype=dtype)
            v = torch.randn(35, 35, device=device, dtype=dtype)
            cases = build_launch_cases(x, v, x, x[0, 0], compile_functions=False)
            self.assertEqual(len(cases), 7)
            self.assertEqual(len({case.name for case in cases}), 7)
            for case in cases:
                with self.subTest(dtype=dtype, case=case.name):
                    torch.testing.assert_close(case.fn(), case.expected, atol=1e-5,
                                               rtol=max(1e-5, torch.finfo(dtype).eps))

    @unittest.skipIf(os.environ.get('TRITON_INTERPRET') == '1', 'requires JIT wrapping')
    def test_tile_sweep_keeps_other_launch_settings_fixed(self):
        from benchmarks.compare_ordering import mix_fixed_order
        from avey_kernels._triton import _mix_kernel
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        implementation = inspect.getclosurevars(mix_fixed_order._init_fn).nonlocals['fn']
        for bm in (16, 32):
            for bd in (64, 128, 256):
                def fn(x):
                    return implementation(x, x, x[0, :, :67].contiguous(), x,
                                          x[0, 0], 1, True, bm, bd)
                graph = make_fx(fn, tracing_mode='fake')(torch.empty(2, 67, 257))
                launches = [n for n in graph.graph.nodes
                            if 'triton_kernel_wrapper_mutation' in str(n.target)]
                self.assertEqual(len(launches), 1)
                args = launches[0].kwargs['kwargs']
                self.assertEqual(set(args), set(_mix_kernel.arg_names))
                self.assertEqual({k: args[k] for k in
                                  ('BM', 'BN', 'BK', 'BD', 'GROUP_M', 'SKIP_CAUSAL')},
                                 dict(BM=bm, BN=64, BK=32, BD=bd, GROUP_M=1, SKIP_CAUSAL=True))
                kernel = kernel_side_table.get_kernel(launches[0].kwargs['kernel_idx'])
                self.assertEqual(len(kernel.configs), 1)
                self.assertEqual(kernel.configs[0].num_warps, 4)
                self.assertEqual(kernel.configs[0].num_stages, 3)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1'
                         or (torch.cuda.is_available() and torch.version.hip is None
                             and torch.cuda.get_device_capability()[0] >= 8),
                         'requires interpreter or NVIDIA SM80+')
    def test_tile_sweep_matches_reference_across_feature_tail(self):
        from benchmarks.compare_ordering import build_tile_cases
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        torch.manual_seed(44)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.rand(1, 67, 257, device=device, dtype=dtype)
            v = torch.randn(67, 67, device=device, dtype=dtype)
            cases = build_tile_cases(x, v, x, x[0, 0], compile_functions=False)
            self.assertEqual(len(cases), 7)
            self.assertEqual(len({case.name for case in cases}), 7)
            self.assertTrue(any('/pytorch/' in case.name for case in cases))
            for case in cases:
                with self.subTest(dtype=dtype, case=case.name):
                    torch.testing.assert_close(case.fn(), case.expected, atol=1e-5,
                                               rtol=max(1e-5, torch.finfo(dtype).eps))

    @unittest.skipIf(os.environ.get('TRITON_INTERPRET') == '1', 'requires JIT wrapping')
    def test_lowered_cases_change_only_group_size(self):
        from benchmarks.compare_ordering import mix_fixed_order
        from avey_kernels._triton import _mix_kernel
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        implementation = inspect.getclosurevars(mix_fixed_order._init_fn).nonlocals['fn']
        for weighted in (False, True):
            for group in (1, 4, 8):
                def fn(x):
                    return implementation(x, x.float(), x[0, :, :35] if weighted else None,
                                          x, x[0, 0], group)
                graph = make_fx(fn, tracing_mode='fake')(torch.empty(2, 35, 133))
                launches = [n for n in graph.graph.nodes
                            if 'triton_kernel_wrapper_mutation' in str(n.target)]
                self.assertEqual(len(launches), 1)
                args = launches[0].kwargs['kwargs']
                self.assertEqual(set(args), set(_mix_kernel.arg_names))
                self.assertEqual({k: args[k] for k in ('BM', 'BN', 'BK', 'BD', 'GROUP_M')},
                                 dict(BM=32, BN=64, BK=32, BD=128, GROUP_M=group))
                self.assertEqual(args['CAUSAL'], weighted)
                self.assertEqual(args['NORMALIZE'], not weighted)
                # wrap_triton represents explicit launch options with a
                # singleton configuration, not the 48-candidate tuner.
                kernel = kernel_side_table.get_kernel(launches[0].kwargs['kernel_idx'])
                self.assertEqual(len(kernel.configs), 1)
                self.assertEqual(kernel.configs[0].num_warps, 4)
                self.assertEqual(kernel.configs[0].num_stages, 3)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1'
                         or (torch.cuda.is_available() and torch.version.hip is None
                             and torch.cuda.get_device_capability()[0] >= 8),
                         'requires interpreter or NVIDIA SM80+')
    def test_all_fixed_orderings_match_reference(self):
        from benchmarks.compare_ordering import build_ordering_cases
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.rand(2, 35, 133, device=device, dtype=dtype)
            v = torch.randn(35, 35, device=device, dtype=dtype)
            cases = build_ordering_cases(x, v, x, x[0, 0], compile_functions=False)
            self.assertEqual(len(cases), 6)
            self.assertEqual(len({case.name for case in cases}), 6)
            for case in cases:
                with self.subTest(case=case.name, dtype=dtype):
                    torch.testing.assert_close(case.fn(), case.expected, atol=1e-5,
                                               rtol=max(1e-5, torch.finfo(dtype).eps))
