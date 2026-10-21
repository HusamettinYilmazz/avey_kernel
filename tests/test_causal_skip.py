"""Causal tile skipping must preserve the diagonal, tails, padding and gating."""
import inspect
import os
import unittest
import torch
import torch.nn.functional as F


CAN_RUN = os.environ.get('TRITON_INTERPRET') == '1' or (
    torch.cuda.is_available() and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8)
if CAN_RUN:
    import triton
    import triton.language as tl
    from avey_kernels._triton import _key_tile_count

    @triton.jit
    def _record_counts(Out, C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       CAUSAL: tl.constexpr, SKIP: tl.constexpr):
        tile = tl.program_id(0)
        count = _key_tile_count(tile, C, BM, BN, CAUSAL, SKIP)
        tl.store(Out + tile, count)


class CausalSkippingTests(unittest.TestCase):
    @unittest.skipIf(os.environ.get('TRITON_INTERPRET') == '1', 'requires JIT wrapping')
    def test_benchmark_changes_only_causal_skip_flag(self):
        from benchmarks.compare_ordering import mix_fixed_order
        from avey_kernels._triton import _mix_kernel
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        implementation = inspect.getclosurevars(mix_fixed_order._init_fn).nonlocals['fn']
        captured = []
        for skip in (False, True):
            def fn(x):
                return implementation(x, x, x[0, :, :67].contiguous(), x, x[0, 0], 1, skip)
            graph = make_fx(fn, tracing_mode='fake')(torch.empty(2, 67, 133))
            launches = [n for n in graph.graph.nodes
                        if 'triton_kernel_wrapper_mutation' in str(n.target)]
            self.assertEqual(len(launches), 1)
            args = dict(launches[0].kwargs['kwargs'])
            self.assertEqual(set(args), set(_mix_kernel.arg_names))
            self.assertEqual(args.pop('SKIP_CAUSAL'), skip)
            captured.append({k: v for k, v in args.items() if not isinstance(v, torch.fx.Node)})
            kernel = kernel_side_table.get_kernel(launches[0].kwargs['kernel_idx'])
            self.assertEqual(len(kernel.configs), 1)
        self.assertEqual(captured[0], captured[1])

    @unittest.skipUnless(CAN_RUN, 'requires interpreter or NVIDIA SM80+')
    def test_future_only_tiles_are_not_visited(self):
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        for c, bm, bn, expected in ((1, 32, 64, [1]), (35, 16, 64, [1, 1, 1]),
                                    (256, 32, 64, [1, 1, 2, 2, 3, 3, 4, 4]),
                                    (65, 64, 32, [2, 3]), (35, 16, 32, [1, 1, 2])):
            for causal, skip in ((True, True), (True, False), (False, True)):
                out = torch.empty(len(expected), dtype=torch.int32, device=device)
                _record_counts[(len(expected),)](out, c, bm, bn, causal, skip)
                counts = out.cpu().tolist()
                target = expected if causal and skip else [(c + bn - 1) // bn] * len(expected)
                self.assertEqual(counts, target)

    @unittest.skipUnless(CAN_RUN, 'requires interpreter or NVIDIA SM80+')
    def test_fixed_causal_comparison_matches_reference_in_all_dtypes(self):
        from benchmarks.compare_ordering import build_causal_cases
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.rand(1, 67, 133, device=device, dtype=dtype)
            v = torch.randn(67, 67, device=device, dtype=dtype)
            cases = build_causal_cases(x, v, x, x[0, 0], compile_functions=False)
            self.assertEqual(len(cases), 3)
            self.assertTrue(any('/pytorch/' in case.name for case in cases))
            for case in cases:
                torch.testing.assert_close(case.fn(), case.expected, atol=1e-5,
                                           rtol=max(1e-5, torch.finfo(dtype).eps))

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1'
                         or (torch.cuda.is_available() and torch.version.hip is None
                             and torch.cuda.get_device_capability()[0] >= 8),
                         'requires interpreter or NVIDIA SM80+')
    def test_skipped_and_full_key_loops_match_causal_reference(self):
        from avey_kernels import avey
        from avey_kernels._triton import _mix_kernel
        import triton
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        torch.manual_seed(43)
        for c, d in ((1, 7), (35, 37), (67, 133), (161, 37)):
            x = torch.randn(2, c, d, device=device)
            valid = torch.ones(2, c, dtype=torch.bool, device=device)
            valid[1] = False
            if c > 1:
                valid[0, -1] = False
            x[~valid] = float('nan')
            norm = F.normalize(x, dim=-1)
            v, gate, bias = torch.randn(c, c, device=device), torch.randn_like(x), torch.randn(d, device=device)
            expected = avey(x, v, valid=valid, z_tl=gate, bias=bias, activation='silu', backend='torch')
            for group in (1, 4, 8):
                for skip in (False, True):
                    with self.subTest(c=c, d=d, group=group, skip=skip):
                        out = torch.empty_like(x)
                        _mix_kernel[(triton.cdiv(c, 32) * triton.cdiv(d, 128), 2)](
                            x, norm, v, gate, bias, valid, out,
                            c, d, 1e-6, True, True, True, True, True, False, 3,
                            32, 64, 32, 128, BATCH=2, DEVICE=0,
                            GROUP_M=group, SKIP_CAUSAL=skip, num_warps=4, num_stages=3)
                        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
