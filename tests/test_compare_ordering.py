"""Fixed ordering comparison must preserve launch settings and kernel math."""
import inspect
import os
import unittest
import torch


class OrderingComparisonTests(unittest.TestCase):
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
