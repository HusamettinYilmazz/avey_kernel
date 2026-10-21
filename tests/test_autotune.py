"""Autotuned dispatch: fake tracing locally, correctness/capture on SM80+."""
import os
import unittest
import torch
from avey_kernels import avey, avey_b


class AutotuneTests(unittest.TestCase):
    def test_one_stage_candidates_extend_existing_search_without_duplicates(self):
        from itertools import product
        from avey_kernels._triton import _AUTOTUNE_CONFIGS
        def key(config):
            return (config.kwargs['BM'], config.kwargs['BN'], config.kwargs['BD'],
                    config.kwargs['GROUP_M'], config.num_warps, config.num_stages)
        actual = {key(c) for c in _AUTOTUNE_CONFIGS}
        old = {(bm, bn, bd, group, 4, 3)
               for bm, bn, bd, group in product((16, 32), (32, 64), (32, 64, 128), (1, 4, 8))}
        old |= {(bm, bn, 128, group, 8, 2)
                for bm, bn, group in product((16, 32), (32, 64), (1, 4, 8))}
        added = {(bm, bn, bd, group, 4, 1)
                 for bm, bn, bd, group in product((16, 32), (32, 64), (32, 64, 128), (1, 4, 8))}
        self.assertEqual(actual, old | added)
        self.assertEqual(len(_AUTOTUNE_CONFIGS), len(actual))

    def test_reference_fallback_preserves_autograd(self):
        x = torch.rand(1, 5, 7, requires_grad=True)
        result = avey_b(x, autotune=True)
        torch.testing.assert_close(result, avey_b(x, backend='torch'))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipIf(os.environ.get('TRITON_INTERPRET') == '1', 'requires JIT wrapper')
    def test_autotuned_op_traces_without_cuda(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from avey_kernels._triton import mix
        def fn(x):
            return mix(x, x, None, None, None, None, 1e-6,
                       False, True, 'identity', 32, True)
        compiled = torch.compile(fn, backend='eager', fullgraph=True)
        with FakeTensorMode():
            x = torch.empty(2, 35, 37)
            self.assertEqual(compiled(x).shape, x.shape)

    @unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None
                         and torch.cuda.get_device_capability()[0] >= 8
                         and os.environ.get('TRITON_INTERPRET') != '1', 'requires NVIDIA SM80+')
    def test_autotuning_matches_reference_and_supports_compiled_graph_replay(self):
        from benchmarks.profile_contextualizer import ProfileCase, measure_case
        torch.manual_seed(42)
        with torch.inference_mode():
            x = torch.rand(2, 35, 37, device='cuda')
            v = torch.randn(35, 35, device='cuda')
            valid = torch.ones(2, 35, dtype=torch.bool, device='cuda')
            valid[1, -3:] = False
            for op, extra in ((avey, (v,)), (avey_b, ())):
                def fn(op=op, extra=extra):
                    return op(x, *extra, valid=valid, activation='silu',
                              z_tl=x, bias=x[0, 0], backend='triton', autotune=True)
                expected = op(x, *extra, valid=valid, activation='silu',
                              z_tl=x, bias=x[0, 0], backend='torch')
                torch.testing.assert_close(fn(), expected, atol=1e-5, rtol=1e-5)
                compiled = torch.compile(fn, fullgraph=True,
                                         options={'triton.cudagraphs': False})
                # Do not silently accept access-analysis failures and the
                # resulting conservative input-cloning fallback.
                with self.assertNoLogs('torch._higher_order_ops.triton_kernel_wrap',
                                       level='WARNING'):
                    torch.testing.assert_close(compiled(), expected, atol=1e-5, rtol=1e-5)
                for call in (fn, compiled):
                    measure_case(ProfileCase('autotuned', call, expected), 'cuda_graph',
                                 warmup=1, iterations=2, repeats=1)
