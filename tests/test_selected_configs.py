"""Capture the launch that actually ran, including cached Inductor winners."""
from types import SimpleNamespace as NS
import os
import torch
import unittest
from unittest.mock import patch


class SelectedConfigTests(unittest.TestCase):
    def test_records_post_dispatch_winner_and_ignores_unexecuted_kernels(self):
        from benchmarks.selected_configs import capture_selected_configs
        first = NS(config=NS(kwargs={'BD': 32}, num_warps=4, num_stages=3))
        winner = NS(config=NS(kwargs={'BD': 128}, num_warps=4, num_stages=1),
                    n_regs=255, n_spills=20, bin=NS(metadata=NS(shared=16384)))

        class FakeAutotuner:
            def __init__(self):
                self.fn = NS(__name__='_mix_kernel', arg_names=['X', 'BM', 'BD'])
                self.filename = '/tmp/generated.py'
                self.triton_meta = {'constants': {1: 32}}
                self.inductor_meta = {'kernel_name': 'mix_0'}
                self.launchers = [first, winner]
            def run(self, *args, **kwargs):
                self.launchers = [winner]
                return args[0]

        used, unused = FakeAutotuner(), FakeAutotuner()
        original = FakeAutotuner.run
        token = object()
        def fn():
            used.run(token, stream=0)
            return used.run(token, stream=0)  # Same cached winner, recorded once.
        with patch('torch._inductor.runtime.triton_heuristics.CachingAutotuner', FakeAutotuner):
            result, records = capture_selected_configs(fn)
        self.assertIs(result, token)
        self.assertIs(FakeAutotuner.run, original)
        self.assertEqual(len(unused.launchers), 2)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['status'], 'selected')
        self.assertEqual(record['parameters'], {'BM': 32, 'BD': 128})
        self.assertEqual((record['num_warps'], record['num_stages']), (4, 1))
        self.assertEqual((record['n_regs'], record['n_spills'], record['shared_bytes']),
                         (255, 20, 16384))

    def test_reads_resources_from_launcher_without_stored_binary(self):
        from benchmarks.selected_configs import capture_selected_configs
        class FakeAutotuner:
            fn = NS(__name__='_mix_kernel', arg_names=[])
            triton_meta = {'constants': {'BM': 32}}
            launchers = [NS(config=NS(kwargs={'BD': 128}, num_warps=4, num_stages=1),
                            n_regs=255, n_spills=0, shared=16384)]
            def run(self):
                return 1
        with patch('torch._inductor.runtime.triton_heuristics.CachingAutotuner', FakeAutotuner):
            _, records = capture_selected_configs(lambda: FakeAutotuner().run())
        self.assertEqual(records[0]['shared_bytes'], 16384)
        self.assertEqual(records[0]['n_spills'], 0)
        self.assertEqual(records[0]['parameters'], {'BM': 32, 'BD': 128})

    def test_reports_unresolved_launcher_selection(self):
        from benchmarks.selected_configs import capture_selected_configs
        class FakeAutotuner:
            fn = NS(__name__='unknown')
            launchers = []
            def run(self):
                return 7
        with patch('torch._inductor.runtime.triton_heuristics.CachingAutotuner', FakeAutotuner):
            result, records = capture_selected_configs(lambda: FakeAutotuner().run())
        self.assertEqual(result, 7)
        self.assertEqual(records[0]['status'], 'unresolved')
        self.assertNotIn('parameters', records[0])

    def test_records_candidate_scores_without_changing_selection(self):
        from benchmarks.selected_configs import capture_autotuning_choices
        class Launcher:
            def __init__(self, bm):
                self.config = NS(kwargs={'BM': bm, 'BN': 64, 'BD': 128, 'GROUP_M': 1},
                                 num_warps=4, num_stages=1)
                self.n_regs, self.n_spills, self.shared = 222, 0, 16384
        a, b = Launcher(32), Launcher(16)
        scores = {a: .1, b: .2}
        class FakeAutotuner:
            fn = NS(__name__='_mix_kernel')
            mutated_arg_names = ['Out']
            def benchmark_all_configs(self, *args, **kwargs):
                return scores
        original = FakeAutotuner.benchmark_all_configs
        with patch('torch._inductor.runtime.triton_heuristics.CachingAutotuner', FakeAutotuner):
            with capture_autotuning_choices() as events:
                returned = FakeAutotuner().benchmark_all_configs()
        self.assertIs(returned, scores)
        self.assertIs(FakeAutotuner.benchmark_all_configs, original)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['mutated_arg_names'], ['Out'])
        self.assertEqual([c['milliseconds'] for c in events[0]['candidates']], [.1, .2])
        self.assertEqual([c['parameters']['BM'] for c in events[0]['candidates']], [32, 16])

    def test_no_candidate_event_is_reported_when_no_tuning_occurs(self):
        from benchmarks.selected_configs import capture_autotuning_choices
        with capture_autotuning_choices() as events:
            pass
        self.assertEqual(events, [])

    def test_restores_observer_after_failure(self):
        from benchmarks.selected_configs import capture_selected_configs
        class FakeAutotuner:
            def run(self):
                raise ValueError('launch failed')
        original = FakeAutotuner.run
        with patch('torch._inductor.runtime.triton_heuristics.CachingAutotuner', FakeAutotuner):
            with self.assertRaisesRegex(ValueError, 'launch failed'):
                capture_selected_configs(lambda: FakeAutotuner().run())
        self.assertIs(FakeAutotuner.run, original)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None
                     and torch.cuda.get_device_capability()[0] >= 8
                     and os.environ.get('TRITON_INTERPRET') != '1', 'requires NVIDIA SM80+')
class CudaSelectedConfigTests(unittest.TestCase):
    def test_cached_compiled_launch_reports_actual_stage_and_tiles(self):
        from benchmarks.compare_ordering import build_stability_cases
        from benchmarks.selected_configs import capture_selected_configs
        with torch.inference_mode():
            x = torch.rand(1, 35, 37, device='cuda')
            v = torch.randn(35, 35, device='cuda')
            cases = build_stability_cases(x, v, x, x[0, 0])
            for case, stages in zip(cases[:2], (3, 1)):
                torch.testing.assert_close(case.fn(), case.expected, atol=1e-5, rtol=1e-5)
                output, records = capture_selected_configs(case.fn)
                torch.testing.assert_close(output, case.expected, atol=1e-5, rtol=1e-5)
                mixes = [r for r in records if '_mix_kernel' in r['kernel']]
                self.assertEqual(len(mixes), 1)
                record = mixes[0]
                self.assertEqual(record['status'], 'selected')
                self.assertEqual((record['num_warps'], record['num_stages']), (4, stages))
                self.assertEqual({k: record['parameters'][k] for k in ('BM', 'BN', 'BK', 'BD')},
                                 dict(BM=32, BN=64, BK=32, BD=128))
                self.assertIsNotNone(record['shared_bytes'])
