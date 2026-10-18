import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


class ProfilingTests(unittest.TestCase):
    def test_tile_selection_rejects_empty_and_unsupported_widths(self):
        from benchmarks.profile_contextualizer import build_cases

        z_tr = torch.ones(1, 2, 3)
        for output_tiles in ((), (16,), (32, 256), (32, 32)):
            with self.subTest(output_tiles=output_tiles), self.assertRaises(ValueError):
                build_cases(z_tr, torch.ones(2, 2), z_tr, torch.zeros(3),
                            include_triton=False, compile_functions=False,
                            output_tiles=output_tiles)

    def test_profile_labels_keep_output_tiles_separate(self):
        from benchmarks.profile_contextualizer import build_cases

        z_tr = torch.ones(1, 2, 3)
        cases = build_cases(z_tr, torch.ones(2, 2), z_tr, torch.zeros(3),
                            compile_functions=False, output_tiles=(32, 64, 128))
        for architecture in ('avey', 'avey_b'):
            for tile in (32, 64, 128):
                for operation in ('fused', 'mix_only'):
                    name = f'{architecture}/{operation}/bd{tile}/eager'
                    matches = [case for case in cases if case.name == name]
                    self.assertEqual(len(matches), 1)
                    self.assertEqual(matches[0].output_tile, tile)

    def test_autotuned_cases_are_distinct_from_fixed_baseline(self):
        from benchmarks.profile_contextualizer import build_cases
        x = torch.ones(1, 2, 3)
        cases = build_cases(x, torch.ones(2, 2), x, torch.zeros(3),
                            compile_functions=False, output_tiles=(128,), autotune=True)
        names = {case.name for case in cases}
        for arch in ('avey', 'avey_b'):
            for component in ('fused', 'mix_only'):
                self.assertIn(f'{arch}/{component}/autotuned/eager', names)
                self.assertIn(f'{arch}/{component}/bd128/eager', names)

    def test_component_cases_follow_the_same_math_as_full_calls(self):
        from benchmarks.profile_contextualizer import build_cases

        z_tr = torch.tensor([[[1., 0.], [3., 0.]]])
        v = torch.tensor([[2., 1.], [4., 3.]])
        z_tl = torch.tensor([[[2., 3.], [4., 5.]]])
        bias = torch.tensor([-3., 1.])
        cases = build_cases(z_tr, v, z_tl, bias, include_triton=False, compile_functions=False)
        expected = {
            'normalization/eager': torch.tensor([[[1., 0.], [1., 0.]]]),
            'avey/reference_eager': torch.tensor([[[-2., 3.], [40., 5.]]]),
            'avey_b/reference_eager': torch.tensor([[[-2.000002, 3.], [-4.000004, 5.]]]),
        }
        self.assertEqual({case.name for case in cases}, set(expected))
        for case in cases:
            torch.testing.assert_close(case.fn(), expected[case.name])

    def test_trace_contains_each_named_component(self):
        from benchmarks.profile_contextualizer import build_cases, export_trace

        z_tr = torch.ones(1, 2, 3)
        cases = build_cases(z_tr, torch.ones(2, 2), z_tr, torch.zeros(3),
                            include_triton=False, compile_functions=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            export_trace(cases, path, steps=1, cuda=False)
            trace = json.loads((path / 'trace.json').read_text())
            names = {event.get('name') for event in trace['traceEvents']}
            for case in cases:
                self.assertIn(case.name, names)
            self.assertTrue((path / 'operators.txt').read_text())


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None
                     and os.environ.get('TRITON_INTERPRET') != '1'
                     and torch.cuda.get_device_capability()[0] >= 8,
                     'requires a real NVIDIA SM80+ GPU')
class CudaProfilingTests(unittest.TestCase):
    def test_timing_and_graph_replay_preserve_outputs(self):
        from benchmarks.profile_contextualizer import build_cases, measure_case

        z_tr = torch.rand(1, 5, 7, device='cuda')
        cases = build_cases(z_tr, torch.randn(5, 5, device='cuda'), z_tr,
                            torch.zeros(7, device='cuda'), output_tiles=(32, 64, 128))
        with torch.inference_mode():
            for case in cases:
                for mode in ('launch', 'cuda_graph'):
                    with self.subTest(case=case.name, mode=mode):
                        result = measure_case(case, mode, warmup=2, iterations=3, repeats=2)
                        self.assertEqual(result['mode'], mode)
                        self.assertGreaterEqual(result['gpu_span_us'], 0.)
                        self.assertGreaterEqual(result['host_dispatch_us'], 0.)


if __name__ == '__main__':
    unittest.main()
