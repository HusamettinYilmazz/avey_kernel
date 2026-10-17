import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


class ProfilingTests(unittest.TestCase):
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
                            torch.zeros(7, device='cuda'))
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
