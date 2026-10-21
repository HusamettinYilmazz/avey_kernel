"""The selection investigation replays exact configurations on shared inputs."""
import unittest
from unittest.mock import patch
import torch


class SelectionInvestigationTests(unittest.TestCase):
    def test_fixed_candidates_and_autotuned_path_share_normalized_input(self):
        from benchmarks import investigate_selection as investigate
        from benchmarks.compare_ordering import torch_dynamic_mix
        x = torch.rand(1, 3, 5)
        captures = []
        def call(x, q, v, gate, bias, *options):
            captures.append((q, v, options))
            return torch_dynamic_mix(x, q, gate, bias)
        with patch.object(investigate, 'mix_fixed_order', side_effect=call), \
             patch('avey_kernels._triton.mix', side_effect=call):
            cases = investigate.build_investigation_cases(x, x, x[0, 0], compile_functions=False)
            self.assertEqual([c.name.split('/')[2] for c in cases],
                             ['bm32_g1', 'bm16_g8', 'autotuned'])
            for case in cases:
                torch.testing.assert_close(case.fn(), case.expected)
        self.assertEqual(len({q.data_ptr() for q, _, _ in captures}), 1)
        self.assertTrue(all(v is None for _, v, _ in captures))
        self.assertEqual(captures[0][2], (1, True, 32, 128, 4, 1))
        self.assertEqual(captures[1][2], (8, True, 16, 128, 4, 1))
        self.assertEqual(captures[2][2], (None, 1e-6, False, True, 'identity', 128, True))
