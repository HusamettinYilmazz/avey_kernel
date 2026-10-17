"""Exercise custom-op fake execution without requiring a CUDA driver."""

import os
import unittest

import torch


@unittest.skipIf(os.environ.get('TRITON_INTERPRET') == '1',
                 'fake execution uses JIT wrapping, not interpreter execution')
class SymbolicTileTests(unittest.TestCase):
    def test_dynamo_accepts_changing_symbolic_tile_widths(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from avey_kernels._triton import mix

        def fn(z_tr, output_tile):
            return mix(z_tr, z_tr, None, None, None, None,
                       1e-6, False, True, 'identity', output_tile)

        # Exercise Dynamo and custom-op fake execution, without pretending to
        # test GPU lowering or executing a CUDA kernel on this CPU-only host.
        compiled = torch.compile(fn, backend='eager', fullgraph=True, dynamic=True)
        with FakeTensorMode():
            z_tr = torch.empty(1, 5, 7)
            for width in (32, 64, 128, 32):
                with self.subTest(width=width):
                    result = compiled(z_tr, width)
                    self.assertEqual(result.shape, z_tr.shape)
                    self.assertEqual(result.dtype, z_tr.dtype)

    def test_backed_symbolic_tile_specializes_during_fake_execution(self):
        from torch._dynamo.source import LocalSource
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
        from avey_kernels._triton import mix

        for width in (32, 64, 128):
            with self.subTest(width=width):
                shape_env = ShapeEnv()
                source = LocalSource('output_tile')
                symbol = shape_env.create_symbol(width, source, dynamic_dim=DimDynamic.DYNAMIC)
                output_tile = shape_env.create_symintnode(symbol, hint=width, source=source)
                self.assertIsInstance(output_tile, torch.SymInt)
                with FakeTensorMode(shape_env=shape_env):
                    z_tr = torch.empty(1, 5, 7)
                    result = mix(z_tr, z_tr, None, None, None, None,
                                 1e-6, False, True, 'identity', output_tile)
                    self.assertEqual(result.shape, z_tr.shape)
                    self.assertEqual(result.dtype, z_tr.dtype)


if __name__ == '__main__':
    unittest.main()
