"""Optional offline CUDA code generation: no GPU or driver required."""

import os
import unittest


@unittest.skipUnless(os.environ.get('AVEY_TEST_CUDA_COMPILE') == '1',
                     'set AVEY_TEST_CUDA_COMPILE=1 for offline CUDA compilation')
class OfflineCudaCompilationTests(unittest.TestCase):
    def test_sm80_specializations_generate_cubin(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
        from avey_kernels._triton import _mix_kernel

        for weighted in (False, True):
            for dtype in ('fp32', 'fp16', 'bf16'):
                with self.subTest(weighted=weighted, dtype=dtype):
                    signature = {name: '*' + dtype for name in ('Z_TR', 'V', 'Z_TL', 'Bias', 'Out')}
                    signature.update(Z_TR_NORM='*fp32', Valid='*i1')
                    constants = dict(C=35, D_PRIME=37, EPS=1e-6, HAS_V=weighted,
                                     HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=True,
                                     CAUSAL=weighted, NORMALIZE=not weighted,
                                     ACT=4, BM=16, BN=32, BK=32, BD=32)
                    kernel = triton.compile(ASTSource(_mix_kernel, signature, constants),
                                            target=GPUTarget('cuda', 80, 32),
                                            options={'num_warps': 4})
                    self.assertTrue(kernel.asm['cubin'])


if __name__ == '__main__':
    unittest.main()
