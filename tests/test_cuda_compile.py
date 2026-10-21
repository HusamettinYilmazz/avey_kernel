"""Optional offline CUDA code generation: no GPU or driver required."""

import os
import unittest


@unittest.skipUnless(os.environ.get('AVEY_TEST_CUDA_COMPILE') == '1',
                     'set AVEY_TEST_CUDA_COMPILE=1 for offline CUDA compilation')
class OfflineCudaCompilationTests(unittest.TestCase):
    def test_byte_backed_validity_mask_compiles_for_access_analysis(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
        from avey_kernels._triton import _mix_kernel

        # Inductor's access-analysis trace represents torch.bool storage as
        # *i8. Loaded mask values must become predicates before use in tl.load.
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                signature = {name: '*fp32' for name in
                             ('Z_TR', 'Z_TR_NORM', 'V', 'Z_TL', 'Bias', 'Out')}
                signature['Valid'] = '*i8'
                constants = dict(C=35, D_PRIME=37, EPS=1e-6, HAS_V=weighted,
                                 HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=True,
                                 CAUSAL=weighted, NORMALIZE=not weighted, ACT=3,
                                 BM=16, BN=32, BK=32, BD=32, BATCH=2, DEVICE=0, GROUP_M=1)
                kernel = triton.compile(ASTSource(_mix_kernel, signature, constants),
                                        target=GPUTarget('cuda', 86, 32),
                                        options={'num_warps': 4})
                self.assertTrue(kernel.asm['cubin'])

    def test_autotune_candidates_generate_sm86_cubin(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
        from avey_kernels._triton import _mix_kernel, _AUTOTUNE_CONFIGS
        for weighted in (False, True):
            for config in _AUTOTUNE_CONFIGS:
                with self.subTest(weighted=weighted, config=str(config)):
                    signature = {name: '*bf16' for name in ('Z_TR', 'V', 'Z_TL', 'Bias', 'Out')}
                    signature.update(Z_TR_NORM='*fp32', Valid='*i1')
                    constants = dict(C=35, D_PRIME=133, EPS=1e-6, HAS_V=weighted,
                                     HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=True,
                                     CAUSAL=weighted, NORMALIZE=not weighted, ACT=4,
                                     BATCH=2, DEVICE=0, **config.kwargs)
                    kernel = triton.compile(ASTSource(_mix_kernel, signature, constants),
                                            target=GPUTarget('cuda', 86, 32),
                                            options={'num_warps': config.num_warps,
                                                     'num_stages': config.num_stages})
                    self.assertTrue(kernel.asm['cubin'])
                    self.assertLessEqual(kernel.metadata.shared, 101376)

    def test_output_tile_variants_generate_sm86_cubin(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
        from avey_kernels._triton import _mix_kernel

        for output_tile in (32, 64, 128):
            for weighted in (False, True):
                for dtype in ('fp32', 'fp16', 'bf16'):
                    with self.subTest(output_tile=output_tile, weighted=weighted, dtype=dtype):
                        signature = {name: '*' + dtype for name in ('Z_TR', 'V', 'Z_TL', 'Bias', 'Out')}
                        signature.update(Z_TR_NORM='*fp32', Valid='*i1')
                        constants = dict(C=128, D_PRIME=128, EPS=1e-6, HAS_V=weighted,
                                         HAS_Z_TL=True, HAS_BIAS=True, HAS_VALID=False,
                                         CAUSAL=weighted, NORMALIZE=not weighted,
                                         ACT=0, BM=16, BN=32, BK=32, BD=output_tile,
                                         BATCH=1, DEVICE=-1, GROUP_M=1)
                        kernel = triton.compile(ASTSource(_mix_kernel, signature, constants),
                                                target=GPUTarget('cuda', 86, 32),
                                                options={'num_warps': 4})
                        self.assertTrue(kernel.asm['cubin'])
                        # RTX 3090's opt-in per-block shared-memory limit.
                        self.assertLessEqual(kernel.metadata.shared, 101376)

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
                                     ACT=4, BM=16, BN=32, BK=32, BD=32,
                                     BATCH=1, DEVICE=-1, GROUP_M=1)
                    kernel = triton.compile(ASTSource(_mix_kernel, signature, constants),
                                            target=GPUTarget('cuda', 80, 32),
                                            options={'num_warps': 4})
                    self.assertTrue(kernel.asm['cubin'])


if __name__ == '__main__':
    unittest.main()
