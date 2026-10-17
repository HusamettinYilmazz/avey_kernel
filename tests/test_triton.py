"""Run on CUDA, or on CPU with TRITON_INTERPRET=1 (no GPU timing)."""

import os
import unittest

import torch
import torch.nn.functional as F

from avey_kernels import avey, avey_b

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
CAN_RUN = INTERPRET or torch.cuda.is_available()


@unittest.skipUnless(CAN_RUN, 'requires CUDA or TRITON_INTERPRET=1')
class TiledKernelTests(unittest.TestCase):
    def test_low_precision_inputs_keep_fp32_normalization(self):
        from avey_kernels._triton import mix

        device = 'cpu' if INTERPRET else 'cuda'
        torch.manual_seed(9)
        for dtype in (torch.float16, torch.bfloat16):
            z_tr = torch.rand(2, 35, 37, device=device, dtype=dtype)
            z_tr_norm = F.normalize(z_tr.float(), dim=-1)
            v = torch.randn(35, 35, device=device, dtype=dtype)
            for normalized in (False, True):
                with self.subTest(dtype=dtype, normalized=normalized):
                    selected_v = None if normalized else v
                    actual = mix(z_tr, z_tr_norm, selected_v, None, None, None, 1e-6,
                                 not normalized, normalized, 'identity')
                    expected = (avey_b(z_tr, backend='torch') if normalized
                                else avey(z_tr, v, backend='torch'))
                    # Triton 3.7's interpreter truncates FP32 -> BF16 in
                    # create_fp_trunc; CUDA rounds to nearest. Permit one ULP
                    # only for this interpreter case, not the GPU checks.
                    rtol = torch.finfo(dtype).eps if INTERPRET and dtype == torch.bfloat16 else 2e-3
                    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=rtol)

    def test_tiled_kernel_matches_reference(self):
        from avey_kernels._triton import mix

        device = 'cpu' if INTERPRET else 'cuda'
        torch.manual_seed(17)
        # Cross query, key, and feature tile boundaries; also test small tails.
        for c, d_prime in ((1, 1), (5, 7), (35, 37), (67, 65)):
            z_tr = torch.randn(2, c, d_prime, device=device)
            z_tr[:, 0] = 0.
            z_tl = torch.randn_like(z_tr)
            bias = torch.randn(d_prime, device=device)
            valid = torch.ones(2, c, device=device, dtype=torch.bool)
            if c > 1:
                valid[0, -1] = False
                valid[1] = False
                z_tr[0, -1] = float('nan')
            z_tr_norm = F.normalize(z_tr, dim=-1)
            v = torch.randn(c, c, device=device)
            for normalized, causal, activation in (
                (False, True, 'identity'), (False, False, 'silu'),
                (True, False, 'relu'), (True, False, 'gelu'),
                (True, False, 'relu2'),
            ):
                with self.subTest(c=c, d_prime=d_prime, normalized=normalized, causal=causal,
                                  activation=activation):
                    selected_v = None if normalized else v
                    kwargs = dict(z_tl=z_tl, bias=bias, valid=valid,
                                  activation=activation, backend='torch')
                    # A large epsilon avoids cancellation amplifying rounding in
                    # random signed inputs; a separate fixture tests cancellation.
                    expected = (avey_b(z_tr, eps=0.25, **kwargs) if normalized
                                else avey(z_tr, v, causal=causal, **kwargs))
                    actual = mix(z_tr, z_tr_norm, selected_v, z_tl, bias, valid, 0.25,
                                 causal, normalized, activation)
                    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

    def test_signed_zero_sum_denominator(self):
        from avey_kernels._triton import mix

        device = 'cpu' if INTERPRET else 'cuda'
        z_tr = torch.tensor([[[1., 0.], [-2., 0.]]], device=device)
        out = mix(z_tr, F.normalize(z_tr, dim=-1), None, None, None, None,
                  0.5, False, True, 'identity')
        torch.testing.assert_close(out, torch.tensor([[[6., 0.], [-6., 0.]]], device=device))


@unittest.skipUnless(torch.cuda.is_available() and not INTERPRET, 'requires real CUDA')
class CudaIntegrationTests(unittest.TestCase):
    def test_low_precision_and_fullgraph_compile(self):
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            # Positive inputs give a well-conditioned Avey-B denominator.
            z_tr = torch.rand(2, 35, 37, device='cuda', dtype=dtype)
            selected_v = torch.randn(35, 35, device='cuda', dtype=dtype)
            z_tl = torch.rand_like(z_tr)
            bias = torch.randn(37, device='cuda', dtype=dtype)
            for fn in (lambda x: avey(x, selected_v, z_tl=z_tl, bias=bias, backend='triton'),
                       lambda x: avey_b(x, z_tl=z_tl, bias=bias, backend='triton')):
                eager = fn(z_tr)
                compiled = torch.compile(fn, fullgraph=True)(z_tr)
                torch.testing.assert_close(compiled, eager, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(
                avey_b(z_tr, backend='triton'), avey_b(z_tr, backend='torch'),
                atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(
                avey(z_tr, selected_v, backend='triton'), avey(z_tr, selected_v, backend='torch'),
                atol=2e-3, rtol=2e-3)

    def test_noncontiguous_inputs_and_gradient_fallback(self):
        z_tr = torch.rand(2, 37, 35, device='cuda').transpose(1, 2)
        torch.testing.assert_close(avey_b(z_tr), avey_b(z_tr, backend='torch'))
        z_tr.requires_grad_()
        avey_b(z_tr).sum().backward()
        self.assertIsNotNone(z_tr.grad)
        with self.assertRaises(ValueError):
            avey_b(z_tr, backend='triton')


if __name__ == '__main__':
    unittest.main()
