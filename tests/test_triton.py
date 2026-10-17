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
    def test_output_tile_variants_match_reference(self):
        from avey_kernels._triton import mix

        device = 'cpu' if INTERPRET else 'cuda'
        torch.manual_seed(29)
        # 133 features crosses even the widest tile and leaves a partial tail;
        # 35 tokens crosses both query and key tile boundaries.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            z_tr = torch.rand(2, 35, 133, device=device, dtype=dtype)
            z_tr_norm = F.normalize(z_tr.float(), dim=-1)
            v = torch.randn(35, 35, device=device, dtype=dtype)
            z_tl = torch.rand_like(z_tr)
            bias = torch.rand(133, device=device, dtype=dtype)
            valid = torch.ones(2, 35, device=device, dtype=torch.bool)
            valid[0, -1] = False
            valid[1] = False
            for normalize in (False, True):
                op, extra = (avey_b, ()) if normalize else (avey, (v,))
                expected = op(z_tr, *extra, z_tl=z_tl, bias=bias, valid=valid, backend='torch')
                for output_tile in (32, 64, 128):
                    with self.subTest(dtype=dtype, normalize=normalize, output_tile=output_tile):
                        actual = mix(z_tr, z_tr_norm, None if normalize else v,
                                     z_tl, bias, valid, 1e-6, not normalize,
                                     normalize, 'identity', output_tile=output_tile)
                        torch.testing.assert_close(actual, expected, atol=2e-3,
                                                   rtol=max(2e-3, torch.finfo(dtype).eps))

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
                    # Nearby FP32 sums can straddle a BF16 rounding midpoint
                    # and become adjacent output values, including on CUDA.
                    # Keep a separate tight check on pre-rounding arithmetic.
                    unrounded = mix(z_tr.float(), z_tr_norm,
                                    selected_v.float() if selected_v is not None else None,
                                    None, None, None, 1e-6,
                                    not normalized, normalized, 'identity')
                    oracle = (avey_b(z_tr.double(), backend='torch') if normalized
                              else avey(z_tr.double(), v.double(), backend='torch'))
                    torch.testing.assert_close(unrounded.double(), oracle, atol=1e-5, rtol=1e-5)
                    rtol = max(2e-3, torch.finfo(dtype).eps)
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
    def test_output_tile_variants_compile_through_public_api(self):
        torch.manual_seed(29)
        z_tr = torch.rand(2, 35, 133, device='cuda')
        v = torch.randn(35, 35, device='cuda')
        z_tl = torch.rand_like(z_tr)
        for op, extra in ((avey, (v,)), (avey_b, ())):
            expected = op(z_tr, *extra, z_tl=z_tl, backend='torch')
            for output_tile in (32, 64, 128):
                with self.subTest(operation=op.__name__, output_tile=output_tile):
                    def fn(x, op=op, extra=extra, output_tile=output_tile):
                        return op(x, *extra, z_tl=z_tl, output_tile=output_tile, backend='triton')

                    eager = fn(z_tr)
                    compiled = torch.compile(fn, fullgraph=True)(z_tr)
                    torch.testing.assert_close(eager, expected, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(compiled, expected, atol=1e-5, rtol=1e-5)

    def test_low_precision_and_fullgraph_compile(self):
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            # Positive inputs give a well-conditioned Avey-B denominator.
            z_tr = torch.rand(2, 35, 37, device='cuda', dtype=dtype)
            selected_v = torch.randn(35, 35, device='cuda', dtype=dtype)
            z_tl = torch.rand_like(z_tr)
            bias = torch.randn(37, device='cuda', dtype=dtype)
            rtol = max(2e-3, torch.finfo(dtype).eps)
            with self.subTest(dtype=dtype):
                for fn in (lambda x: avey(x, selected_v, z_tl=z_tl, bias=bias, backend='triton'),
                           lambda x: avey_b(x, z_tl=z_tl, bias=bias, backend='triton')):
                    eager = fn(z_tr)
                    compiled = torch.compile(fn, fullgraph=True)(z_tr)
                    torch.testing.assert_close(compiled, eager, atol=2e-3, rtol=rtol)
                for name, op, extra in (('Avey', avey, (selected_v,)), ('Avey-B', avey_b, ())):
                    with self.subTest(operation=name):
                        # Verify the underlying FP32 computation against FP64
                        # before allowing dtype-dependent output rounding.
                        unrounded = op(z_tr.float(), *(v.float() for v in extra), backend='triton')
                        oracle = op(z_tr.double(), *(v.double() for v in extra), backend='torch')
                        torch.testing.assert_close(unrounded.double(), oracle, atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(
                            op(z_tr, *extra, backend='triton'), op(z_tr, *extra, backend='torch'),
                            atol=2e-3, rtol=rtol)

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
