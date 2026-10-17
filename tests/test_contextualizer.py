import unittest

import torch


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from avey_kernels import avey, avey_b
        cls.avey = staticmethod(avey)
        cls.avey_b = staticmethod(avey_b)

    def test_avey_weight_orientation_and_causality(self):
        # Parallel first/third rows exercise off-diagonal V entries and causality.
        z_tr = torch.tensor([[[1., 0.], [0., 2.], [3., 0.]]])
        v = torch.tensor([[2., 7., 11.], [13., 3., 17.], [5., 19., 4.]])
        actual = self.avey(z_tr, v, backend='torch')
        torch.testing.assert_close(actual, torch.tensor([[[2., 0.], [0., 6.], [17., 0.]]]))
        unmasked = self.avey(z_tr, v, causal=False, backend='torch')
        torch.testing.assert_close(unmasked[0, 0], torch.tensor([35., 0.]))

    def test_avey_b_normalizes_before_bias_activation_and_gate(self):
        z_tr = torch.tensor([[[1., 0.], [3., 0.]]])
        z_tl = torch.tensor([[[2., 3.], [4., 5.]]])
        bias = torch.tensor([-3., 1.])
        actual = self.avey_b(z_tr, z_tl=z_tl, bias=bias, activation='relu',
                             eps=2., backend='torch')
        torch.testing.assert_close(actual, torch.tensor([[[0., 3.], [0., 5.]]]))

    def test_zero_rows_and_padding_do_not_leak_bias(self):
        z_tr = torch.tensor([[[0., 0.], [2., 0.], [100., 100.]]])
        valid = torch.tensor([[True, True, False]])
        out = self.avey_b(z_tr, valid=valid, bias=torch.ones(2), backend='torch')
        torch.testing.assert_close(out, torch.tensor([[[1., 1.], [3., 1.], [0., 0.]]]))

    def test_signed_similarities_use_sum_not_softmax_or_abs(self):
        z_tr = torch.tensor([[[1., 0.], [-2., 0.]]], dtype=torch.float64)
        out = self.avey_b(z_tr, eps=0.5, backend='torch')
        torch.testing.assert_close(out, torch.tensor([[[6., 0.], [-6., 0.]]], dtype=z_tr.dtype))

    def test_torch_reference_supports_gradients(self):
        z_tr = torch.rand(1, 3, 2, dtype=torch.float64, requires_grad=True)
        v = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, v_arg: self.avey(x, v_arg, backend='torch'), (z_tr, v)))
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: self.avey_b(x, backend='torch'), (z_tr,)))

    def test_nan_padding_cannot_contaminate_valid_gradients(self):
        for normalized in (False, True):
            z_tr = torch.tensor([[[1., 2.], [3., 4.], [float('nan'), float('nan')]]],
                             requires_grad=True)
            z_tl = torch.tensor([[[1., 1.], [1., 1.], [float('nan'), float('nan')]]],
                                requires_grad=True)
            valid = torch.tensor([[True, True, False]])
            out = (self.avey_b(z_tr, z_tl=z_tl, valid=valid, backend='torch') if normalized
                   else self.avey(z_tr, torch.ones(3, 3), z_tl=z_tl, valid=valid, backend='torch'))
            out.sum().backward()
            self.assertTrue(torch.isfinite(z_tr.grad).all().item())
            self.assertTrue(torch.isfinite(z_tl.grad).all().item())
            torch.testing.assert_close(z_tr.grad[:, -1], torch.zeros(1, 2))
            torch.testing.assert_close(z_tl.grad[:, -1], torch.zeros(1, 2))

    def test_input_contract(self):
        z_tr = torch.ones(1, 3, 2)
        with self.assertRaises(ValueError):
            self.avey(z_tr, torch.ones(2, 2))
        with self.assertRaises(ValueError):
            self.avey(z_tr, None)
        with self.assertRaises(ValueError):
            self.avey_b(z_tr, eps=0.)
        with self.assertRaises(ValueError):
            self.avey_b(z_tr, activation='softmax')
        with self.assertRaises(ValueError):
            self.avey_b(z_tr, backend='unknown')
        with self.assertRaises(ValueError):
            self.avey_b(z_tr, valid=torch.ones(1, 3))
        for output_tile in (16, 256, 32.0):
            with self.subTest(output_tile=output_tile), self.assertRaises(ValueError):
                self.avey_b(z_tr, output_tile=output_tile)

    def test_auto_backend_preserves_autograd(self):
        z_tr = torch.ones(1, 3, 2, requires_grad=True)
        self.avey_b(z_tr).sum().backward()
        self.assertIsNotNone(z_tr.grad)


if __name__ == '__main__':
    unittest.main()
