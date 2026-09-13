import unittest
from gradient_diagnostic import gradient_statistics, query_row_hook

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'torch unavailable')
class GradientTests(unittest.TestCase):
    def test_query_rows_match_dense_embedding_partial_gradient(self):
        dense = torch.nn.Embedding(9,4)
        ids = torch.tensor([2,6])
        tokens = torch.tensor([[1,2,6,2,3]])
        weights = torch.arange(20.).reshape(1,5,4)
        baseline = dense(tokens)
        (baseline*weights).sum().backward()
        expected = dense.weight.grad[ids].clone()
        dense.weight.requires_grad_(False)
        leaf = dense.weight[ids].detach().clone().requires_grad_(True)
        handle = dense.register_forward_hook(query_row_hook(ids,leaf))
        try:
            actual = dense(tokens)
            self.assertTrue(torch.equal(actual,baseline))
            (actual*weights).sum().backward()
            self.assertTrue(torch.equal(leaf.grad,expected))
        finally:
            handle.remove()

    def test_zero_norm_is_undefined_and_conflict_is_negative(self):
        a = torch.tensor([1.,2.])
        self.assertAlmostEqual(gradient_statistics(a,-a)['cosine'],-1.)
        self.assertIsNone(gradient_statistics(a*0,a)['cosine'])
        with self.assertRaises(ValueError):
            gradient_statistics(a*float('nan'),a)
        with self.assertRaises(ValueError):
            gradient_statistics(a, a.reshape(1, 2))


if __name__ == '__main__':
    unittest.main()
