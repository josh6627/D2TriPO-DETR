"""Check that recomputing pairwise activations preserves training gradients."""
import copy
import unittest

import torch
from torch import nn

from models.dab_deformable_detr.dab_deformable_detr import DABDeformableDETR, MLP


class AdjacencyMemoryTests(unittest.TestCase):
    def make_head(self):
        # Exercise the real methods without constructing the backbone/CUDA ops.
        model = DABDeformableDETR.__new__(DABDeformableDETR)
        nn.Module.__init__(model)
        for name in ('adj1_linear', 'adj2_linear', 'adj_query', 'adj_key'):
            setattr(model, name, nn.ModuleList([nn.Linear(8, 8) for _ in range(2)]))
        model.WG = nn.ModuleList([nn.Linear(16, 1) for _ in range(2)])
        model.adj_embed = nn.ModuleList([MLP(16, 8, 1, 3) for _ in range(2)])
        return model

    def test_outputs_and_gradients_across_layers(self):
        torch.manual_seed(7)
        eager = self.make_head()
        recomputed = copy.deepcopy(eager)
        inputs = [torch.randn(2, 2, 7, 8, requires_grad=True) for _ in range(3)]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        expected = torch.stack([eager._adjacency_logits(*(x[i] for x in inputs), i)
                                for i in range(2)])
        actual = torch.stack([recomputed._predict_adjacency(*(x[i] for x in copies), i)
                              for i in range(2)])
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for original, copied in zip(inputs, copies):
            torch.testing.assert_close(copied.grad, original.grad)
        for original, copied in zip(eager.parameters(), recomputed.parameters()):
            torch.testing.assert_close(copied.grad, original.grad)

    def test_autocast_backward_outside_context(self):
        eager = self.make_head()
        recomputed = copy.deepcopy(eager)
        inputs = [torch.randn(1, 7, 8, requires_grad=True) for _ in range(3)]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = eager._adjacency_logits(*inputs, 0)
            actual = recomputed._predict_adjacency(*copies, 0)
        torch.testing.assert_close(actual, expected)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        for original, copied in zip(inputs, copies):
            torch.testing.assert_close(copied.grad, original.grad)
        for original, copied in zip(eager.parameters(), recomputed.parameters()):
            if original.grad is not None:
                torch.testing.assert_close(copied.grad, original.grad)

    def test_fewer_saved_activations_and_eval_matches(self):
        model = self.make_head()
        inputs = [torch.randn(1, 30, 8, requires_grad=True) for _ in range(3)]

        def saved_elements(fn):
            sizes = []
            def pack(tensor):
                sizes.append(tensor.numel())
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                output = fn(*inputs, 0)
            return sum(sizes), output

        eager_size, expected = saved_elements(model._adjacency_logits)
        checkpoint_size, actual = saved_elements(model._predict_adjacency)
        self.assertLess(checkpoint_size, eager_size / 10)
        torch.testing.assert_close(actual, expected)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model._predict_adjacency(*inputs, 0), expected)


if __name__ == '__main__':
    unittest.main()
