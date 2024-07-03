import unittest
import torch
import torch.nn as nn
from typing import Tuple

from vmod import routing

class TestRouting(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_experts_choose_mask_router(self):
        num_groups = 2
        tokens_per_group = 4
        hidden_dim = 3
        num_experts = 2
        expert_capacity = 2

        torch.manual_seed(0)  # For reproducibility

        token_inputs = torch.rand(num_groups, tokens_per_group, hidden_dim, device=self.device)

        router = routing.ExpertsChooseMaskedRouter(
            router_hidden_dim=hidden_dim,
            num_experts=num_experts,
            jitter_noise=0.,
            ignore_padding_tokens=False,
            dtype=torch.float32
        ).to(self.device)

        router_mask = router(token_inputs, num_experts, expert_capacity)

        expected_mask = torch.tensor([
            [
                [[0, 1], [1, 0]],
                [[0, 0], [0, 1]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
                [[0, 0], [0, 1]],
            ],
        ], dtype=torch.int32, device=self.device)

        self.assertTrue(torch.allclose(router_mask.dispatch_mask, expected_mask))

        expected_weights = torch.tensor([
            [
                [[0., 0.49609375], [0.50390625, 0.]],
                [[0., 0.], [0., 0.50390625]],
                [[0.49804688, 0.], [0., 0.]],
                [[0., 0.], [0., 0.]],
            ],
            [
                [[0.49804688, 0.], [0., 0.]],
                [[0., 0.49414062], [0., 0.]],
                [[0., 0.], [0.5078125, 0.]],
                [[0., 0.], [0., 0.5078125]],
            ],
        ], dtype=torch.float32, device=self.device)

        self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-6))

        # Auxiliary loss is always 0 for experts choose tokens routing
        self.assertAlmostEqual(router_mask.auxiliary_loss, 0., places=6)
        self.assertAlmostEqual(router_mask.router_z_loss, 0.5041504, places=6)

    def test_tokens_choose_multiple_experts_mask_router(self):
        num_groups = 2
        tokens_per_group = 4
        hidden_dim = 3
        num_experts = 3
        num_selected_experts = 2
        expert_capacity = 1

        torch.manual_seed(0)  # For reproducibility

        token_inputs = torch.rand(num_groups, tokens_per_group, hidden_dim, device=self.device)

        router = routing.TokensChooseMaskedRouter(
            router_hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_selected_experts=num_selected_experts,
            jitter_noise=0.01,
            batch_prioritized_routing=True,
            ignore_padding_tokens=False,
            dtype=torch.float32
        ).to(self.device)

        router_mask = router(token_inputs, num_experts, expert_capacity)

        expected_mask = torch.tensor([
            [
                [[True], [False], [True]],
                [[False], [True], [False]],
                [[False], [False], [False]],
                [[False], [False], [False]],
            ],
            [
                [[True], [True], [False]],
                [[False], [False], [True]],
                [[False], [False], [False]],
                [[False], [False], [False]],
            ],
        ], dtype=torch.bool, device=self.device)

        self.assertTrue(torch.allclose(router_mask.dispatch_mask, expected_mask))

        expected_weights = torch.tensor([
            [
                [[0.33203125], [0.], [0.3359375]],
                [[0.], [0.3359375], [0.]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
            ],
            [
                [[0.33007812], [0.34179688], [0.]],
                [[0.], [0.], [0.3359375]],
                [[0.], [0.], [0.]],
                [[0.], [0.], [0.]],
            ],
        ], dtype=torch.float32, device=self.device)

        self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-6))

        self.assertAlmostEqual(router_mask.auxiliary_loss, 2.001709, places=6)
        self.assertAlmostEqual(router_mask.router_z_loss, 1.2714844, places=6)

    def test_routers_ignore_padding(self):
        num_groups = 2
        tokens_per_group = 6
        hidden_dim = 2
        num_experts = 2
        num_selected_experts = 2
        expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens

        torch.manual_seed(0)  # For reproducibility

        token_inputs = torch.rand(num_groups, tokens_per_group, hidden_dim, device=self.device)
        # Simulate masked inputs
        padding_mask = torch.randint(0, 2, (num_groups, tokens_per_group, 1), device=self.device)
        token_inputs *= padding_mask

        router = routing.TokensChooseMaskedRouter(
            router_hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_selected_experts=num_selected_experts,
            jitter_noise=0.,
            batch_prioritized_routing=True,
            ignore_padding_tokens=True,
            dtype=torch.float32
        ).to(self.device)

        router_mask = router(token_inputs, num_experts, expert_capacity)

        expected_mask = torch.tensor([
            [
                [[False], [False]],
                [[True], [True]],
                [[False], [False]],
                [[False], [False]],
                [[False], [False]],
                [[False], [False]],
            ],
            [
                [[False], [False]],
                [[True], [True]],
                [[False], [False]],
                [[False], [False]],
                [[False], [False]],
                [[False], [False]],
            ],
        ], dtype=torch.bool, device=self.device)

        self.assertTrue(torch.allclose(router_mask.dispatch_mask, expected_mask))

        expected_weights = torch.tensor([
            [
                [[0.], [0.]],
                [[0.50390625], [0.49804688]],
                [[0.0], [0.]],
                [[0.0], [0.]],
                [[0.0], [0.]],
                [[0.0], [0.]],
            ],
            [
                [[0.], [0.]],
                [[0.50390625], [0.49414062]],
                [[0.], [0.]],
                [[0.], [0.]],
                [[0.], [0.]],
                [[0.], [0.]],
            ],
        ], dtype=torch.float32, device=self.device)

        self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-6))

        self.assertAlmostEqual(router_mask.auxiliary_loss, 0.6951497, places=6)
        self.assertAlmostEqual(router_mask.router_z_loss, 0.48541257, places=6)

if __name__ == '__main__':
    unittest.main()