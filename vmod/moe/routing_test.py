import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, List
from dataclasses import dataclass, replace

# Assume the Router classes and related functions are implemented in a file named 'routing.py'
from routing import TokensChooseMaskedRouter, ExpertsChooseMaskedRouter, RouterWeights, _load_balancing_loss

class RoutingTest(unittest.TestCase):
    def test_load_balancing_loss(self):
        num_tokens = 5
        num_experts = 2
        num_selected_experts = 1
        torch.manual_seed(0)
        router_probs = torch.rand(num_tokens, num_experts)
        expert_indices = torch.randint(0, 2, (num_tokens, num_selected_experts))
        self.assertAlmostEqual(
            _load_balancing_loss(router_probs, expert_indices).item(), 0.931183934211731, places=5)

    def test_tokens_choose_one_expert_mask_router(self):
        num_groups = 2
        tokens_per_group = 3
        hidden_dim = 4
        num_experts = 2
        num_selected_experts = 1  # Switch routing case
        expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
        torch.manual_seed(0)

        token_inputs = torch.rand(num_groups, tokens_per_group, hidden_dim)
        router = TokensChooseMaskedRouter(
            router_weights=RouterWeights(hidden_dim, num_experts),
            num_selected_experts=num_selected_experts,
            jitter_noise=0.,
            batch_prioritized_routing=True,
            ignore_padding_tokens=False,
            dtype=torch.float32)
        
        router_mask = router(token_inputs, expert_capacity)

        expected_mask = torch.tensor([
            [
                [[True], [False]],
                [[True], [False]],
                [[True], [False]],
            ],
            [
                [[True], [False]],
                [[True], [False]],
                [[True], [False]],
            ],
        ], dtype=torch.bool)

        self.assertTrue(torch.allclose(router_mask.dispatch_mask, expected_mask))

        expected_weights = torch.tensor([
            [
                [[0.5087], [0.]],
                [[0.5064], [0.]],
                [[0.5076], [0.]],
            ],
            [
                [[0.5014], [0.]],
                [[0.5095], [0.]],
                [[0.5098], [0.]],
            ],
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-4))

        self.assertAlmostEqual(router_mask.auxiliary_loss.item(), 1.0145, places=4)
        self.assertAlmostEqual(router_mask.router_z_loss.item(), 0.4829, places=4)

    def test_experts_choose_mask_router(self):
        num_groups = 2
        tokens_per_group = 4
        hidden_dim = 3
        num_experts = 2
        expert_capacity = 2

        torch.manual_seed(0)
        token_inputs = torch.rand(num_groups, tokens_per_group, hidden_dim)

        router_weights = RouterWeights(hidden_dim, num_experts)
        router = ExpertsChooseMaskedRouter(
            router_weights=router_weights,
            jitter_noise=0.,
            dtype=torch.float32,
            ignore_padding_tokens=False)
        
        router_mask = router(token_inputs, num_experts, expert_capacity)

        expected_mask = torch.tensor([
            [
                [[1, 0], [0, 0]],
                [[0, 0], [1, 0]],
                [[0, 0], [0, 1]],
                [[0, 1], [0, 0]],
            ],
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0, 0], [1, 0]],
                [[0, 1], [0, 0]],
            ],
        ], dtype=torch.int32)

        self.assertTrue(torch.allclose(router_mask.dispatch_mask, expected_mask))

        expected_weights = torch.tensor([
            [
                [[0.5013, 0.], [0., 0.]],
                [[0., 0.], [0.5068, 0.]],
                [[0., 0.], [0., 0.5030]],
                [[0., 0.4983], [0., 0.]],
            ],
            [
                [[0.4967, 0.], [0., 0.]],
                [[0., 0.], [0., 0.5069]],
                [[0., 0.], [0.5072, 0.]],
                [[0., 0.4941], [0., 0.]],
            ],
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-4))

        # Auxiliary loss is always 0. for experts choose tokens routing.
        self.assertEqual(router_mask.auxiliary_loss, 0.)
        self.assertAlmostEqual(router_mask.router_z_loss.item(), 0.4806286, places=6)

    def test_routers_ignore_padding(self):
        num_groups = 2
        tokens_per_group = 6
        hidden_dim = 2
        num_experts = 2
        num_selected_experts = 2
        expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
        torch.manual_seed(0)

        token_inputs = torch.rand(num_groups, tokens_per_group, hidden_dim)
        # Simulate masked inputs.
        padding_mask = torch.randint(0, 2, (num_groups, tokens_per_group, 1))
        token_inputs *= padding_mask

        router_weights = RouterWeights(hidden_dim, num_experts)

        with self.subTest(name='tokens_choose_masked_router'):
            router = TokensChooseMaskedRouter(
                router_weights=router_weights,
                num_selected_experts=num_selected_experts,
                jitter_noise=0.,
                batch_prioritized_routing=True,
                ignore_padding_tokens=True,
                dtype=torch.float32)
            
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
            ], dtype=torch.bool)

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
            ], dtype=torch.float32)
            self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-6))

            self.assertAlmostEqual(router_mask.auxiliary_loss, 0.6951497, places=5)
            self.assertAlmostEqual(router_mask.router_z_loss, 0.48541257, places=5)

        with self.subTest(name='experts_choose_masked_router'):
            router = ExpertsChooseMaskedRouter(
                router_weights=router_weights,
                jitter_noise=0.,
                ignore_padding_tokens=True,
                dtype=torch.float32)
            
            router_mask = router(token_inputs, num_experts, expert_capacity)

            expected_mask = torch.tensor([
                [
                    [[0], [0]],
                    [[1], [1]],
                    [[0], [0]],
                    [[0], [0]],
                    [[0], [0]],
                    [[0], [0]],
                ],
                [
                    [[0], [0]],
                    [[1], [0]],
                    [[0], [1]],
                    [[0], [0]],
                    [[0], [0]],
                    [[0], [0]],
                ],
            ], dtype=torch.bool)

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
                    [[0.50390625], [0.]],
                    [[0.], [0.49804688]],
                    [[0.], [0.]],
                    [[0.], [0.]],
                    [[0.], [0.]],
                ],
            ], dtype=torch.float32)
            self.assertTrue(torch.allclose(router_mask.combine_array, expected_weights, atol=1e-6))

            self.assertEqual(router_mask.auxiliary_loss, 0.)
            self.assertAlmostEqual(router_mask.router_z_loss, 0.48541257, places=5)

if __name__ == '__main__':
    unittest.main()