import unittest
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

# Assume these are implemented in your codebase
from moe_layers import MoeLayer, RouterMask
from routing import TokensChooseMaskedRouter, RouterWeights, Mlp

class TestMoeLayer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_layer_variables(self, module: nn.Module, init_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {k: v.to(self.device) for k, v in module.state_dict().items()}

    def test_moe_layer_runs(self):
        batch_size = 3
        max_seq_length = 4
        num_tokens = batch_size * max_seq_length
        hidden_dim = 2
        num_experts = 4

        torch.manual_seed(0)

        router = TokensChooseMaskedRouter(
            router_weights=RouterWeights(hidden_dim, num_experts),
            jitter_noise=0.0,
            num_selected_experts=2,
            batch_prioritized_routing=True,
            ignore_padding_tokens=True,
            dtype=torch.float32,
        )

        expert = Mlp(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            activation_fn=nn.GELU(),
            dropout_rate=0.1,
        )

        moe_layer = MoeLayer(
            num_experts=num_experts,
            max_group_size=num_tokens,
            train_capacity_factor=1.5,
            eval_capacity_factor=1.5,
            expert=expert,
            router=router,
            num_expert_partitions=num_experts,
            num_model_partitions=1,
            dtype=torch.float32,
            split_params=False,
        ).to(self.device)

        init_batch = {
            'inputs': torch.ones((batch_size, max_seq_length, hidden_dim), dtype=torch.float32)
        }
        params = self.init_layer_variables(moe_layer, init_batch)

        expected_keys = {'router', 'expert'}
        self.assertEqual(set(params.keys()), expected_keys)

        inputs = torch.rand((batch_size, max_seq_length, hidden_dim), device=self.device) * 20 - 10
        actual_outputs = moe_layer(inputs)

        self.assertEqual(actual_outputs.shape, (batch_size, max_seq_length, hidden_dim))

        # Check if metrics are stored
        self.assertTrue(hasattr(moe_layer, 'last_metrics'))
        for metric in ['auxiliary_loss', 'router_z_loss', 'fraction_tokens_left_behind', 'expert_usage', 'router_confidence']:
            self.assertIn(metric, moe_layer.last_metrics)

    def test_dense_general_expert(self):
        batch_size = 3
        max_seq_length = 8
        num_tokens = batch_size * max_seq_length
        hidden_dims = (2, 3)  # 2D hidden_dims
        num_experts = 4
        output_features = (3, 12)

        torch.manual_seed(0)

        expert = DenseGeneral(
            input_dims=hidden_dims,
            output_dims=output_features,
            use_bias=False,
        )

        router = TokensChooseMaskedRouter(
            router_weights=RouterWeights(hidden_dims, num_experts),
            num_selected_experts=1,
            dtype=torch.float32,
            jitter_noise=0.0,
            batch_prioritized_routing=False,
            ignore_padding_tokens=False,
        )

        moe_layer = MoeLayer(
            num_experts=num_experts,
            max_group_size=num_tokens,
            train_capacity_factor=1.0,
            eval_capacity_factor=1.0,
            expert=expert,
            router=router,
            num_expert_partitions=num_experts,
            num_model_partitions=1,
            dtype=torch.float32,
            split_params=False,
        ).to(self.device)

        init_batch = {
            'inputs': torch.ones((batch_size, max_seq_length, *hidden_dims), dtype=torch.float32)
        }
        params = self.init_layer_variables(moe_layer, init_batch)

        expected_keys = {'router', 'expert'}
        self.assertEqual(set(params.keys()), expected_keys)

        inputs = torch.rand((batch_size, max_seq_length, *hidden_dims), device=self.device) * 20 - 10
        actual_outputs = moe_layer(inputs)

        self.assertEqual(actual_outputs.shape, (batch_size, max_seq_length, *output_features))

    def test_scatter_mask_dispatch_equal(self):
        batch_size = 4
        max_seq_length = 4
        hidden_dim = 2
        num_experts = 2
        tokens_per_group = 8
        num_groups = batch_size * max_seq_length // tokens_per_group

        torch.manual_seed(0)

        expert = MlpBlock(
            hidden_dim,
            intermediate_dim=2,
            output_dim=hidden_dim,
            activation_fn=nn.GELU(),
            dropout_rate=0.0,
        )

        def create_moe_layer(router):
            return MoeLayer(
                num_experts=num_experts,
                max_group_size=tokens_per_group,
                train_capacity_factor=1.0,
                eval_capacity_factor=1.0,
                expert=expert,
                router=router,
                num_expert_partitions=num_experts,
                num_model_partitions=1,
                dtype=torch.float32,
                split_params=False,
            ).to(self.device)

        router_weights = RouterWeights(hidden_dim, num_experts)
        masked_router = TokensChooseMaskedRouter(
            router_weights=router_weights,
            jitter_noise=0.0,
            num_selected_experts=2,
            batch_prioritized_routing=True,
            dtype=torch.float32,
            ignore_padding_tokens=False,
        )
        masked_moe_layer = create_moe_layer(masked_router)

        scatter_router = TokensChooseScatterRouter(
            router_weights=router_weights,
            jitter_noise=0.0,
            num_selected_experts=2,
            batch_prioritized_routing=True,
            dtype=torch.float32,
            ignore_padding_tokens=False,
        )
        scatter_moe_layer = create_moe_layer(scatter_router)

        inputs = torch.rand((batch_size, max_seq_length, hidden_dim), device=self.device) * 20 - 10

        # Mock the router weights to ensure both layers compute with the same logits
        mock_router_logits = torch.rand((num_groups, tokens_per_group, num_experts), device=self.device) * 2 - 1

        with unittest.mock.patch.object(masked_router.router_weights, 'forward', return_value=mock_router_logits):
            masked_outputs = masked_moe_layer(inputs)

        with unittest.mock.patch.object(scatter_router.router_weights, 'forward', return_value=mock_router_logits):
            scatter_outputs = scatter_moe_layer(inputs)

        self.assertTrue(torch.allclose(masked_outputs, scatter_outputs, rtol=1e-5, atol=1e-5))

    def test_num_groups(self):
        test_cases = [
            (8, 32, 2, 1, 4),
            (9, 32, 2, 1, 4),
            (16, 32, 4, 2, 8),
            (32, 32, 2, 1, 2),
            (64, 32, 2, 1, 2),
        ]

        for max_group_size, num_tokens, num_experts, num_expert_replicas, expected_num_groups in test_cases:
            with self.subTest(f"max_group_size={max_group_size}, num_tokens={num_tokens}, num_experts={num_experts}, num_expert_replicas={num_expert_replicas}"):
                self.assertEqual(
                    MoeLayer._num_groups(num_tokens, max_group_size, num_experts, num_expert_replicas, strict_group_size=False),
                    expected_num_groups
                )

    def test_strict_group_size(self):
        with self.assertRaises(ValueError):
            MoeLayer._num_groups(num_tokens=16, max_group_size=16, num_experts=2, num_expert_replicas=1, strict_group_size=True)

    def test_num_expert_replicas(self):
        test_cases = [
            (1, 1, 4),
            (2, 1, 2),
            (2, 2, 1),
            (4, 1, 1),
        ]

        for num_expert_partitions, num_model_partitions, expected_num_replicas in test_cases:
            with self.subTest(f"num_expert_partitions={num_expert_partitions}, num_model_partitions={num_model_partitions}"):
                with unittest.mock.patch('torch.cuda.device_count', return_value=4):
                    self.assertEqual(
                        MoeLayer._num_expert_replicas(num_expert_partitions, num_model_partitions),
                        expected_num_replicas
                    )

    def test_maybe_pad(self):
        test_cases = [
            (2, 1, 0, 0),
            (16, 2, 1, 0),
            (32, 1, 1, 0),
            (7, 1, 0, 6),
            (9, 1, 0, 1),
        ]

        for num_experts, num_expert_replicas, expected_batch_padding, expected_seq_padding in test_cases:
            with self.subTest(f"num_experts={num_experts}, num_expert_replicas={num_expert_replicas}"):
                original_seq_length = 8
                original_batch_size = 3
                inputs = torch.ones((original_batch_size, original_seq_length, 2), dtype=torch.float32)
                padded_inputs = MoeLayer._maybe_pad(inputs, num_experts, num_expert_replicas)
                self.assertEqual(
                    padded_inputs.shape,
                    (original_batch_size + expected_batch_padding, original_seq_length + expected_seq_padding, 2)
                )
                if expected_batch_padding > 0 or expected_seq_padding > 0:
                    self.assertAlmostEqual(
                        torch.sum(torch.abs(padded_inputs[original_batch_size:, original_seq_length:])).item(),
                        0.0
                    )

if __name__ == '__main__':
    unittest.main()