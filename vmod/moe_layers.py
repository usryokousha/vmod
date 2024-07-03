import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from routing import ExpertsChooseMaskedRouter, MaskedRouter, RouterMask

class MoeLayer(nn.Module):
    def __init__(
        self,
        num_experts: int,
        max_group_size: int,
        train_capacity_factor: float,
        eval_capacity_factor: float,
        expert: Union[nn.Module, nn.Linear],
        router: MaskedRouter,
        num_expert_partitions: int,
        num_model_partitions: int,
        min_expert_capacity: int = 4,
        dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
        split_params: bool = True,
        strict_group_size: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.max_group_size = max_group_size
        self.train_capacity_factor = train_capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.expert = expert
        self.router = router
        self.num_expert_partitions = num_expert_partitions
        self.num_model_partitions = num_model_partitions
        self.min_expert_capacity = min_expert_capacity
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.split_params = split_params
        self.strict_group_size = strict_group_size

        if self.num_expert_partitions > self.num_experts:
            raise ValueError(
                f'The number of expert partitions ({self.num_expert_partitions}) '
                f'cannot be greater than the number of experts ({self.num_experts}).'
            )

        self.num_expert_replicas = self._num_expert_replicas(
            self.num_expert_partitions, self.num_model_partitions
        )

    def forward(
        self,
        inputs: torch.Tensor,
        enable_dropout: bool = True,
    ) -> torch.Tensor:
        original_batch_size, original_seq_length, *hidden_dims = inputs.shape

        padded_inputs = self._maybe_pad(inputs, self.num_experts, self.num_expert_replicas)
        padded_batch_size, padded_seq_length, *_ = padded_inputs.shape

        num_tokens = padded_batch_size * padded_seq_length

        num_groups = self._num_groups(
            num_tokens,
            self.max_group_size,
            self.num_experts,
            self.num_expert_replicas,
            self.strict_group_size,
        )
        tokens_per_group = num_tokens // num_groups

        capacity_factor = self.train_capacity_factor if enable_dropout else self.eval_capacity_factor
        expert_capacity = max(int(round(capacity_factor * tokens_per_group / self.num_experts)), self.min_expert_capacity)

        grouped_inputs = padded_inputs.reshape(num_groups, tokens_per_group, *hidden_dims)

        if isinstance(self.router, MaskedRouter):  # Assuming this is a MaskedRouter
            outputs = self._mask_and_dispatch_to_experts(
                grouped_inputs,
                enable_dropout,
                expert_capacity,
            )
        else:
            raise ValueError(f'Unrecognized router type: {self.router}')

        result = outputs.reshape(padded_batch_size, padded_seq_length, *outputs.shape[2:])
        if padded_seq_length - original_seq_length > 0 or padded_batch_size - original_batch_size > 0:
            result = result[:original_batch_size, :original_seq_length]

        return result

    def _mask_and_dispatch_to_experts(
        self,
        token_inputs: torch.Tensor,
        enable_dropout: bool,
        expert_capacity: int,
    ) -> torch.Tensor:
        num_groups, tokens_per_group = token_inputs.shape[:2]

        router_mask: RouterMask = self.router(
            token_inputs,
            self.num_experts,
            expert_capacity,
            apply_jitter=enable_dropout,
        )

        expert_inputs = torch.einsum(
            'gt...,gtec->gec...',
            token_inputs,
            router_mask.dispatch_mask,
        )

        expert_outputs = self._call_experts(expert_inputs, enable_dropout)

        combined_outputs = torch.einsum(
            'gec...,gtec->gt...',
            expert_outputs,
            router_mask.combine_array,
        )

        num_tokens_dispatched_somewhere = torch.max(
            router_mask.dispatch_mask, dim=(-1, -2)
        )[0].sum()

        if self.router.ignore_padding_tokens:
            num_tokens = torch.sum(torch.abs(token_inputs) > 0, dim=-1).float().sum()
        else:
            num_tokens = float(num_groups * tokens_per_group)

        fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / num_tokens

        num_tokens_dispatched = router_mask.dispatch_mask.sum()
        router_confidence = router_mask.combine_array.sum() / num_tokens_dispatched

        if isinstance(self.router, ExpertsChooseMaskedRouter):
            expert_usage = 1.0
        else:
            total_expert_capacity = self.num_experts * expert_capacity * num_groups
            expert_usage = num_tokens_dispatched / total_expert_capacity

        self._sow_expert_metrics(
            router_mask.auxiliary_loss,
            router_mask.router_z_loss,
            fraction_tokens_left_behind,
            router_confidence,
            expert_usage,
        )

        return combined_outputs

    def _call_experts(self, inputs: torch.Tensor, enable_dropout: bool) -> torch.Tensor:
        num_groups, num_experts, capacity, *hidden_dims = inputs.shape
        inputs = inputs.to(self.dtype)

        # Reshape for expert computation
        inputs = inputs.reshape(num_experts, -1, *hidden_dims)

        # Apply expert transformation
        if isinstance(self.expert, nn.Linear):
            outputs = self.expert(inputs)
        elif isinstance(self.expert, nn.Module):  # Assuming this is an MLP
            outputs = self.expert(inputs, enable_dropout=enable_dropout)
        else:
            raise ValueError(f'Unsupported expert class: {type(self.expert)}')

        # Reshape back to original shape
        outputs = outputs.reshape(num_groups, num_experts, capacity, -1)

        return outputs

    def _sow_expert_metrics(
        self,
        auxiliary_loss: float,
        router_z_loss: float,
        fraction_tokens_left_behind: float,
        router_confidence: float,
        expert_usage: float,
    ) -> None:
        # In PyTorch, we'll use a dictionary to store these metrics
        metrics = {
            'auxiliary_loss': auxiliary_loss,
            'router_z_loss': router_z_loss,
            'fraction_tokens_left_behind': fraction_tokens_left_behind,
            'router_confidence': router_confidence,
            'expert_usage': expert_usage,
        }
        self.last_metrics = metrics  # Store the metrics as an attribute

    @staticmethod
    def _num_groups(
        num_tokens: int,
        max_group_size: int,
        num_experts: int,
        num_expert_replicas: int = 1,
        strict_group_size: bool = False,
    ) -> int:
        min_num_groups = num_tokens // max_group_size
        min_num_groups = max(min_num_groups, num_expert_replicas * num_experts)

        def viable(n):
            return num_tokens % n == 0 and n % (num_expert_replicas * num_experts) == 0

        num_groups = min_num_groups
        while num_groups < num_tokens and not viable(num_groups):
            num_groups += 1

        if num_tokens % num_groups > 0:
            raise ValueError(
                'Group size and the number of experts must divide evenly into the '
                f'global number of tokens, but num_tokens={num_tokens}, while '
                f'num_groups={num_groups} for max_group_size={max_group_size} '
                f'and num_experts={num_experts}, each with {num_expert_replicas} '
                'replicas.'
            )

        group_size = num_tokens // num_groups

        if strict_group_size and group_size != max_group_size:
            raise ValueError(
                f'Selected group_size={group_size} is less than the '
                f'max_group_size={max_group_size}. Exiting because strict mode is '
                'active (strict_group_size=True)'
            )

        return num_groups

    @staticmethod
    def _num_expert_replicas(num_expert_partitions: int, num_model_partitions: int) -> int:
        raise NotImplementedError

    @staticmethod
    def _maybe_pad(inputs: torch.Tensor, num_experts: int, num_expert_replicas: int = 1) -> torch.Tensor:
        batch_size, seq_length, *_ = inputs.shape
        num_tokens = batch_size * seq_length
        total_num_experts = num_expert_replicas * num_experts

        if num_tokens % total_num_experts != 0:
            min_batch_padding = 1
            num_padding_tokens = seq_length
            while (num_tokens + num_padding_tokens) % total_num_experts != 0:
                min_batch_padding += 1
                num_padding_tokens += seq_length

            min_seq_padding = 1
            num_padding_tokens = batch_size
            while (num_tokens + num_padding_tokens) % total_num_experts != 0:
                min_seq_padding += 1
                num_padding_tokens += batch_size

            if min_seq_padding * batch_size > min_batch_padding * seq_length:
                min_seq_padding = 0
            else:
                min_batch_padding = 0

            result = F.pad(
                inputs,
                (0, 0, 0, min_seq_padding, 0, min_batch_padding),
                'constant',
                0,
            )

            return result
        else:
            return inputs