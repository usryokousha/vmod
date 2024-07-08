# MoE implementation based on the one in Flaxformer:
# https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/moe_layers.py
# A PyTorch implementation supporting Expert Choice and Token Choice routing does not seem to exist.
# This implementation attempts to provide similar functionality to the Flaxformer implementation
# while also taking into account PyTorch's distributed data parallelism features.

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

from vmod import routing

from typing import Optional, Tuple, Any


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class MoeLayer(nn.Module):
    def __init__(
        self,
        router: routing.Router,
        experts: nn.ModuleList,
        train_capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_expert_capacity: int = 4,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.router = router
        self.experts = experts
        self.train_capacity_factor = train_capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.experts = experts
        self.group = group if group is not None else dist.group.WORLD

        # Set expert flag for all parameters
        for expert in self.experts:
            for p in expert.parameters():
                p.expert = True
        self.num_groups = dist.get_world_size(group=self.group)
        self.num_local_experts = len(self.experts)

    def _call_experts(self, inputs: torch.Tensor) -> torch.Tensor:
        num_groups, num_experts, capacity, *hidden_dims = inputs.shape

        # All-to-all communication
        inputs = _AllToAll.apply(self.group, inputs)

        # Reshape for expert computation
        inputs = inputs.reshape(num_experts, -1, *hidden_dims)

        # Apply expert transformation
        chunks = torch.chunk(inputs, self.num_local_experts, dim=0)
        outputs = []
        for expert, chunk in zip(self.experts, chunks):

            if isinstance(self.experts, nn.Module):
                outputs = self.experts(chunk)
            else:
                raise ValueError(f"Unsupported expert class: {type(self.experts)}")
            outputs += [expert(chunk)]
        output = torch.cat(outputs, dim=0)

        # All-to-all communication
        output = _AllToAll.apply(self.group, output)

        # Reshape back to original shape
        outputs = outputs.reshape(num_groups, num_experts, capacity, -1)

        return outputs

    def _mask_and_dispatch_to_experts(
        self,
        token_inputs: torch.Tensor,
        enable_dropout: bool,
        expert_capacity: int,
    ) -> torch.Tensor:
        num_groups, tokens_per_group = token_inputs.shape[:2]

        router_mask: routing.RouterMask = self.router(
            token_inputs,
            self.num_local_experts,
            expert_capacity,
            apply_jitter=enable_dropout,
        )

        expert_inputs = torch.einsum(
            "gt...,gtec->gec...",
            token_inputs,
            router_mask.dispatch_mask,
        )

        expert_outputs = self._call_experts(expert_inputs, enable_dropout)

        combined_outputs = torch.einsum(
            "gec...,gtec->gt...",
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

        if isinstance(self.router, routing.ExpertsChooseMaskedRouter):
            expert_usage = 1.0
        else:
            total_expert_capacity = self.num_experts * expert_capacity * num_groups
            expert_usage = num_tokens_dispatched / total_expert_capacity

        metrics = {
            "auxiliary_loss": router_mask.auxiliary_loss,
            "router_z_loss": router_mask.router_z_loss,
            "fraction_tokens_left_behind": fraction_tokens_left_behind,
            "router_confidence": router_confidence,
            "expert_usage": expert_usage,
        }

        return combined_outputs, metrics

    def forward(
        self,
        inputs: torch.Tensor,
        enable_dropout: bool = True,
    ) -> torch.Tensor:
        original_batch_size, original_seq_length, *hidden_dims = inputs.shape

        padded_inputs = self._maybe_pad(inputs, self.num_local_experts, self.num_groups)
        padded_batch_size, padded_seq_length, *_ = padded_inputs.shape

        num_tokens = padded_batch_size * padded_seq_length
        tokens_per_group = num_tokens // self.num_groups

        capacity_factor = (
            self.train_capacity_factor if enable_dropout else self.eval_capacity_factor
        )
        expert_capacity = max(
            int(round(capacity_factor * tokens_per_group / self.num_local_experts)),
            self.min_expert_capacity,
        )

        grouped_inputs = padded_inputs.reshape(
            self.num_groups, tokens_per_group, *hidden_dims
        )

        if isinstance(
            self.router, routing.MaskedRouter
        ):  # Assuming this is a MaskedRouter
            outputs = self._mask_and_dispatch_to_experts(
                grouped_inputs,
                enable_dropout,
                expert_capacity,
            )
        else:
            raise ValueError(f"Unrecognized router type: {self.router}")

        result = outputs.reshape(
            padded_batch_size, padded_seq_length, *outputs.shape[2:]
        )
        if (
            padded_seq_length - original_seq_length > 0
            or padded_batch_size - original_batch_size > 0
        ):
            result = result[:original_batch_size, :original_seq_length]

        return result

    @staticmethod
    def _maybe_pad(inputs: torch.Tensor, num_groups: int) -> torch.Tensor:
        batch_size, seq_length, *_ = inputs.shape
        num_tokens = batch_size * seq_length

        if num_tokens % num_groups != 0:
            min_batch_padding = 1
            num_padding_tokens = seq_length
            while (num_tokens + num_padding_tokens) % num_groups != 0:
                min_batch_padding += 1
                num_padding_tokens += seq_length

            min_seq_padding = 1
            num_padding_tokens = batch_size
            while (num_tokens + num_padding_tokens) % num_groups != 0:
                min_seq_padding += 1
                num_padding_tokens += batch_size

            if min_seq_padding * batch_size > min_batch_padding * seq_length:
                min_seq_padding = 0
            else:
                min_batch_padding = 0

            result = F.pad(
                inputs,
                (0, 0, 0, min_seq_padding, 0, min_batch_padding),
                "constant",
                0,
            )

            return result
        else:
            return inputs
