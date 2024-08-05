import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from routing import RouterWeights, RouterMask

from typing import Optional, Callable, Union, List


def _combine_residual_tokens(
    input_tokens: torch.Tensor, output_tokens: torch.Tensor, router_mask: RouterMask
) -> torch.Tensor:
    """
    Combine input and output tokens based on the router mask.

    Args:
        input_tokens: Input tokens to the router.
        output_tokens: Output tokens from the router.
        router_mask: Router mask.

    Returns:
        Combined output tokens based on the router mask.
    """
    assert (
        input_tokens.shape == output_tokens.shape
    ), "Input and output tokens must have the same shape"
    num_groups, tokens_per_group, *hidden_dims = input_tokens.shape

    # Get inverse of the dispatch mask to get the residual mask
    residual_mask = router_mask.dispatch_mask.any(dim=(2, 3)).logical_not()
    residual_mask = residual_mask.view(1, 1, *hidden_dims)

    # Combine input and output tokens based on the residual mask
    return torch.where(residual_mask, input_tokens, output_tokens)


class MoeLayer(nn.Module):
    """
    Mixture of Experts Layer.

    Args:
        layer_index: Layer index.
        experts: Expert modules.
        num_experts: Number of experts.
        max_group_size: Maximum group size.
        train_capacity_factor: Train capacity factor.
        eval_capacity_factor: Evaluation capacity factor.
        router: Router module.
        min_expert_capacity: Minimum expert capacity.
        num_expert_replicas: Optionally inferred by the number of experts.
        dropout_rate: Dropout rate.
        dtype: Data type.
    """

    def __init__(
        self,
        layer_index: int,
        experts: Union[nn.Module, List[nn.Module]],
        num_experts: Optional[int] = None,
        max_group_size: int = 4096,
        train_capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        router: nn.Module = None,
        min_expert_capacity: int = 4,
        num_expert_replicas: Optional[int] = None,
        dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.layer_index = layer_index
        if isinstance(experts, nn.Module):
            assert (
                num_experts is not None
            ), "num_experts must be specified when providing a single expert module"
            self.experts = nn.ModuleList([experts for _ in range(num_experts)])
            self.num_experts = num_experts
        else:
            self.experts = nn.ModuleList(experts)
            self.num_experts = len(self.experts)
            if num_experts is not None and num_experts != self.num_experts:
                raise ValueError(
                    f"num_experts ({num_experts}) does not match the number of provided experts ({self.num_experts})"
                )

        self.max_group_size = max_group_size
        self.train_capacity_factor = train_capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.router = router if router is not None else Router(self.num_experts)
        self.min_expert_capacity = min_expert_capacity
        self.num_expert_replicas = (
            num_expert_replicas
            if num_expert_replicas is not None
            else self._num_expert_replicas(self.num_experts)
        )
        self.dropout_rate = dropout_rate
        self.dtype = dtype

    def _maybe_pad(
        inputs: torch.Tensor, num_experts: int, num_expert_replicas: int
    ) -> torch.Tensor:
        """Pads input array if number of tokens < number of experts.

        This function pads the input sequence to ensure that the number of tokens is
        divisble by the total number of experts.

        Args:
            inputs: Batch of input embeddings of shape <float>[batch_size, seq_length,
            hidden_dims].
            num_experts: Number of unique experts.
            num_expert_replicas: Number of copies of each expert.

        Returns:
            Input embeddings of shape <float>[batch_size + min_batch_padding, seq_length
            + min_seq_padding, hidden_dims]. Only one of min_batch_padding or
            min_seq_padding can be nonzero; both will be zero if no padding is required.
        """
        batch_size, seq_length, *_ = inputs.shape
        num_tokens = batch_size * seq_length
        total_num_experts = num_expert_replicas * num_experts

        if num_tokens % total_num_experts != 0:
            # Let's see how much padding is required if we pad the batch dimension.
            min_batch_padding = 1
            num_padding_tokens = seq_length
            while (num_tokens + num_padding_tokens) % total_num_experts != 0:
                # This loop will always yield
                # num_padding_tokens <= abs(total_num_experts * seq_length - num_tokens)
                # or, equivalently,
                # min_batch_padding <= abs(total_num_experts - batch_size).
                min_batch_padding += 1
                num_padding_tokens += seq_length

            # Alternatively, we could pad along the sequence dimension.
            min_seq_padding = 1
            num_padding_tokens = batch_size
            while (num_tokens + num_padding_tokens) % total_num_experts != 0:
                # This loop will always yield
                # num_padding_tokens <= abs(total_num_experts * batch_size - num_tokens)
                # or, equivalently,
                # min_seq_padding <= abs(total_num_experts - seq_length).
                min_seq_padding += 1
                num_padding_tokens += batch_size

            # Use the dimension which requires the least padding.
            if min_seq_padding * batch_size > min_batch_padding * seq_length:
                min_seq_padding = 0
            else:
                min_batch_padding = 0

            # TODO: Rather than relying on one of the dimensions, we
            #  should select the minimal amount of padding along a mixture of both of
            #  the sequence and batch dimensions.

            result = F.pad(
                inputs,
                ((0, min_batch_padding), (0, min_seq_padding), (0, 0)),
                "constant",
                constant_values=0,
            )

            logging.warning(
                (
                    "Efficiency warning: Batch size / sequence length temporarily"
                    " padded by %d tokens in MoE layer to ensure that the total number"
                    " of tokens is divisible by the total number of experts (%d). For"
                    " improved efficiency, consider increasing the number of tokens (by"
                    " increasing the batch size or beam size), and/or decreasing the"
                    " number of expert copies (by increasing the expert parallelism or"
                    " decreasing the number of experts)."
                ),
                min_batch_padding * seq_length + min_seq_padding * batch_size,
                total_num_experts,
            )

            return result
        else:
            return inputs

    def _num_expert_replicas(self, num_experts: int, num_model_partitions: int = 1):
        """
        Infer the number of expert replicas.

        Args:
            num_experts: Number of unique experts.
            num_model_partitions: Number of model partitions.
        Returns:
            Number of replicas per expert
        """
        return max(1, torch.cuda.device_count() // (num_experts * num_model_partitions))
