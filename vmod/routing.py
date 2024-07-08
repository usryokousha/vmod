import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from dataclasses import dataclass, replace


@dataclass
class RouterMask:
    """
    Dataclass to store the routing instructions for masked matmul dispatch routers.

    Attributes:
        dispatch_mask (torch.Tensor): Boolean tensor indicating which tokens are routed to which experts.
            Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        combine_array (torch.Tensor): Float tensor used for combining expert outputs and scaling with router probability.
            Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        auxiliary_loss (float): Load balancing loss for the router.
        router_z_loss (float): Router z-loss to encourage router logits to remain small for improved stability.
    """

    dispatch_mask: torch.Tensor
    combine_array: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.0

    def replace(self, **kwargs: Any) -> "RouterMask":
        """
        Create a new RouterMask instance with updated attributes.

        Args:
            **kwargs: New attribute values to replace in the RouterMask.

        Returns:
            RouterMask: New RouterMask instance with updated attributes.
        """
        return replace(self, **kwargs)


class RouterWeights(nn.Linear):
    """
    Module to compute router logits from token inputs.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        bias: bool = True,
        dtype=torch.float32,
        **kwargs
    ):
        super().__init__(dim, num_experts, bias=bias, dtype=dtype, **kwargs)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=2e-2)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Router(nn.Module):
    """
    Abstract base router class, defining router API and inner workings.
    """

    def __init__(
        self,
        router_weights: RouterWeights,
        jitter_noise: float,
        dtype: torch.dtype,
        ignore_padding_tokens: bool,
    ):
        super().__init__()
        self.router_weights = router_weights
        self.jitter_noise = jitter_noise
        self.dtype = dtype
        self.ignore_padding_tokens = ignore_padding_tokens

    def forward(
        self,
        token_inputs: torch.Tensor,
        num_experts: int,
        expert_capacity: int,
        apply_jitter: bool = True,
    ) -> RouterMask:
        token_inputs = token_inputs.to(torch.float32)

        if apply_jitter and self.jitter_noise > 0:
            token_inputs *= torch.empty_like(token_inputs).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        router_logits = self.router_weights(token_inputs)
        router_probabilities = F.softmax(router_logits, dim=-1)

        if self.ignore_padding_tokens:
            padding_mask = (torch.sum(torch.abs(token_inputs), dim=-1) > 0).unsqueeze(
                -1
            )
            router_logits *= padding_mask
        else:
            padding_mask = None

        routing_instructions = self._compute_routing_instructions(
            router_probabilities, padding_mask, expert_capacity
        )

        router_z_loss = torch.mean(torch.square(F.log_softmax(router_logits, dim=-1)))

        return routing_instructions.replace(router_z_loss=router_z_loss)

    def _compute_routing_instructions(
        self,
        router_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        expert_capacity: int,
    ) -> RouterMask:
        raise NotImplementedError(
            "Router is an abstract class that should be subclassed."
        )


class MaskedRouter(Router):
    """Abstract base router class for masked matmul dispatch routers.

    MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine
    array for sending and receiving (via masked matmuls) inputs and outputs to and
    from experts.
    """

    def _compute_routing_instructions(
        self,
        router_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        expert_capacity: int,
    ) -> RouterMask:
        """Computes masks for the top-k experts per token.

        Args:
          router_probs: <float32>[num_groups, tokens_per_group, num_experts]
            probabilities used to determine the routing of tokens to the experts.
          padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
            used to identify padding tokens that should be ignored by the router.
          expert_capacity: Each group will send this many tokens to each expert.

        Returns:
          Router mask arrays.
        """
        raise NotImplementedError(
            "MaskedRouter is an abstract class that should be subclassed."
        )


class ExpertsChooseMaskedRouter(Router):
    """
    Masked matmul router using experts choose tokens assignment.

    This router uses the same mechanism as in Mixture-of-Experts with Expert
    Choice (https://arxiv.org/abs/2202.09368): each expert selects its top
    expert_capacity tokens. An individual token may be processed by multiple
    experts or none at all.

    Note: "experts choose routing" should not be used in decoder blocks because it
    breaks the autoregressive behavior -- the model will learn to cheat by using
    future token information to improve current token predictions.
    """

    def __init__(
        self,
        router_weights: nn.Linear,
        jitter_noise: float,
        dtype: torch.dtype,
        ignore_padding_tokens: bool,
        auxiliary_loss: bool = False,
    ):
        super().__init__(router_weights, jitter_noise, dtype, ignore_padding_tokens)
        self.auxiliary_loss = auxiliary_loss

    def _compute_routing_instructions(
        self,
        router_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        expert_capacity: int,
    ) -> RouterMask:
        """
        Computes masks for the highest probability token per expert.

        Args:
            router_probs: [num_groups, tokens_per_group, num_experts]
                probabilities used to determine the routing of tokens to the experts.
            padding_mask: [num_groups, tokens_per_group] padding logit mask
                used to identify padding tokens that should be down-weighted by the
                router.
            expert_capacity: Each group will send this many tokens to each expert.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        tokens_per_group = router_probs.shape[1]

        if padding_mask is not None:
            # Because experts choose tokens, we mask probabilities corresponding to
            # tokens before the top-k operation. Note that, unlike for masked-based
            # tokens-choose routing, the experts here may still choose to select the
            # (down-weighted) padding tokens.
            router_probs *= padding_mask.unsqueeze(-1)

        # Transpose router_probs for each group
        router_probs_t = router_probs.transpose(1, 2)

        # Top expert_capacity router probability and corresponding token indices for
        # each expert. Shapes: [num_groups, num_experts, expert_capacity].
        expert_gate, expert_index = torch.topk(
            router_probs_t, k=expert_capacity, dim=-1
        )

        # Convert to one-hot mask of expert indices for each token in each group.
        # Shape: [num_groups, num_experts, expert_capacity, tokens_per_group].
        dispatch_mask = F.one_hot(expert_index, num_classes=tokens_per_group).to(
            torch.int32
        )

        # Move axes to conform with shape expected by MoeLayer API.
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        dispatch_mask = dispatch_mask.permute(0, 3, 1, 2)

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
        # expert_capacity].
        combine_array = torch.einsum(
            "...ec,...tec->...tec", expert_gate, dispatch_mask.float()
        )

        # Return to default dtype now that router computation is complete.
        combine_array = combine_array.to(router_probs.dtype)

        # Each expert is choosing tokens until it reaches full capacity, so we don't
        # need an auxiliary loading balancing loss for expert choice routing.
        auxiliary_loss = 0.0
        if self.auxiliary_loss:
            auxiliary_loss = self._auxiliary_loss(router_probs, dispatch_mask)

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)

    def _auxiliary_loss(self, router_probs: torch.Tensor, dispatch_mask: torch.Tensor):
        return F.binary_cross_entropy(
            router_probs, dispatch_mask.any(dim=-1).float(), reduction="mean"
        )


class VariableCapacityExpertsChooseMaskedRouter(ExpertsChooseMaskedRouter):
    """
    Masked matmul router using experts choose tokens assignment with variable capacities.

    This router allows specifying different capacity factors for each expert.
    """

    def __init__(
        self,
        router_weights: nn.Linear,
        jitter_noise: float,
        dtype: torch.dtype,
        ignore_padding_tokens: bool,
        capacity_factors: Tuple[float, ...],
        auxiliary_loss: bool = False,
    ):
        super().__init__(
            router_weights, jitter_noise, dtype, ignore_padding_tokens, auxiliary_loss
        )
        self.capacity_factors = capacity_factors
        assert (
            len(capacity_factors) == router_weights.out_features
        ), "Number of capacity factors must match number of experts"
        assert sum(capacity_factors) == 1.0, "Capacity factors must sum to 1.0"

    def _compute_routing_instructions(
        self,
        router_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        expert_capacity: int,
    ) -> RouterMask:
        num_groups, tokens_per_group, num_experts = router_probs.shape

        if padding_mask is not None:
            router_probs *= padding_mask.unsqueeze(-1)

        router_probs_t = router_probs.transpose(1, 2)

        # Calculate individual expert capacities
        expert_capacities = [
            max(1, int(factor * expert_capacity)) for factor in self.capacity_factors
        ]

        # Initialize tensors to store results
        dispatch_mask = torch.zeros(
            num_groups,
            tokens_per_group,
            num_experts,
            expert_capacity,
            dtype=torch.int32,
            device=router_probs.device,
        )
        combine_array = torch.zeros(
            num_groups,
            tokens_per_group,
            num_experts,
            expert_capacity,
            dtype=router_probs.dtype,
            device=router_probs.device,
        )

        # Perform top-k for each expert separately
        for i, capacity in enumerate(expert_capacities):
            expert_gate, expert_index = torch.topk(
                router_probs_t[:, i], k=capacity, dim=-1
            )

            # Create dispatch mask for this expert
            dispatch_mask[:, :, i, :capacity] = F.one_hot(
                expert_index, num_classes=tokens_per_group
            ).permute(0, 2, 1)

            # Create combine array for this expert
            combine_array[:, :, i, :capacity] = expert_gate.unsqueeze(
                1
            ) * dispatch_mask[:, :, i, :capacity].to(router_probs.dtype)

        auxiliary_loss = 0.0
        if self.auxiliary_loss:
            auxiliary_loss = self._auxiliary_loss(router_probs, dispatch_mask)

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


class TokensChooseMaskedRouter(MaskedRouter):
    """
    Masked matmul router using tokens choose top-k experts assignment.

    This router uses the same mechanism as in Switch Transformer
    (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are
    sorted by router_probs and then routed to their choice of expert until the
    expert's expert_capacity is reached. There is no guarantee that each token is
    processed by an expert, or that each expert receives at least one token.

    Attributes:
        num_selected_experts: Maximum number of experts to which each token is
            routed. Tokens may be routed to fewer experts if particular experts are
            oversubscribed / reach capacity.
        batch_prioritized_routing: Whether or not to use Batch Prioritized Routing
            (BPR), originally introduced in V-MoE (https://arxiv.org/abs/2106.05974).
            With BPR, we prioritize routing those top-k tokens with the highest
            router probability, rather than simply using each tokens left-to-right
            ordering in the batch. This prioritization is important because the
            experts have limited capacity.
    """

    def __init__(
        self,
        router_weights: nn.Linear,
        jitter_noise: float,
        dtype: torch.dtype,
        ignore_padding_tokens: bool,
        num_selected_experts: int,
        batch_prioritized_routing: bool,
    ):
        super().__init__(router_weights, jitter_noise, dtype, ignore_padding_tokens)
        self.num_selected_experts = num_selected_experts
        self.batch_prioritized_routing = batch_prioritized_routing

    def _compute_routing_instructions(
        self,
        router_probs: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        expert_capacity: int,
    ) -> RouterMask:
        """
        Computes masks for the top-k experts per token.

        Args:
            router_probs: [num_groups, tokens_per_group, num_experts]
                probabilities used to determine the routing of tokens to the experts.
            padding_mask: [num_groups, tokens_per_group] padding logit mask
                used to identify padding tokens that should be ignored by the router.
            expert_capacity: Each group will send this many tokens to each expert.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        num_groups, _, num_experts = router_probs.shape

        # Top-k router probability and corresponding expert indices for each token.
        # Shape: [num_groups, tokens_per_group, num_selected_experts].
        expert_gate, expert_index = torch.topk(
            router_probs, k=self.num_selected_experts, dim=-1
        )

        if padding_mask is not None:
            # Mask applied to gate. Exclude choices corresponding to padding tokens.
            gate_mask = padding_mask.unsqueeze(-1)
            expert_gate *= gate_mask

            # Set `expert_index` elements corresponding to padding to negative
            # numbers. Negative `expert_index` elements will ultimately be dropped in
            # the one_hot conversion to the `expert_mask`.
            # First convert nonzero padding elements to negative values.
            expert_index *= 2 * gate_mask - 1
            # Handle zero padding elements by negatively shifting all padding.
            expert_index += (gate_mask - 1).repeat(1, 1, self.num_selected_experts)

            # To correctly compute load balancing loss, we also mask out probs.
            router_probs *= gate_mask

        auxiliary_loss = load_balancing_loss(router_probs, expert_index)

        if self.batch_prioritized_routing:
            # Sort tokens according to their routing probability per group, so that
            # the highest probability tokens are routed first.
            permutation = torch.argsort(-expert_gate[..., 0], dim=-1)
            # Shape: [num_groups, tokens_per_group, num_selected_experts]
            expert_index = torch.take_along_dim(
                expert_index, permutation.unsqueeze(-1), dim=-2
            )

        # Make num_selected_experts the leading axis to ensure that top-1 choices
        # have priority over top-2 choices, which have priority over top-3 choices,
        # etc.
        expert_index = expert_index.transpose(1, 2)
        # Shape: [num_groups, num_selected_experts * tokens_per_group]
        expert_index = expert_index.reshape(num_groups, -1)

        # Create mask out of indices.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)

        # Experts have a fixed capacity that we cannot exceed. A token's priority
        # within the expert's buffer is given by the masked, cumulative capacity of
        # its target expert.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0
        # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape(
            (num_groups, self.num_selected_experts, -1, num_experts)
        )
        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        token_priority = token_priority.transpose(1, 2)
        # For each token, across all selected experts, select the only non-negative
        # (unmasked) priority. Now, for group G routing to expert E, token T has
        # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
        # is its targeted expert.
        # Shape: [num_groups, tokens_per_group, num_experts].
        token_priority = torch.max(token_priority, dim=2).values

        if self.batch_prioritized_routing:
            # Place token priorities in original ordering of tokens.
            inv_permutation = torch.argsort(permutation, dim=-1)
            token_priority = torch.take_along_dim(
                token_priority, inv_permutation.unsqueeze(-1), dim=-2
            )

        # Token T can only be routed to expert E if its priority is positive and
        # less than the expert capacity. One-hot matrix will ignore indices outside
        # the range [0, expert_capacity).
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity].
        dispatch_mask = F.one_hot(
            token_priority.add(1).clamp(0, expert_capacity).long(), expert_capacity + 1
        )
        dispatch_mask = dispatch_mask[..., 1:].bool()

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
        # expert_capacity].
        combine_array = torch.einsum(
            "...te,...tec->...tec", router_probs, dispatch_mask.float()
        )

        # Return to default dtype now that router computation is complete.
        combine_array = combine_array.to(router_probs.dtype)

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


def load_balancing_loss(
    router_probs: torch.Tensor, expert_indices: torch.Tensor
) -> float:
    """
    Computes auxiliary load balancing loss as in Switch Transformer.
    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
    implements the loss function presented in equations (4) - (6). It aims to
    penalize those cases where the routing between experts is unbalanced.

    Args:
        router_probs: Probability assigned to each expert per token. Shape:
            [num_groups, tokens_per_group, num_experts].
        expert_indices: [num_groups, tokens_per_group, num_selected_experts]
            indices identifying the top num_selected_experts for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.size(-1)
    # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_indices, num_experts).to(torch.int32)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [num_groups, tokens_per_group, num_experts]
    expert_mask = torch.max(expert_mask, dim=-2).values
    tokens_per_group_and_expert = torch.mean(expert_mask.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(router_probs.float(), dim=-2)
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert,
    ) * (num_experts**2)


def router_z_loss(router_logits: torch.Tensor) -> float:
    """
    Compute router z-loss.
    The router z-loss was introduced in Designing Effective Sparse Expert Models
    (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
    small in an effort to improve stability.

    Args:
        router_logits: [num_groups, tokens_per_group, num_experts] router logits.

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss).item() / (num_groups * tokens_per_group)
