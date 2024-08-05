import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Callable, List
from dataclasses import dataclass, replace


class Mlp(nn.Module):
    """
    A two layer MLP with choice of activation function.

    Attributes:
        input_dim: Dimension of the input tensor.
        output_dim: Dimension of the output tensor.
        multiplier: Multiplier for the hidden layer dimension.
        hidden_dim: Dimension of the hidden layer. If None, it is set to 0.5 * input_dim.
        activation: Activation function to use. Default is GELU.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        multiplier: int = 0.5,
        hidden_dim: Optional[int] = None,
        activation: Optional[Callable] = F.gelu,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = (
            int(input_dim * multiplier) if hidden_dim is None else hidden_dim
        )
        self.output_dim = output_dim
        self.activation = activation

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


@dataclass
class RouterMask:
    """
    Dataclass to store the routing instructions for masked matmul dispatch routers.

    Attributes:
        dispatch_mask (torch.Tensor): Boolean tensor indicating which tokens are routed to which experts.
            Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        combine_array (torch.Tensor): Float tensor used for combining expert outputs and scaling with router probability.
            Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        auxiliary_loss (torch.Tensor): Load balancing loss for the router.
        router_z_loss (torch.Tensor): Router z-loss to encourage router logits to remain small for improved stability.
        router_causal_loss (torch.Tensor): Router causal loss to encourage top-k router probabilities to be above 0.5.
    """

    dispatch_mask: torch.Tensor
    combine_array: torch.Tensor
    auxiliary_loss: Optional[torch.Tensor] = None
    router_z_loss: Optional[torch.Tensor] = None
    router_causal_loss: Optional[torch.Tensor] = None

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

    def __init__(self, dim: int, num_experts: int, bias: bool = True):
        super().__init__(dim, num_experts, bias=bias, dtype=torch.float32)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=2e-2)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Router(nn.Module):
    """Abstract base router class, defining router API and inner workings.

    Attributes:
        router_weights: Configurable module used to compute router logits from token
        inputs.
        jitter_noise: Amplitude of jitter noise applied to router logits.
        dtype: Numeric float type for returned combine array. All actual
        computations are performed in float32 of the input for stability.
        ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
        that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
        padding tokens, while others (e.g. TokensChooseScatterRouter and
        ExpertsChooseMaskedRouter) will simply down-weight the probability of
        selecting padding tokens.
    """

    def __init__(
        self,
        router_weights: RouterWeights,
        jitter_noise: float,
        ignore_padding_tokens: bool,
        router_causal_loss: Optional[Callable] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.router_weights = router_weights
        self.jitter_noise = jitter_noise
        self.ignore_padding_tokens = ignore_padding_tokens
        self.router_causal_loss = router_causal_loss
        self.dtype = dtype

    def forward(
        self,
        token_inputs: torch.Tensor,
        expert_capacity: int,
        apply_jitter: bool = True,
    ) -> RouterMask:
        token_inputs = token_inputs.to(torch.float32)

        if self.dtype is None:
            self.dtype = token_inputs.dtype

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

        router_z_loss = _router_z_loss(router_logits)

        router_causal_loss = torch.tensor(
            0.0, device=token_inputs.device, dtype=self.dtype
        )
        if self.router_causal_loss is not None:
            router_causal_loss = self.router_causal_loss(
                token_inputs, routing_instructions.dispatch_mask
            )

        return routing_instructions.replace(
            router_z_loss=router_z_loss, router_causal_loss=router_causal_loss
        )

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

    Attributes:
        router_weights: Configurable module used to compute router logits from token
            inputs.
        jitter_noise: Amplitude of jitter noise applied to router logits.
        ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
            that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
            padding tokens, while others (e.g. TokensChooseScatterRouter and
            ExpertsChooseMaskedRouter) will simply down-weight the probability of
            selecting padding tokens.
        return_residual_mask: Whether to return a mask for routing tokens to the
            residual expert.
        router_causal_loss: Router causal loss to encourage top-k router probabilities
            to be above 0.5.
        dtype: Numeric float type for returned combine array. All actual
            computations are performed in float32 of the input for stability.
    """

    def __init__(
        self,
        router_weights: nn.Linear,
        jitter_noise: float,
        ignore_padding_tokens: bool,
        router_causal_loss: Optional[Callable] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            router_weights,
            jitter_noise,
            ignore_padding_tokens,
            router_causal_loss,
            dtype,
        )

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
        combine_array = combine_array.to(self.dtype)

        # Each expert is choosing tokens until it reaches full capacity, so we don't
        # need an auxiliary loading balancing loss for expert choice routing.
        auxiliary_loss = torch.tensor(0.0, device=router_probs.device, dtype=self.dtype)

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


class ExpertsPreferredMaskedRouter(MaskedRouter):
    """
    Masked matmul router using expert-preferred token assignment with variable capacities.

    This router is similar to ExpertsChooseMaskedRouter but allows for different capacities
    for each expert. Experts choose tokens in order, with each expert selecting up to its
    specified capacity.

    Attributes:
        router_weights: Configurable module used to compute router logits from token inputs.
        jitter_noise: Amplitude of jitter noise applied to router logits.
        dtype: Numeric float type for returned combine array. All actual computations
            are performed in float32 of the input for stability.
        ignore_padding_tokens: Whether to ignore padding tokens during routing.
    """

    def __init__(
        self,
        router_weights: nn.Linear,
        jitter_noise: float,
        ignore_padding_tokens: bool,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            router_weights, jitter_noise, ignore_padding_tokens, dtype=dtype
        )

    def _compute_routing_instructions(
        self,
        router_probs: torch.Tensor,
        expert_capacities: List[int],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> RouterMask:
        """
        Compute routing instructions based on router probabilities and expert capacities.

        Args:
            router_probs: Router probabilities.
                Shape: [num_groups, tokens_per_group, num_experts]
            expert_capacities: List of capacities for each expert.
            padding_mask: Optional boolean tensor indicating which tokens are padding.
                Shape: [num_groups, tokens_per_group]

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        num_groups, tokens_per_group, num_experts = router_probs.shape
        device = router_probs.device

        assert len(expert_capacities) == num_experts, "Number of expert capacities must match number of experts"

        if padding_mask is not None:
            router_probs = router_probs * padding_mask.unsqueeze(-1)

        max_capacity = max(expert_capacities)
        dispatch_mask = torch.zeros(
            (num_groups, tokens_per_group, num_experts, max_capacity),
            dtype=torch.int32,
            device=device,
        )
        combine_array = torch.zeros_like(dispatch_mask, dtype=self.dtype)

        # Transpose router_probs for each group
        router_probs_t = router_probs.transpose(1, 2)

        for expert_idx, capacity in enumerate(expert_capacities):
            expert_probs = router_probs_t[:, expert_idx, :]
            available_mask = (dispatch_mask.sum(dim=(2, 3)) == 0).float()
            masked_probs = expert_probs * available_mask

            expert_gate, expert_index = torch.topk(masked_probs, k=capacity, dim=-1)

            # Update dispatch mask
            dispatch_mask[torch.arange(num_groups).unsqueeze(1), expert_index, expert_idx, torch.arange(capacity)] = 1

            # Update combine array
            combine_array[torch.arange(num_groups).unsqueeze(1), expert_index, expert_idx, torch.arange(capacity)] = expert_gate

        # Reshape dispatch mask and combine array to match expected output shape
        dispatch_mask = dispatch_mask.permute(0, 1, 2, 3)
        combine_array = combine_array.permute(0, 1, 2, 3)

        # Calculate auxiliary loss (you may want to implement a custom loss for this router)
        auxiliary_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

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
        router_weights: Configurable module used to compute router logits from token
            inputs.
        jitter_noise: Amplitude of jitter noise applied to router logits.
        dtype: Numeric float type for returned combine array. All actual
            computations are performed in float32 of the input for stability.
        ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
            that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
            padding tokens, while others (e.g. TokensChooseScatterRouter and
            ExpertsChooseMaskedRouter) will simply down-weight the probability of
            selecting padding tokens.
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
        ignore_padding_tokens: bool,
        num_selected_experts: int,
        batch_prioritized_routing: bool,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            router_weights, jitter_noise, ignore_padding_tokens, dtype=dtype
        )
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

        auxiliary_loss = _load_balancing_loss(router_probs, expert_index)

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
        ).to(self.dtype)

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


def _load_balancing_loss(
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


def _router_z_loss(router_logits: torch.Tensor) -> float:
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
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


class RouterCausalLoss(nn.Module):
    """
    Encourages topk router probabilities to be above 0.5
    when sampling autoregressively.

    Args:
        dim: Dimension of the input tensor.

    Returns:
        Binary cross entropy loss.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.router_predictor = Mlp(dim, 1)

    def forward(
        self,
        input_tokens: torch.Tensor,
        dispatch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_tokens: [num_groups, tokens_per_group, dim]
            dispatch_mask: [num_groups, tokens_per_group, num_experts, expert_capacity]
        """
        router_target = dispatch_mask.any(dim=(2, 3)).float()
        router_logits = self.router_predictor(input_tokens)
        return F.binary_cross_entropy_with_logits(router_logits, router_target)
