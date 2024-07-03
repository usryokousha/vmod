import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, List
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

    def replace(self, **kwargs: Any) -> 'RouterMask':
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
    def __init__(self, hidden_dim: int, num_experts: int, bias: bool = True, **kwargs):
        super().__init__(hidden_dim, num_experts, bias=bias, **kwargs)
    
    
class Mlp(nn.Module):
    """
    MLP with GELU activation and optional dropout.
    """
    def __init__(self, input_dim: int, multiplier: int = 4, hidden_dim: Optional[int] = None, output_dim: int = None, dropout_rate: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = multiplier * input_dim
        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, enable_dropout: bool = False) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        if enable_dropout:
            x = self.dropout(x)
        return self.fc2(x)

class Router(nn.Module):
    """
    Abstract base router class defining the router API and inner workings.

    Attributes:
        router_hidden_dim (int): Dimensionality of the input tokens.
        num_experts (int): Number of experts to route to.
        jitter_noise (float): Amplitude of jitter noise applied to router logits.
        dtype (torch.dtype): Numeric float type for returned combine array.
        ignore_padding_tokens (bool): Whether to ignore padding tokens during routing.
    """
    def __init__(self, router_hidden_dim: int, num_experts: int, jitter_noise: float, dtype: torch.dtype, ignore_padding_tokens: bool):
        super().__init__()
        self.router_weights = RouterWeights(router_hidden_dim, num_experts)
        self.jitter_noise = jitter_noise
        self.dtype = dtype
        self.ignore_padding_tokens = ignore_padding_tokens

    def forward(self, token_inputs: torch.Tensor, expert_capacity: int, apply_jitter: bool = True) -> RouterMask:
        """
        Compute dispatch and combine arrays for routing to experts.

        Args:
            token_inputs (torch.Tensor): Input tensor of shape [num_groups, tokens_per_group, hidden_dim].
            expert_capacity (int): Number of tokens each expert can process.
            apply_jitter (bool): Whether to apply jitter noise during routing.

        Returns:
            RouterMask: Routing instructions.
        """
        router_probs, router_logits = self._compute_router_probabilities(token_inputs, apply_jitter)

        if self.ignore_padding_tokens:
            padding_mask = (torch.sum(torch.abs(token_inputs), dim=-1) > 0).to(token_inputs.dtype)
            router_logits *= padding_mask.unsqueeze(-1)
        else:
            padding_mask = None

        instructions = self._compute_routing_instructions(router_probs, padding_mask, expert_capacity)
        return instructions.replace(router_z_loss=self._router_z_loss(router_logits))

    def _compute_router_probabilities(self, token_inputs: torch.Tensor, apply_jitter: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute router probabilities from input tokens.

        Args:
            token_inputs (torch.Tensor): Input tensor of shape [num_groups, tokens_per_group, hidden_dim].
            apply_jitter (bool): Whether to apply jitter noise.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Router probabilities and logits, both of shape [num_groups, tokens_per_group, num_experts].
        """
        token_inputs = token_inputs.to(torch.float32)

        if apply_jitter and self.jitter_noise > 0:
            token_inputs *= 1.0 + torch.rand_like(token_inputs) * self.jitter_noise * 2 - self.jitter_noise

        router_logits = self.router_weights(token_inputs)
        router_probabilities = F.softmax(router_logits, dim=-1)

        return router_probabilities, router_logits

    def _compute_routing_instructions(self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int) -> RouterMask:
        """
        Compute instructions for routing inputs to experts.

        Args:
            router_probs (torch.Tensor): Router probabilities of shape [num_groups, tokens_per_group, num_experts].
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [num_groups, tokens_per_group] or None.
            expert_capacity (int): Number of tokens each expert can process.

        Returns:
            RouterMask: Routing instructions.
        """
        raise NotImplementedError("Router is an abstract class that should be subclassed.")
    
    @staticmethod
    def _router_z_loss(router_logits: torch.Tensor) -> float:
        """
        Compute router z-loss to encourage router logits to remain small.

        Args:
            router_logits (torch.Tensor): Router logits of shape [num_groups, tokens_per_group, num_experts].

        Returns:
            float: Scalar router z-loss.
        """
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = log_z ** 2
        return z_loss.sum().item() / z_loss.numel()


class MaskedRouter(Router):
    """
    Abstract base router class for masked matmul dispatch routers.
    """
    def _compute_routing_instructions(self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int) -> RouterMask:
        """
        Compute masks for routing inputs to experts.

        Args:
            router_probs (torch.Tensor): Router probabilities of shape [num_groups, tokens_per_group, num_experts].
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [num_groups, tokens_per_group] or None.
            expert_capacity (int): Number of tokens each expert can process.

        Returns:
            RouterMask: Routing instructions.
        """
        raise NotImplementedError("MaskedRouter is an abstract class that should be subclassed.")

class ExpertsChooseMaskedRouter(MaskedRouter):
    """
    Masked matmul router using experts choose tokens assignment.

    This router uses the mechanism from "Mixture-of-Experts with Expert Choice":
    each expert selects its top expert_capacity tokens. An individual token may
    be processed by multiple experts or none at all.
    """
    def _compute_routing_instructions(self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int) -> RouterMask:
        """
        Compute masks for the highest probability tokens per expert.

        Args:
            router_probs (torch.Tensor): Router probabilities of shape [num_groups, tokens_per_group, num_experts].
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [num_groups, tokens_per_group] or None.
            expert_capacity (int): Number of tokens each expert can process.

        Returns:
            RouterMask: Routing instructions.
        """
        tokens_per_group = router_probs.shape[1]

        if padding_mask is not None:
            router_probs *= padding_mask.unsqueeze(-1)

        # Transpose for expert-first view
        router_probs_t = router_probs.transpose(1, 2)

        # Top expert_capacity router probability and corresponding token indices for each expert
        expert_gate, expert_index = torch.topk(router_probs_t, k=expert_capacity, dim=-1)

        # Convert to one-hot mask of expert indices for each token in each group
        dispatch_mask = F.one_hot(expert_index, num_classes=tokens_per_group).to(torch.int32)

        # Move axes to conform with shape expected by MoeLayer API
        dispatch_mask = dispatch_mask.permute(0, 3, 1, 2)

        # Compute combine array
        combine_array = torch.einsum('...ec,...tec->...tec', expert_gate, dispatch_mask.float())

        # Convert to default dtype
        combine_array = combine_array.to(self.dtype)

        # No auxiliary loss for expert choice routing
        auxiliary_loss = 0.0

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)

class TokensChooseMaskedRouter(MaskedRouter):
    """
    Masked matmul router using tokens choose top-k experts assignment.

    This router allows each token to choose its top experts. Tokens are sorted by
    router probabilities and then routed to their choice of expert until the
    expert's capacity is reached. There is no guarantee that each token is
    processed by an expert, or that each expert receives at least one token.

    Attributes:
        num_selected_experts (int): Maximum number of experts to which each token is routed.
        batch_prioritized_routing (bool): Whether to use Batch Prioritized Routing (BPR).
    """

    def __init__(self, router_hidden_dim: int, num_experts: int,  jitter_noise: float, dtype: torch.dtype,
                 ignore_padding_tokens: bool, num_selected_experts: int, batch_prioritized_routing: bool):
        super().__init__(router_hidden_dim, num_experts, jitter_noise, dtype, ignore_padding_tokens)
        self.num_selected_experts = num_selected_experts
        self.batch_prioritized_routing = batch_prioritized_routing

    def _compute_routing_instructions(self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int) -> RouterMask:
        """
        Compute masks for the top-k experts per token.

        Args:
            router_probs (torch.Tensor): Router probabilities of shape [num_groups, tokens_per_group, num_experts].
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [num_groups, tokens_per_group] or None.
            expert_capacity (int): Number of tokens each expert can process.

        Returns:
            RouterMask: Routing instructions.
        """
        num_groups, tokens_per_group, num_experts = router_probs.shape

        if padding_mask is not None:
            router_probs = router_probs * padding_mask.unsqueeze(-1)

        # Top-k router probability and corresponding expert indices for each token
        expert_gate, expert_index = torch.topk(router_probs, k=self.num_selected_experts, dim=-1)

        if self.batch_prioritized_routing:
            # Sort tokens by their top probability
            sort_indices = torch.argsort(expert_gate[..., 0], dim=-1, descending=True)
            expert_index = torch.gather(expert_index, 1, sort_indices.unsqueeze(-1).expand_as(expert_index))
            expert_gate = torch.gather(expert_gate, 1, sort_indices.unsqueeze(-1).expand_as(expert_gate))

        # Create dispatch mask
        dispatch_mask = torch.zeros(num_groups, tokens_per_group, num_experts, expert_capacity, dtype=torch.bool, device=router_probs.device)
        
        for i in range(self.num_selected_experts):
            # For each expert choice, find the number of tokens already assigned to each expert
            expert_counts = torch.sum(dispatch_mask, dim=1)
            
            # Create a mask for tokens that can still be assigned (haven't reached expert capacity)
            can_assign = expert_counts[torch.arange(num_groups).unsqueeze(1), expert_index[:, :, i]] < expert_capacity
            
            # Assign tokens to experts
            token_indices = torch.arange(tokens_per_group, device=router_probs.device).unsqueeze(0).expand(num_groups, -1)
            dispatch_mask[torch.arange(num_groups).unsqueeze(1), token_indices, expert_index[:, :, i], expert_counts[torch.arange(num_groups).unsqueeze(1), expert_index[:, :, i]]] = can_assign

        # Create combine array
        combine_array = torch.zeros_like(dispatch_mask, dtype=self.dtype)
        combine_array[dispatch_mask] = expert_gate.view(-1)[dispatch_mask.view(-1)]

        # Compute load balancing auxiliary loss
        aux_loss = self._load_balancing_loss(router_probs, expert_index)

        return RouterMask(dispatch_mask, combine_array, aux_loss)

    @staticmethod
    def _load_balancing_loss(router_probs: torch.Tensor, expert_index: torch.Tensor) -> float:
        """
        Compute auxiliary load balancing loss as in Switch Transformer.

        Args:
            router_probs (torch.Tensor): Router probabilities of shape [num_groups, tokens_per_group, num_experts].
            expert_index (torch.Tensor): Expert indices of shape [num_groups, tokens_per_group, num_selected_experts].

        Returns:
            float: Auxiliary loss for load balancing.
        """
        num_experts = router_probs.shape[-1]
        
        # Create a mask of the selected experts
        expert_mask = F.one_hot(expert_index, num_classes=num_experts).sum(dim=-2)
        expert_mask = (expert_mask > 0).float()

        tokens_per_group_and_expert = expert_mask.float().mean(dim=-2)
        router_prob_per_group_and_expert = router_probs.mean(dim=-2)
        
        return (tokens_per_group_and_expert * router_prob_per_group_and_expert).mean() * (num_experts ** 2)
    
class VariableCapacityMaskedRouter(MaskedRouter):
    """
    Masked matmul router using experts choose tokens assignment with variable expert capacities.

    This router uses a mechanism similar to "Mixture-of-Experts with Expert Choice":
    each expert selects its top tokens based on its specific capacity. An individual token may
    be processed by multiple experts or none at all.
    """
    def __init__(self, router_hidden_dim: int, num_experts: int, jitter_noise: float, dtype: torch.dtype,
                 ignore_padding_tokens: bool, expert_capacity_factors: List[float]):
        super().__init__(router_hidden_dim, num_experts, jitter_noise, dtype, ignore_padding_tokens)
        self.expert_capacity_factors = expert_capacity_factors
        assert len(expert_capacity_factors) == num_experts, "Number of expert capacity factors must match number of experts"

    def _compute_routing_instructions(self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], base_expert_capacity: int) -> RouterMask:
        """
        Compute masks for the highest probability tokens per expert, respecting variable expert capacities.

        Args:
            router_probs (torch.Tensor): Router probabilities of shape [num_groups, tokens_per_group, num_experts].
            padding_mask (Optional[torch.Tensor]): Padding mask of shape [num_groups, tokens_per_group] or None.
            base_expert_capacity (int): Base capacity for experts, which will be multiplied by each expert's capacity factor.

        Returns:
            RouterMask: Routing instructions.
        """
        num_groups, tokens_per_group, num_experts = router_probs.shape

        if padding_mask is not None:
            router_probs *= padding_mask.unsqueeze(-1)

        # Transpose for expert-first view
        router_probs_t = router_probs.transpose(1, 2)

        # Calculate expert capacities
        expert_capacities = [int(base_expert_capacity * factor) for factor in self.expert_capacity_factors]
        max_expert_capacity = max(expert_capacities)

        # Top-k router probability and corresponding token indices for each expert
        expert_gates = []
        expert_indices = []
        for i, capacity in enumerate(expert_capacities):
            gate, index = torch.topk(router_probs_t[:, i], k=capacity, dim=-1)
            expert_gates.append(gate)
            expert_indices.append(index)

        # Convert to one-hot mask of expert indices for each token in each group
        dispatch_mask = torch.zeros(num_groups, tokens_per_group, num_experts, max_expert_capacity, dtype=torch.int32, device=router_probs.device)
        for i, (indices, capacity) in enumerate(zip(expert_indices, expert_capacities)):
            dispatch_mask[:, :, i, :capacity] = F.one_hot(indices, num_classes=tokens_per_group).to(torch.int32)

        # Move axes to conform with shape expected by MoeLayer API
        dispatch_mask = dispatch_mask.permute(0, 2, 1, 3)

        # Compute combine array
        combine_array = torch.zeros_like(dispatch_mask, dtype=self.dtype)
        for i, (gate, capacity) in enumerate(zip(expert_gates, expert_capacities)):
            combine_array[:, i, :, :capacity] = gate.unsqueeze(1)

        # No auxiliary loss for expert choice routing
        auxiliary_loss = 0.0

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)