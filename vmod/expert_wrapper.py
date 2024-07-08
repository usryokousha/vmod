import torch.nn as nn

from typing import Any



class ExpertWrapper(nn.Module):
    """
    Wrapper class for Mixture-of-Depths experts.

    Unlike normal Mixtures-of-Experts, Mixture-of-Depths experts
    can perform routing on any arbitrary kind of computation.  This can
    be extended to blocks with different configurations which share weights.
    
    This can be used to wrap an attention block with specific parameters
    for each expert.  The wrapper will forward all attributes to the
    original module.  This is particularly useful when you want each expert
    to have different sized attention windows.

    Usage:
    ```
    original_attention = FlashSelfAttention(
        causal=False,
        window_size=(-1, -1))

    expert_wrapper_16 = ExpertWrapper(
                    original_attention,
                    window_size=(16, 16))

    expert_wrapper_full = ExpertWrapper(
                    original_attention,
                    window_size=(-1, -1))

    mod_experts = [expert_wrapper_16, expert_wrapper_full]
    ```
    """

    def __init__(self, original_module: nn.Module, **kwargs):
        super().__init__()
        # make copy of properties of the original module
        self.original_module = original_module
        for name, value in vars(original_module).items():
            setattr(self, name, kwargs.get(name, value))

    def __getattr__(self, name) -> Any:
        """
        Forward attribute access to the original module.
        """
        try:
            return self.original_module.__getattr__(name)
        except AttributeError:
            print(f"AttributeError: {name} not found in {self.original_module}")
