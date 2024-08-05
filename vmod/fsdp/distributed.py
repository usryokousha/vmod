import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def create_expert_process_groups(num_experts, world_size):
    """
    Create process groups for each expert.
    """
    expert_groups = []
    for i in range(num_experts):
        ranks = [i % world_size]  # Assign expert to a single GPU
        group = dist.new_group(ranks)
        expert_groups.append(group)
    return expert_groups


class ExpertModule(torch.nn.Module):
    def __init__(self, expert_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_id = expert_id
        self.process_group = None

    def set_process_group(self, process_group):
        self.process_group = process_group

    def to_gpu(self):
        device = torch.device(f"cuda:{self.expert_id % torch.cuda.device_count()}")
        self.to(device)


def moe_auto_wrap_policy(
    module,
    recurse,
    unwrapped_params,
    module_is_root,
    *,
    min_num_params=1e8,
    force_leaf_modules=None,
    exclude_wrap_modules=None,
):
    """
    A custom auto wrap policy for Mixture of Experts models.

    This policy wraps experts individually with their own process groups
    and applies the default FSDP wrapping policy to other parts of the model.
    """

    if isinstance(module, ExpertModule):
        return True

    if not module_is_root:
        return size_based_auto_wrap_policy(
            module,
            recurse,
            unwrapped_params,
            module_is_root,
            min_num_params=min_num_params,
            force_leaf_modules=force_leaf_modules,
            exclude_wrap_modules=exclude_wrap_modules,
        )

    return False


def create_moe_model_with_fsdp(model, num_experts):
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create process groups for experts
    expert_groups = create_expert_process_groups(num_experts, world_size)

    # Assign each expert to a GPU and set its process group
    for i, expert in enumerate(model.experts):
        if i % world_size == rank:
            expert.to_gpu()
            expert.set_process_group(expert_groups[i])

    # Wrap experts with FSDP individually
    for i, expert in enumerate(model.experts):
        if i % world_size == rank:
            model.experts[i] = FSDP(
                expert,
                process_group=expert.process_group,
                device_id=torch.cuda.current_device(),
            )

    # Wrap the entire model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=moe_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )

    return fsdp_model
