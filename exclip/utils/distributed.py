import os

import torch
import torch.distributed as dist


def setup_distributed():
    """
    Initializes distributed training environment using PyTorch.
    This function sets the CUDA device for the current process based on the
    'LOCAL_RANK' environment variable and initializes the default process group
    for distributed training using the NCCL backend.

    Returns:
        int: The local rank of the current process, as specified by the 'LOCAL_RANK' environment variable.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return local_rank


def cleanup():
    """
    Cleans up the distributed training environment by destroying the default process group.
    """
    dist.destroy_process_group()
