import logging
import os

import torch
import torch.distributed as dist


def load_checkpoint(args, model, optimizer, lr_scheduler) -> None:
    """
    Loads model, optimizer, and learning rate scheduler states from a checkpoint file.

    This function attempts to load a checkpoint from a specified path, or from default
    locations within the output directory. If a checkpoint is found, it loads the model
    weights, and optionally the optimizer and learning rate scheduler states, unless
    `args.only_load_model_weights` is set to True.

        args: An object containing configuration parameters. Must have the following attributes:
            - output_dir (str): Directory where checkpoints are stored.
            - ckpt_path (str): Path to a specific checkpoint file.
            - only_load_model_weights (bool): If True, only the model weights are loaded.
        model (torch.nn.Module): The model whose state will be loaded.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be loaded (if present in checkpoint).
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler whose state will be loaded (if present in checkpoint).

    Returns:
        None

    Logs:
        - Information about which checkpoint is loaded, or if no checkpoint is found.
        - Confirmation of loading model, optimizer, and lr_scheduler states.
    """
    logger = logging.getLogger("xclip")
    ckpt_top_res = os.path.join(args.output_dir, "checkpoint_top_result.pt")
    ckpt = os.path.join(args.output_dir, "checkpoint.pt")
    if os.path.isfile(args.ckpt_path):
        logger.info(f"loading checkpoint from: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path)
    elif os.path.isfile(ckpt_top_res):
        logger.info(f"found and loading top result checkpoint from: {ckpt_top_res}")
        ckpt = torch.load(ckpt_top_res)
    elif os.path.isfile(ckpt):
        logger.info(f"found and loading checkpoint from: {ckpt}")
        ckpt = torch.load(ckpt)
    else:
        logger.info(f"no checkpoint found in {args.output_dir} or at {args.ckpt_path}")
        return

    model.load_state_dict(ckpt["model"], strict=False)
    logger.info("model state dict loaded")
    if ("optimizer" in ckpt) and (not args.only_load_model_weights):
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("optimizer state dict loaded")
    if ("lr_scheduler" in ckpt) and (not args.only_load_model_weights):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        logger.info("lr_scheduler state dict loaded")
    return None


def save_checkpoint(
    cfg, args, model, optim, lr_scheduler, epoch, lowest_loss=False
) -> None:
    """
    Saves the current training checkpoint, including model state, optimizer state, learning rate scheduler state, epoch, arguments, and configuration.
    Args:
        cfg (Any): Configuration object or dictionary containing experiment settings.
        args (Any): Arguments object, typically containing runtime parameters such as output directory.
        model (torch.nn.Module): The model whose state is to be saved.
        optim (torch.optim.Optimizer): The optimizer whose state is to be saved.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler whose state is to be saved.
        epoch (int): The current epoch number.
        lowest_loss (bool, optional): If True, saves the checkpoint as 'checkpoint_lowest_loss.pt'; otherwise, saves as 'checkpoint.pt'. Defaults to False.
    Returns:
        None
    """

    filename = "checkpoint_lowest_loss.pt" if lowest_loss else "checkpoint.pt"
    checkpoint_path = os.path.join(args.output_dir, filename)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "args": args,
        "config": cfg,
    }
    print(f"saving checkpoint at: {checkpoint_path}")
    save_on_master(ckpt, checkpoint_path)
    return None


def is_dist_avail_and_initialized():
    """
    Checks if distributed training is available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """
    Gets the rank of the current process in distributed training.

    Returns:
        int: The rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """ Checks if the current process is the main process in distributed training.
    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """ Saves the given arguments to a file only if the current process is the main process in distributed training.
    Args:
        *args: Positional arguments to be saved.
        **kwargs: Keyword arguments to be saved.
    """
    if is_main_process():
        torch.save(*args, **kwargs)
