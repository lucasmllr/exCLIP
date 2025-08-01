import logging

import torch.optim as optim
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from torch.optim.lr_scheduler import ExponentialLR


def get_optimizer(cfg, params):
    """
    Build and return the optimizer and learning rate scheduler based on the configuration.

    Args:
        cfg: Configuration object containing optimizer and scheduler settings.
        params: Iterable of parameters to optimize.

    Returns:
        Tuple[torch.optim.Optimizer, ignite.handlers.param_scheduler.ParamScheduler]:
            The constructed optimizer and learning rate scheduler.
    """
    logger = logging.getLogger("exclip")
    logger.info("build optimizer")
    optim_cfg = cfg.optimizer
    match optim_cfg.name:
        case "adam":
            optimizer = optim.Adam(params=params, lr=optim_cfg.lr)
        case "adamw":
            optimizer = optim.AdamW(
                params=params,
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
            )

    torch_lr_scheduler = ExponentialLR(
        optimizer=optimizer, gamma=cfg.optimizer.scheduler.gamma
    )

    logger.info("build learning rate scheduler with warmup")
    lr_scheduler = create_lr_scheduler_with_warmup(
        torch_lr_scheduler,
        warmup_start_value=optim_cfg.scheduler.start_lr,
        warmup_end_value=optim_cfg.lr,
        warmup_duration=optim_cfg.scheduler.warmup_duration,
    )
    return optimizer, lr_scheduler
