import logging

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from ..utils.checkpoints import save_checkpoint
from .eval_epoch import eval_epoch
from .train_epoch import train_epoch


@record
def train(
    cfg,
    args,
    model,
    model_inital,
    objective_fn,
    dataloaders,
    optimizer,
    lr_scheduler,
    device,
    grad_clip=False,
):
    """
    Trains and evaluates the model.

    Args:
        cfg: Configuration object with training parameters.
        args: Command-line arguments or configuration namespace.
        model: The model to be trained (torch.nn.Module).
        model_inital: The initial model state (torch.nn.Module).
        objective_fn: Objective function(s) used for training and evaluation.
        dataloaders: Dictionary containing 'train', 'dev', and optionally 'test' DataLoader objects.
        optimizer: Optimizer for model parameters.
        lr_scheduler: Learning rate scheduler function.
        device: Device on which to perform training (e.g., torch.device).
        grad_clip (bool, optional): If True, applies gradient clipping during training.
    Returns:
        None
    """
    logger = logging.getLogger("exclip")
    logger.info("begin training")

    lr_scheduler(None)
    if args.distributed:
        dist.barrier()
        torch.cuda.synchronize()
    if args.pre_eval or args.eval_only:
        model.eval()
        if dataloaders.get("test") is not None:
            eval_epoch(
                model=model,
                model_inital=model_inital,
                dataset_valid=dataloaders.get("test"),
                objective_fn=objective_fn,
                device=device,
                args=args,
                cfg=cfg,
                epoch=-1,
                split="test",
            )
        eval_epoch(
            model=model,
            model_inital=model_inital,
            dataset_valid=dataloaders.get("dev"),
            objective_fn=objective_fn,
            device=device,
            args=args,
            cfg=cfg,
            epoch=-1,
            split="eval",
        )

        if args.eval_only:
            logger.info("exit program")
            return
    if args.distributed:
        dist.barrier()
        torch.cuda.synchronize()
    for epoch in range(cfg.training.epochs):
        logger.info(f"[{epoch}] | start new epoch")
        logger.info(f'[{epoch}] | learning rate: {optimizer.param_groups[0]["lr"]}')
        model.train()
        train_epoch(
            model=model,
            model_inital=model_inital,
            dataset_train=dataloaders.get("train"),
            objective_fn=objective_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            args=args,
            cfg=cfg,
            epoch=epoch,
            grad_clip=grad_clip,
            dataloaders=dataloaders,
        )
        if args.distributed:
            dist.barrier()
            torch.cuda.synchronize()

        save_checkpoint(
            cfg=cfg,
            args=args,
            model=model,
            optim=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )

        if args.distributed:
            dist.barrier()
            torch.cuda.synchronize()

        model.eval()
        loss_eval, _, _, _ = eval_epoch(
            model=model,
            model_inital=model_inital,
            dataset_valid=dataloaders.get("dev"),
            objective_fn=objective_fn,
            device=device,
            args=args,
            cfg=cfg,
            epoch=epoch,
            split="eval",
        )
        if dataloaders.get("test") is not None:
            eval_epoch(
                model=model,
                model_inital=model_inital,
                dataset_valid=dataloaders.get("test"),
                objective_fn=objective_fn,
                device=device,
                args=args,
                cfg=cfg,
                epoch=epoch,
                split="test",
            )
        if loss_eval < model.lowest_loss_eval:
            logger.info(f"[{epoch}] | saving lowest loss ({loss_eval:.4f}) checkpoint")
            save_checkpoint(
                cfg=cfg,
                args=args,
                model=model,
                optim=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                lowest_loss=True,
            )
            model.lowest_loss_eval = loss_eval

        if args.distributed:
            dist.barrier()
            torch.cuda.synchronize()

        # step with the lr scheduler
        lr_scheduler(None)

    return None
