import argparse
import copy
import logging
import os
import resource

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel

from exclip.datasets.build import build_datasets
from exclip.models.build import build_model
from exclip.training.train import train
from exclip.utils.arg_parser import get_argparser
from exclip.utils.checkpoints import load_checkpoint
from exclip.utils.config import get_config
from exclip.utils.distributed import setup_distributed
from exclip.utils.log_config import configure_logging
from exclip.utils.objective import get_objective_fn
from exclip.utils.optimizer import get_optimizer

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.multiprocessing.set_sharing_strategy("file_system")


@record
def main(args: argparse.ArgumentParser) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(args.output_dir)

    logger = logging.getLogger("exclip")
    logger.info(args)

    logger.info(f"torch version: {torch.__version__}")

    if args.eval_only:
        logger.info("script is starting in validation only mode")

    local_rank = None
    if args.distributed:
        local_rank = setup_distributed()

    if args.distributed:
        args.device = torch.device(f"cuda:{local_rank}")
        logger.info(f"local rank {args.device}")
    cfg = get_config(args=args)
    logger.info(cfg)

    # initialize the model based on the config
    model = build_model(cfg=cfg, device=args.device)

    model_inital = copy.deepcopy(model)
    # for param in model_inital.parameters():
    #     param.requires_grad = False

    model_inital.to(args.device)
    model.to(args.device)
    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True,
        )
        dist.barrier()
        torch.cuda.synchronize()
        logger.info(f"distributed barrier and synchronized")
    logger.info(f"model type: {type(model)}")

    # initialize optimizer and corresponding learning rate scheduler
    optimizer, lr_scheduler = get_optimizer(cfg=cfg, params=model.parameters())

    if args.eval_only or args.load_weights:
        load_checkpoint(
            args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        model.to(device=args.device)

    # initialize loss function
    objective_fn = get_objective_fn(cfg=cfg, device=args.device)

    # initialize dataset and dataloaders
    data_loaders = build_datasets(cfg=cfg, args=args)

    train(
        cfg=cfg,
        args=args,
        model=model,
        model_inital=model_inital,
        objective_fn=objective_fn,
        dataloaders=data_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=args.device,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("XCLIP", parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
