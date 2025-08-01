import logging
import time

import torch

from .coco import CocoDataset, coco_collate_fn
from .flickr30k import FlickrDataset, flickr_collate_fn
from .hnc import HNCDataset, hnc_collate_fn


def build_datasets(cfg, args, demo=False):
    """
    Builds and returns dataloaders for the specified dataset.

    Depending on the configuration, this function initializes datasets and dataloaders
    for training, validation, and testing splits for one of the supported datasets:
    HNC, COCO, or Flickr30k. It also sets up appropriate samplers and collate functions,
    and supports distributed training and demo mode.

    Args:
        cfg: Configuration object containing dataset and training parameters.
        args: Arguments object with runtime options (e.g., batch size, num_workers, distributed).
        demo (bool, optional): If True, disables training dataloader creation. Defaults to False.
    """
    logger = logging.getLogger("exclip")
    logger.info("build datasets")
    match cfg.training.dataset:
        case "hnc":
            start_time = time.time()
            dataset_train = (
                HNCDataset(cfg=cfg, split="train") if not demo else None
            )
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the train dataset"
            )
            start_time = time.time()
            dataset_valid = HNCDataset(cfg=cfg, split="valid")
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the train dataset"
            )
            start_time = time.time()
            dataset_test = HNCDataset(cfg=cfg, split="test")
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the train dataset"
            )
            collate_fn = hnc_collate_fn
        case "coco":
            start_time = time.time()
            dataset_train = CocoDataset(
                root=cfg.datasets.coco.train_images,
                cpts_ann=cfg.datasets.coco.train_cpts_annotations,
                instances_ann=cfg.datasets.coco.train_instances_annotations,
            )
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the train dataset"
            )
            start_time = time.time()
            dataset_valid = CocoDataset(
                root=cfg.datasets.coco.val_images,
                cpts_ann=cfg.datasets.coco.val_cpts_annotations,
                instances_ann=cfg.datasets.coco.val_instances_annotations,
            )
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the train dataset"
            )
            dataset_test = None
            collate_fn = coco_collate_fn
        case "flickr":
            start_time = time.time()
            dataset_train = (
                FlickrDataset(dataset_dir=cfg.datasets.flickr.dir, split="train")
                if not demo
                else None
            )
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the train dataset"
            )
            start_time = time.time()
            dataset_valid = FlickrDataset(
                dataset_dir=cfg.datasets.flickr.dir, split="val"
            )
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the val dataset"
            )
            start_time = time.time()
            dataset_test = FlickrDataset(
                dataset_dir=cfg.datasets.flickr.dir, split="test"
            )
            logger.info(
                f"{(time.time() - start_time):.2f}s elapsed to load the test dataset"
            )
            collate_fn = flickr_collate_fn

    logger.info("build dataset samplers")
    if args.distributed:
        sampler_train = (
            torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
            if not demo
            else None
        )
        sampler_valid = torch.utils.data.DistributedSampler(dataset_valid, shuffle=True)
        sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=True)
    else:
        sampler_train = (
            torch.utils.data.RandomSampler(dataset_train) if not demo else None
        )
        sampler_valid = torch.utils.data.RandomSampler(dataset_valid)
        sampler_test = (
            torch.utils.data.RandomSampler(dataset_test) if dataset_test else None
        )

    batch_sampler_train = (
        torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
        if not demo
        else None
    )
    batch_sampler_valid = torch.utils.data.BatchSampler(
        sampler_valid,
        args.batch_size * args.inference_batch_size_scale,
        drop_last=False,
    )
    batch_sampler_test = (
        torch.utils.data.BatchSampler(
            sampler_test,
            args.batch_size * args.inference_batch_size_scale,
            drop_last=False,
        )
        if dataset_test
        else None
    )

    logger.info("build data loaders")
    dataloader_train = (
        (
            torch.utils.data.DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
        if not demo
        else None
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_sampler=batch_sampler_valid,
    )
    dataloader_test = (
        torch.utils.data.DataLoader(
            dataset_test,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler_test,
        )
        if dataset_test
        else None
    )

    datasets = {
        "train": dataloader_train,
        "dev": dataloader_valid,
        "test": dataloader_test,
    }
    return datasets
