import csv
import logging
import os

import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from ..utils.checkpoints import save_checkpoint
from .eval_epoch import calc_pseudo_accuracy, eval_epoch
from .forward_wrappers import forward_coco, forward_hnc


def train_epoch(
    model,
    model_inital,
    dataset_train,
    objective_fn,
    optimizer,
    lr_scheduler,
    device,
    epoch,
    args,
    cfg,
    print_every=50,
    save_every=1000,
    grad_clip=False,
    dataloaders=None,
    lambda_l2=1.0,
):
    """
    Trains the model for one epoch using the provided training dataset and configuration.

        model (torch.nn.Module): The model to be trained.
        model_inital (torch.nn.Module): The initial (untrained) model, used for L2 regularization.
        dataset_train (Iterable): The training dataset or dataloader.
        objective_fn (Callable): The objective/loss function to optimize.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device or str): Device to run training on ('cpu' or 'cuda').
        epoch (int): Current epoch number.
        args (Namespace): Additional arguments, typically from argument parser.
        cfg (Namespace): Configuration object with training/model settings.
        print_every (int, optional): Frequency (in steps) to print training logs. Defaults to 50.
        save_every (int, optional): Frequency (in steps) to save checkpoints. Defaults to 1000.
        grad_clip (bool, optional): Whether to apply gradient clipping. Defaults to False.
        dataloaders (dict, optional): Dictionary of dataloaders for evaluation during training. Defaults to None.
        lambda_l2 (float, optional): Weight for L2 regularization penalty. Defaults to 1.0.

    Returns:
        None
    """
    logger = logging.getLogger("xclip")

    params_inital = list(model_inital.parameters())
    params = list(model.parameters())

    metric = MeanMetric().to(device=device)
    metric_itm = MeanMetric().to(device=device)
    metric_row_acc = MeanMetric().to(device=device)
    metric_column_acc = MeanMetric().to(device=device)

    if grad_clip:
        model_params = model.parameters()
    gradscaler = GradScaler()

    for i, batch in enumerate(dataset_train):
        if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, DistributedDataParallel
        ):
            model.module.n_train_steps += args.batch_size
            train_steps = model.module.n_train_steps
        else:
            model.n_train_steps += args.batch_size
            train_steps = model.n_train_steps

        optimizer.zero_grad()
        match cfg.training.dataset:
            case "hnc":
                output, loss, itm_labels, text_labels = forward_hnc(
                    batch, model, device=device, objective_fn=objective_fn
                )
            case "coco":
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    output, loss = forward_coco(batch, model, device=device, cfg=cfg)

            case "flickr":
                # coco forward works for flickr as well
                output, loss = forward_coco(batch, model, device=device, cfg=cfg)

        if model.itm:
            itm_acc = (output.argmax(-1) == itm_labels).int().cpu().numpy()
            metric_itm.update(itm_acc)
        if not model.itm:
            row_acc, column_acc = calc_pseudo_accuracy(output)
        if cfg.model.l2_penalty:
            loss_l2 = compute_l2_distance(params, params_inital)
            loss = loss + lambda_l2 * loss_l2

        # loss.backward()
        # optimizer.step()
        gradscaler.scale(loss).backward()
        gradscaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=2.0)
        gradscaler.step(optimizer)
        gradscaler.update()

        metric.update(loss.item())

        if not model.itm:
            metric_row_acc.update(row_acc)
            metric_column_acc.update(column_acc)

        if (i + 1) % print_every == 0:
            if model.itm:
                logger.info(
                    f"[{epoch}] | [{i+1}/{len(dataset_train)}] - Loss: {metric.compute():.3f} - Acc@ITM: {metric_itm.compute()*100:.2f}"
                )
            else:
                logger.info(
                    f"[{epoch}] | [{i+1}/{len(dataset_train)}] - Loss: {metric.compute():.3f} - Acc@Text: {metric_row_acc.compute()*100:.2f} - Acc@Image: {metric_column_acc.compute()*100:.2f}"
                )

        if (i + 1) % save_every == 0:
            save_checkpoint(
                cfg=cfg,
                args=args,
                model=model,
                optim=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
            )
        if cfg.training.eval_powers_of_two and (
            2**model.counter_powers_of_two == (i + 1)
        ):
            assert (
                dataloaders is not None
            ), "dataloaders need to be passed to the training function"
            if dataloaders.get("test") is not None:
                eval_epoch(
                    model=model,
                    model_inital=model_inital,
                    dataset_valid=dataloaders.get("test"),
                    objective_fn=objective_fn,
                    device=device,
                    args=args,
                    cfg=cfg,
                    epoch=f"2^{model.counter_powers_of_two}",
                    split="test",
                )
            loss_eval, l2_dist, row_acc, column_acc = eval_epoch(
                model=model,
                model_inital=model_inital,
                dataset_valid=dataloaders.get("dev"),
                objective_fn=objective_fn,
                device=device,
                args=args,
                cfg=cfg,
                epoch=f"2^{model.counter_powers_of_two}",
                split="eval",
            )
            if loss_eval < model.lowest_loss_eval:
                logger.info(
                    f"[{epoch}] | saving lowest loss ({loss_eval:.4f}) checkpoint"
                )
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

            log_file(
                output_dir=args.output_dir,
                filename=f"loss_norm_{cfg.model.norm_embeddings}_shift_{cfg.model.shift_embeddings}.csv",
                train_steps=train_steps,
                train_batches=int(train_steps / args.batch_size),
                counter_powers_of_two=model.counter_powers_of_two,
                log_value=loss_eval.item(),
            )
            log_file(
                output_dir=args.output_dir,
                filename=f"l2_norm_{cfg.model.norm_embeddings}_shift_{cfg.model.shift_embeddings}.csv",
                train_steps=train_steps,
                train_batches=int(train_steps / args.batch_size),
                counter_powers_of_two=model.counter_powers_of_two,
                log_value=l2_dist,
            )
            log_file(
                output_dir=args.output_dir,
                filename=f"accuracy_row_norm_{cfg.model.norm_embeddings}_shift_{cfg.model.shift_embeddings}.csv",
                train_steps=train_steps,
                train_batches=int(train_steps / args.batch_size),
                counter_powers_of_two=model.counter_powers_of_two,
                log_value=row_acc.item(),
            )
            log_file(
                output_dir=args.output_dir,
                filename=f"accuracy_column_norm_{cfg.model.norm_embeddings}_shift_{cfg.model.shift_embeddings}.csv",
                train_steps=train_steps,
                train_batches=int(train_steps / args.batch_size),
                counter_powers_of_two=model.counter_powers_of_two,
                log_value=column_acc.item(),
            )
            model.counter_powers_of_two += 1

    if model.itm:
        logger.info(
            f"[{epoch}] | [{i+1}/{len(dataset_train)}] - Loss: {metric.compute():.3f} - Acc@ITM: {metric_itm.compute()*100:.2f}"
        )
    else:
        logger.info(
            f"[{epoch}] | [{i+1}/{len(dataset_train)}] - Loss: {metric.compute():.3f} - Acc@Row: {metric_row_acc.compute()*100:.2f} - Acc@Column: {metric_column_acc.compute()*100:.2f}"
        )


def log_file(
    output_dir, filename, counter_powers_of_two, train_steps, train_batches, log_value
):
    """
    Logs training metrics to a CSV file.

    Args:
        output_dir (str): Directory to save the log file.
        filename (str): Name of the log file.
        counter_powers_of_two (int): Current power of two counter.
        train_steps (int): Number of training steps completed.
        train_batches (int): Number of training batches completed.
        log_value (float): Value to log (e.g., loss, accuracy).
    """
    with open(os.path.join(output_dir, filename), "a", newline="") as csvfile:
        spamwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        spamwriter.writerow(
            [f"2^{counter_powers_of_two}", train_steps, train_batches, log_value]
        )
    

def compute_l2_distance(model, model_initial):
    """
    Computes the total L2 (Euclidean) distance between the parameters of the current model and the initial model.

        model (Iterable[torch.Tensor]): An iterable (e.g., list) of model parameters (tensors) at the current state.
        model_initial (Iterable[torch.Tensor]): An iterable (e.g., list) of model parameters (tensors) at the initial state.

        torch.Tensor: A scalar tensor representing the sum of L2 distances between corresponding parameters of the two models.
    """
    params = model
    params_inital = model_initial
    l2_distances = []
    for i, param in enumerate(params):
        l2_dist = ((param - params_inital[i]) ** 2).sum()
        l2_distances.append(l2_dist)
    return torch.stack(l2_distances).sum()
