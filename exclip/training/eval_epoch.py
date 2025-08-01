import logging

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from .forward_wrappers import forward_coco, forward_hnc


@torch.inference_mode()
def eval_epoch(
    model,
    model_inital,
    dataset_valid,
    objective_fn,
    device,
    epoch,
    args,
    cfg,
    split,
    print_every=50,
):
    """
    Evaluates the model for one epoch on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        model_inital (torch.nn.Module): The initial model (for L2 distance calculation).
        dataset_valid (Iterable): Validation dataset loader.
        objective_fn (callable): Objective/loss function.
        device (torch.device): Device to run evaluation on.
        epoch (int): Current epoch number.
        args (Namespace): Arguments containing configuration such as batch size.
        cfg (Namespace): Configuration object with training and dataset info.
        split (str): Dataset split to evaluate ('eval' or 'test').
        print_every (int, optional): Print logging info every N batches. Defaults to 50.

    Returns:
        tuple: (mean loss, mean L2 distance, mean row accuracy, mean column accuracy)
    """
    metric = MeanMetric().to(device=device)
    metric_itm = MeanMetric().to(device=device)
    metric_row_acc = MeanMetric().to(device=device)
    metric_column_acc = MeanMetric().to(device=device)
    logger = logging.getLogger("xclip")
    logger.info(f"[{epoch}] | begin {split} validation")

    params_inital = list(model_inital.parameters())
    params = list(model.parameters())
    l2_distances = []
    for i, param in enumerate(params):
        l2_dist = (param - params_inital[i]).pow(2).sum().sqrt().cpu().numpy()
        l2_distances.append(l2_dist)
    logger.info(f"[{epoch}] | l2 distance: {np.mean(np.array(l2_distances))}")

    for i, batch in enumerate(dataset_valid):
        if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, DistributedDataParallel
        ):
            match split:
                case "eval":
                    model.module.n_valid_steps += args.batch_size
                    valid_steps = model.module.n_valid_steps
                case "test":
                    model.module.n_test_steps += args.batch_size
                    valid_steps = model.module.n_test_steps
        else:
            match split:
                case "eval":
                    model.n_valid_steps += args.batch_size
                    valid_steps = model.n_valid_steps
                case "test":
                    model.n_test_steps += args.batch_size
                    valid_steps = model.n_test_steps

        match cfg.training.dataset:
            case "hnc":
                output, loss, itm_labels, text_labels = forward_hnc(
                    batch, model, device=device, objective_fn=objective_fn
                )
            case "coco":
                output, loss = forward_coco(batch, model, device=device, cfg=cfg)
            case "flickr":
                output, loss = forward_coco(batch, model, device=device, cfg=cfg)

        metric.update(loss.item())
        if model.itm:
            itm_acc = (output.argmax(-1) == itm_labels).int().cpu().numpy()
            metric_itm.update(itm_acc)
        if not model.itm:
            row_acc, column_acc = calc_pseudo_accuracy(output)
            metric_row_acc.update(row_acc)
            metric_column_acc.update(column_acc)

        if (i + 1) % print_every == 0:
            if model.itm:
                logger.info(
                    f"[{epoch}] | [{i+1}/{len(dataset_valid)}] - Loss: {metric.compute():.3f} - Acc@ITM: {metric_itm.compute()*100:.2f}"
                )
            else:
                logger.info(
                    f"[{epoch}] | [{i+1}/{len(dataset_valid)}] - Loss: {metric.compute():.3f} - Acc@Text: {metric_row_acc.compute()*100:.2f} - Acc@Image: {metric_column_acc.compute()*100:.2f}"
                )
    if model.itm:
        logger.info(
            f"[{epoch}] | [{i+1}/{len(dataset_valid)}] - Loss: {metric.compute():.3f} - Acc@ITM: {metric_itm.compute()*100:.2f}"
        )
    else:
        logger.info(
            f"[{epoch}] | [{i+1}/{len(dataset_valid)}] - Loss: {metric.compute():.3f} - Acc@Text: {metric_row_acc.compute()*100:.2f} - Acc@Image: {metric_column_acc.compute()*100:.2f}"
        )
    return (
        metric.compute(),
        np.mean(np.array(l2_distances)),
        metric_row_acc.compute(),
        metric_column_acc.compute(),
    )


@torch.no_grad()
def calc_pseudo_accuracy(output):
    """
    Calculates the pseudo accuracy for the model's output logits.

    Given the model's output logits for rows and columns, this function computes the predicted class indices by taking the argmax along the class dimension. It then compares these predictions to the ground truth labels, which are assumed to be the indices themselves (i.e., a range from 0 to N-1 for N samples), and returns the accuracy for both rows and columns as NumPy arrays of 0s and 1s.

        output (tuple of torch.Tensor): A tuple containing two tensors:
            - output[0]: Logits for rows, shape (N, C)
            - output[1]: Logits for columns, shape (N, C)
            where N is the number of samples and C is the number of classes.

        tuple of np.ndarray: Two arrays of shape (N,), each containing 1 for correct predictions and 0 for incorrect ones:
            - First array: Row-wise accuracy (predicted vs. ground truth)
            - Second array: Column-wise accuracy (predicted vs. ground truth)
    """
    rows = output[0].argmax(1)
    columns = output[1].argmax(1)
    labels_soft = torch.arange(len(output[0]), device=output[0].device)

    return (rows == labels_soft).int().cpu().numpy(), (
        columns == labels_soft
    ).int().cpu().numpy()
