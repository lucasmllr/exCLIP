import clip
import open_clip
import torch
from torch.nn.parallel import DistributedDataParallel


def forward_hnc(
    batch: tuple, model: torch.nn.Module, device: torch.device, objective_fn=None
) -> tuple:
    """
    Performs a forward pass for the Hierarchical Neural Concept (HNC) model using a given batch of data.

    batch (tuple): A tuple containing input data, where:
        - batch[0]: A dictionary with keys "images" (list of images) and "text" (list of text strings).
        - batch[1]: Tensor of image-text matching (ITM) labels.
        - batch[2]: Tensor of text labels.
    model (torch.nn.Module): The HNC model or a wrapped model (e.g., DataParallel, DistributedDataParallel).
    device (torch.device): The device to which tensors should be moved.
    objective_fn (dict, optional): Dictionary containing objective functions, must include "itm_loss" if model.itm is True.

    tuple: A tuple containing:
        - output: The model's output for the given images and texts.
        - loss: The computed loss (ITM loss if model.itm is True, otherwise CLIP loss).
        - itm_labels (torch.Tensor): The image-text matching labels.
        - text_labels (torch.Tensor): The text labels.
    """
    itm_labels = batch[1].to(device=device)
    text_labels = batch[2].to(device=device)

    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, DistributedDataParallel
    ):
        images = torch.stack(
            [model.module.prep(img) for img in batch[0].get("images")]
        ).to(device=device)
    else:
        images = torch.stack([model.prep(img) for img in batch[0].get("images")]).to(
            device=device
        )
    texts = clip.tokenize(batch[0].get("text")).to(device=device)

    output = model(image=images, text=texts)
    if model.itm:
        assert objective_fn is not None, "no objective function passed"
        loss = objective_fn["itm_loss"](output, itm_labels)
    else:
        loss = clip_loss(output[0][:-1, :-1])

    return output, loss, itm_labels, text_labels


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss for a given set of logits.

    Args:
        logits (torch.Tensor): The input logits.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    return torch.nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the CLIP loss for a given similarity matrix.

    Args:
        similarity (torch.Tensor): The input similarity matrix.

    Returns:
        torch.Tensor: The computed CLIP loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def forward_coco(batch, model, device, cfg):
    """
    Performs a forward pass for the COCO dataset using the provided model and configuration.

    Args:
        batch (dict): A dictionary containing input data with keys:
            - "images": List of images.
            - "text": List of text strings.
        model (torch.nn.Module): The model or a wrapped model (e.g., DataParallel, DistributedDataParallel).
        device (torch.device): The device to which tensors should be moved.
        cfg (object): Configuration object with model settings (expects cfg.model.name).

    Returns:
        tuple: A tuple containing:
            - output: The model's output for the given images and texts.
            - loss (torch.Tensor): The computed CLIP loss.
    """
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, DistributedDataParallel
    ):
        images = torch.stack(
            [model.module.prep(img) for img in batch.get("images")]
        ).to(device=device)
    else:
        images = torch.stack([model.prep(img) for img in batch.get("images")]).to(
            device=device
        )
    texts = None
    if cfg.model.name == "clip":
        texts = clip.tokenize(batch.get("text"), truncate=True).to(device=device)
    elif cfg.model.name == "open-clip":
        texts = open_clip.tokenize(batch.get("text")).to(device=device)

    output = model(image=images, text=texts)
    loss = clip_loss(output[0])

    return output, loss
