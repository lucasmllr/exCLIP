import logging

import clip
import open_clip
import torch
from dotwiz import DotWiz

from .wrapper import exCLIP, exOpenCLIP


def build_model(cfg: DotWiz, device: str):
    """
    Builds and returns a model instance based on the provided configuration.
    This function supports building either a CLIP or OpenCLIP-based model, applying additional
    wrappers and preprocessing as specified in the configuration. The model is prepared for use
    on the specified device, and relevant preprocessing transforms are attached to the model.
        cfg (DotWiz): Configuration object containing model parameters. Must include:
            - model.name (str): Model type, either "clip" or "open-clip".
            - model.base_name (str): Base model name for loading.
            - model.pretraining (str, optional): Pretrained weights identifier (for open-clip).
            - model.shift_embeddings (bool): Whether to shift embeddings.
            - model.norm_embeddings (bool): Whether to normalize embeddings.
            - model.itm (bool, optional): Whether to use image-text matching (for clip).
        device (str): Device identifier (e.g., "cpu" or "cuda") to load the model onto.
        nn.Module: The constructed model, ready for training or inference, with preprocessing
        transforms attached as the `prep` attribute.
    """
    logger = logging.getLogger("exclip")
    logger.info("build model")
    device = torch.device(device)
    match cfg.model.name:
        case "clip":
            model, prep = clip.load(cfg.model.base_name, device=torch.device("cpu"))
            model = exCLIP(
                model,
                device=device,
                shift_embeddings=cfg.model.shift_embeddings,
                norm_embeddings=cfg.model.norm_embeddings,
                itm=cfg.model.itm,
            )
            model.prep = prep
        case "open-clip":
            model_name = cfg.model.base_name
            pretrained_weights = cfg.model.pretraining
            clip_model, _, prep = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_weights
            )
            model = exOpenCLIP(
                clip_model,
                device=device,
                shift_embeddings=cfg.model.shift_embeddings,
                norm_embeddings=cfg.model.norm_embeddings,
            )
            model.scale_cos = False
            model.prep = prep

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of parameters: {n_parameters}")
    return model
