import logging

import torch


def get_objective_fn(cfg, device):
    """
    Constructs and returns a dictionary of objective (loss) functions based on the provided configuration.
    cfg (dict): Configuration dictionary containing training parameters, including the type of objective function to use.
    device (torch.device or str): The device on which the loss functions should be allocated (e.g., 'cpu' or 'cuda').
    dict: A dictionary mapping loss function names (e.g., 'text_loss', 'itm_loss', 'qa_loss') to their corresponding
            torch.nn loss function instances allocated on the specified device.
            torch.nn loss function instances allocated on the specified device.
    """
    logger = logging.getLogger("exclip")
    logger.info("build objective/loss functions")
    match cfg.get("training").get("objective_fn"):
        case "cross_entropy":
            objective_fn = torch.nn.CrossEntropyLoss().to(device=device)

    objective_fn = {
        "text_loss": torch.nn.CrossEntropyLoss().to(device=device),
        "itm_loss": torch.nn.CrossEntropyLoss().to(device=device),
        "qa_loss": torch.nn.CrossEntropyLoss().to(device=device),
    }
    return objective_fn
