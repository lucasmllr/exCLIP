from typing import Union

import torch
from torch import Tensor


def interpolate_reference_embedding(e: Tensor, interpolations: Union[int, Tensor]):
    """Linearly interpolates N steps between the instance at index=1 and the reference at index=2.
    The batch must consist only of instance and reference."""
    if e.shape[0] != 2:
        breakpoint()
    assert e.shape[0] == 2
    d = len(e.shape)
    device = e.device
    dtype = e.dtype
    # NOTE: interpolations with half precision are not accurate...
    if isinstance(interpolations, int):
        assert interpolations > 0
        s = 1 / interpolations
        a = torch.arange(1, 0, -s, dtype=dtype).to(device)
    elif isinstance(interpolations, Tensor):
        assert len(interpolations.shape) == 1
        a = interpolations.clone().type(dtype).to(device)
    else:
        raise TypeError(
            f"interpolations must be int or tensor, got {type(interpolations)}"
        )
    x, r = e[0].unsqueeze(0), e[-1].unsqueeze(0)
    for _ in range(d - 1):  # same dims as e
        a.unsqueeze_(1)
    g = r + a * (x - r)
    g = torch.cat([g, r])
    return g


def interpolation_hook(interpolations: Union[int, Tensor], cache: list):
    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], interpolations)
        cache.append(g)
        return (g,) + inpt[1:]

    return hook


def transformer_interpolation_hook(interpolations: Union[int, Tensor], cache: list):
    def hook(model, inpt):
        e = inpt[0]
        # batch shape: from (S, B, D) to (B, S, D)
        e = e.permute(1, 0, 2)
        g = interpolate_reference_embedding(e, interpolations)
        cache.append(g)
        # back to batch shape (S, B, D)
        g = g.permute(1, 0, 2)
        return (g,) + inpt[1:]

    return hook


def saving_hook(cache: list):
    def hook(model, inpt):
        cache.append(inpt)
        return inpt

    return hook
