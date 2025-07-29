from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Tensor


def image_attribution_interpolation(
    image: Image,
    A: Tensor,
    alpha: float = 0.5,
    color: str = "RGB",
    clip_negative_attributions: bool = False,
    range: Optional[float] = None,
    cmap=cv2.COLORMAP_JET,
):
    assert len(A.shape) == 2, f"A must be a 2d heat map but got {A.shape}. If it is still 3d containing a token span in the first dimension, use A.sum(dim=0) to reduce it to 2d."
    img = image
    img = img.convert(color)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = A.clone().cpu().numpy()
    if clip_negative_attributions:
        heatmap = np.clip(heatmap, 0, None)
    if range is None:
        range = np.max(np.abs(heatmap))
    else:
        heatmap = np.clip(heatmap, -range, range)
    heatmap /= range
    heatmap = np.uint8(heatmap * 125 + 125)
    heatmap = cv2.resize(
        heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT
    )
    heatmap = cv2.applyColorMap(heatmap, cmap)
    oupt = np.uint8((1 - alpha) * img + alpha * heatmap)
    oupt = cv2.cvtColor(oupt, cv2.COLOR_BGR2RGB)
    return oupt


def plot_bbox(img: Image, bbox: Tuple, color: str = "red", linewidth: int = 3):
    x, y, w, h = [int(b) for b in bbox]
    f = plt.figure()
    ax = f.add_subplot(111)
    box = plt.Rectangle(
        (x, y), w, h, facecolor="none", edgecolor=color, linewidth=linewidth
    )
    ax.add_patch(box)
    ax.imshow(img)
    ax.set_yticks([])
    ax.set_xticks([])
    return f


def token_attr_to_tex(A, tokens):
    assert len(A.shape) == 1
    assert A.shape[0] == len(tokens)
    heat = list((A / A.abs().sum() * 100).numpy())
    s = []
    for t, h in zip(tokens, heat):
        if h >= 0:
            c = "red"
        else:
            c = "blue"
        s.append(rf"\adjustbox{{bgcolor={c}!{int(abs(h))}}}{{\strut {t}}}")
    return " ".join(s)
