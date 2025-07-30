from typing import Tuple
from PIL import Image
from torchvision.transforms.functional import center_crop, resize


def crop(img: Image, resolution: int, bbox: Tuple[int] = None):
    resized = resize(img, resolution, interpolation=Image.BICUBIC)
    f = resized.size[0] / img.size[0]
    cropped = center_crop(resized, resolution)
    sx = (cropped.size[0] - resized.size[0]) / 2
    sy = (cropped.size[1] - resized.size[1]) / 2
    if bbox is not None:
        x, y, w, h = [int(b * f) for b in bbox]
        x_min = x + sx
        x_max = x_min + w
        y_min = y + sy
        y_max = y_min + h
        x_min = max(min(x_min, resolution), 0)
        y_min = max(min(y_min, resolution), 0)
        x_max = max(min(x_max, resolution), 0)
        y_max = max(min(y_max, resolution), 0)
        w_new = x_max - x_min
        h_new = y_max - y_min
        transformed_bbox = [x_min, y_min, w_new, h_new]
        return cropped, transformed_bbox
    else:
        return cropped
