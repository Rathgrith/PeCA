import json

import torch
import torch.nn.functional as F
from torchvision.io import read_image


def get_image(image_path):
    return read_image(image_path).float() / 255.0


def get_seg_info(seg_image_path, json_colors_path, seg_size):
    seg_image = read_image(seg_image_path)
    if seg_size is not None:
        seg_image = F.interpolate(seg_image.unsqueeze(0), size=seg_size, mode="nearest").squeeze(0)

    color_data = None
    if json_colors_path is not None:
        with open(json_colors_path, "r", encoding="utf-8") as handle:
            color_data = json.load(handle)

    _, height, width = seg_image.shape
    seg_idx_image = (seg_image[0] << 16) + (seg_image[1] << 8) + seg_image[2]
    seg_list = torch.unique(seg_idx_image[seg_idx_image != 0])
    seg_num = len(seg_list)
    seg_colors = torch.empty((seg_num, 4), dtype=torch.float32)
    seg_coordinates = torch.empty((seg_num, 4), dtype=torch.int64)
    seg_sizes = torch.empty(seg_num, dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    new_seg_image = torch.zeros_like(seg_idx_image)

    for idx, seg_idx in enumerate(seg_list):
        mask = seg_idx_image == seg_idx
        if color_data is not None:
            rgba_value = color_data.get(str(seg_idx.item()), [-1, -1, -1, -1])
            seg_colors[idx] = torch.tensor(rgba_value, dtype=torch.float32)
        seg_sizes[idx] = mask.sum()
        xs = xx[mask]
        ys = yy[mask]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        seg_coordinates[idx] = torch.tensor(
            [
                (y_min + y_max) // 2,
                (x_min + x_max) // 2,
                y_max - y_min + 1,
                x_max - x_min + 1,
            ],
            dtype=torch.int64,
        )
        new_seg_image = torch.where(mask, idx + 1, new_seg_image)

    return seg_num, seg_sizes, seg_colors, seg_coordinates, new_seg_image
