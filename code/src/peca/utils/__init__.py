from pathlib import Path

import yaml

from .data_process import (
    get_file_count,
    get_folder_names,
    list_valid_frame_indices,
    make_video_data_list,
    move_data_to_device,
    normalize_color,
    normalize_coordinate,
    save_json_pred,
)
from .image_process import get_image, get_seg_info
from .matching import build_valid_mask_from_colors, mutual_nearest_filter
from .metrics import calculate_accuracy
from .pooling import segment_pooling
from .visualization import (
    colorize_target_image,
    fill_blue_line_gaps,
    save_image_pred,
    tensor_to_uint8_image,
)


def load_runtime_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


__all__ = [
    "build_valid_mask_from_colors",
    "calculate_accuracy",
    "colorize_target_image",
    "fill_blue_line_gaps",
    "get_file_count",
    "get_folder_names",
    "get_image",
    "get_seg_info",
    "list_valid_frame_indices",
    "load_runtime_config",
    "make_video_data_list",
    "move_data_to_device",
    "mutual_nearest_filter",
    "normalize_color",
    "normalize_coordinate",
    "save_image_pred",
    "save_json_pred",
    "segment_pooling",
    "tensor_to_uint8_image",
]
