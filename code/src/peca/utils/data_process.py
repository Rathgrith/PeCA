import json
import os

import torch


def save_json_pred(color_list_pred, char_name, frame_name, save_path):
    if isinstance(frame_name, int):
        frame_name = str(frame_name).zfill(4)
    folder_path = os.path.join(save_path, "json", char_name)
    os.makedirs(folder_path, exist_ok=True)
    color_dict = {
        str(idx + 1): [int(value) for value in color.tolist()] for idx, color in enumerate(color_list_pred)
    }
    with open(os.path.join(folder_path, f"{frame_name}.json"), "w", encoding="utf-8") as handle:
        json.dump(color_dict, handle)


def normalize_color(color):
    return color / 255


def normalize_coordinate(coords, seg_size):
    height, width = seg_size
    factors = torch.tensor(
        [height, width, height, width],
        dtype=torch.float32,
        device=coords.device,
    )
    return coords / factors


def move_data_to_device(data, device):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device=device, dtype=torch.float32)
    return data


def get_folder_names(path):
    return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))


def get_file_count(path):
    return sum(os.path.isfile(os.path.join(path, name)) for name in os.listdir(path))


def _frame_all_exists(base_path, frame_idx_str):
    return all(
        os.path.exists(os.path.join(base_path, subdir, f"{frame_idx_str}.png"))
        for subdir in ("line", "gt", "seg")
    ) and os.path.exists(os.path.join(base_path, "seg", f"{frame_idx_str}.json"))


def list_valid_frame_indices(base_path):
    line_dir = os.path.join(base_path, "line")
    if not os.path.isdir(line_dir):
        return []
    indices = []
    for name in os.listdir(line_dir):
        if not name.lower().endswith(".png"):
            continue
        stem, _ = os.path.splitext(name)
        if not stem.isdigit():
            continue
        index = int(stem)
        if _frame_all_exists(base_path, f"{index:04d}"):
            indices.append(index)
    return sorted(set(indices))


def make_video_data_list(data_root, clip_interval, consecutive_ref_mode="first"):
    mode = str(consecutive_ref_mode or "first").lower()
    if mode not in ("first", "inbetween"):
        raise ValueError(f"Unsupported consecutive_ref_mode: {mode}")

    data_list = []
    for char_name in get_folder_names(data_root):
        valid_indices = list_valid_frame_indices(os.path.join(data_root, char_name))
        if not valid_indices:
            continue
        frame_count = len(valid_indices)
        interval = frame_count if clip_interval == "max" else int(clip_interval)
        for clip_start in range(0, frame_count, interval):
            clip_indices = valid_indices[clip_start : min(clip_start + interval, frame_count)]
            stop = max(0, len(clip_indices) - (2 if mode == "inbetween" else 1))
            for position in range(stop):
                data_list.append([char_name, [clip_indices[position]], [clip_indices[position + 1]]])
    return data_list
