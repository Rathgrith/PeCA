import os

import torch
from torchvision.utils import save_image


def tensor_to_uint8_image(tensor):
    """Convert a CHW tensor to an RGB uint8 array for W&B or notebooks."""
    if tensor is None:
        return None
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    tensor = tensor[:3] if tensor.shape[0] >= 3 else tensor.repeat(3, 1, 1)
    image = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu()
    return (image * 255).round().to(torch.uint8).numpy()


def colorize_target_image(color_list_pred, image_tgt, seg_image_tgt):
    """Render per-segment RGBA predictions into an image tensor."""
    image_tgt = image_tgt.permute(1, 2, 0)
    combined_image = torch.zeros_like(image_tgt, device=image_tgt.device)
    for idx, color in enumerate(color_list_pred):
        mask = (seg_image_tgt == idx + 1).unsqueeze(-1).expand(-1, -1, 4).bool()
        combined_image = torch.where(mask, color / 255, combined_image)
    black_mask = (image_tgt == torch.tensor([0, 0, 0, 1], device=image_tgt.device)).all(dim=-1)
    combined_image[black_mask] = image_tgt[black_mask]
    return combined_image


def fill_blue_line_gaps(
    image_pred,
    line_image,
    blue_thresh=0.9,
    dark_thresh=0.1,
    alpha_thresh=0.5,
):
    """Fill blue guide-line pixels from their nearest non-line prediction."""
    if line_image.dim() != 3 or line_image.shape[0] != 4:
        return image_pred

    line_rgba = line_image.permute(1, 2, 0)
    alpha = line_rgba[..., 3]
    blue_mask = (
        (alpha > alpha_thresh)
        & (line_rgba[..., 2] >= blue_thresh)
        & (line_rgba[..., 0] <= dark_thresh)
        & (line_rgba[..., 1] <= dark_thresh)
    )
    valid = alpha <= alpha_thresh
    if not torch.any(blue_mask) or not torch.any(valid):
        return image_pred

    try:
        from scipy.ndimage import distance_transform_edt

        valid_np = valid.detach().cpu().numpy()
        blue_np = blue_mask.detach().cpu().numpy()
        _, (idx_y, idx_x) = distance_transform_edt(~valid_np, return_indices=True)
        filled = image_pred.detach().cpu().numpy()
        filled[blue_np] = filled[idx_y[blue_np], idx_x[blue_np]]
        return torch.from_numpy(filled).to(image_pred.device)
    except Exception:
        return image_pred


def save_image_pred(image_pred, char_name, frame_name, save_path):
    """Save an HWC prediction under ``images/<sequence>/<frame>.png``."""
    if isinstance(frame_name, int):
        frame_name = str(frame_name).zfill(4)
    folder_path = os.path.join(save_path, "images", char_name)
    os.makedirs(folder_path, exist_ok=True)
    save_image(image_pred.permute(2, 0, 1), os.path.join(folder_path, f"{frame_name}.png"))
