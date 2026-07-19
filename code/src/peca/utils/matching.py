import torch


def build_valid_mask_from_colors(colors: torch.Tensor, padding_value: float = -1.0) -> torch.Tensor:
    """
    Build a boolean mask that marks real segments (not padding) from a color tensor.

    Args:
        colors: tensor shaped [..., 4] where padding entries are filled with `padding_value`.
        padding_value: value used for padding (default: -1.0).

    Returns:
        Boolean mask with shape colors.shape[:-1]; True where the segment is valid.
    """
    if colors.numel() == 0:
        return torch.zeros(colors.shape[:-1], dtype=torch.bool, device=colors.device)

    padding_color = torch.tensor(
        [padding_value, padding_value, padding_value, padding_value],
        device=colors.device,
        dtype=colors.dtype,
    )
    return ~torch.all(colors == padding_color, dim=-1)


def mutual_nearest_filter(
    sim_map: torch.Tensor,
    valid_tgt_mask: torch.Tensor | None = None,
    valid_src_mask: torch.Tensor | None = None,
    fill_value: float = -1e4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep only mutual nearest-neighbor pairs in a similarity map.

    Args:
        sim_map: similarity scores of shape [T, S] or [B, T, S] (higher is better).
        valid_tgt_mask: optional boolean mask over targets [T] or [B, T].
        valid_src_mask: optional boolean mask over sources [S] or [B, S].
        fill_value: score used to suppress non-mutual entries.

    Returns:
        filtered_map: same shape as sim_map with non-mutual entries suppressed.
        mutual_mask: boolean mask of mutual pairs with same shape as sim_map.
    """
    added_batch_dim = False
    if sim_map.dim() == 2:
        sim_map = sim_map.unsqueeze(0)
        if valid_tgt_mask is not None:
            valid_tgt_mask = valid_tgt_mask.unsqueeze(0)
        if valid_src_mask is not None:
            valid_src_mask = valid_src_mask.unsqueeze(0)
        added_batch_dim = True

    B, T, S = sim_map.shape
    device = sim_map.device
    suppress_val = torch.as_tensor(fill_value, device=device, dtype=sim_map.dtype)

    masked = sim_map.clone()
    if valid_tgt_mask is not None:
        masked = masked.masked_fill(~valid_tgt_mask.unsqueeze(-1), suppress_val)
    if valid_src_mask is not None:
        masked = masked.masked_fill(~valid_src_mask.unsqueeze(-2), suppress_val)

    tgt_best = masked.argmax(dim=-1)  # [B, T]
    src_best = masked.argmax(dim=-2)  # [B, S]

    mutual_mask = torch.zeros_like(masked, dtype=torch.bool)
    tgt_idx = torch.arange(T, device=device)[None, :].expand(B, -1)
    back_tgt = src_best.gather(1, tgt_best)
    is_mutual = back_tgt == tgt_idx
    mutual_mask.scatter_(2, tgt_best.unsqueeze(-1), is_mutual.unsqueeze(-1))

    filtered = sim_map.masked_fill(~mutual_mask, suppress_val)
    row_has_match = mutual_mask.any(dim=-1, keepdim=True)
    filtered = torch.where(row_has_match, filtered, sim_map)

    if valid_tgt_mask is not None:
        filtered = filtered.masked_fill(~valid_tgt_mask.unsqueeze(-1), suppress_val)
    if valid_src_mask is not None:
        filtered = filtered.masked_fill(~valid_src_mask.unsqueeze(-2), suppress_val)

    if added_batch_dim:
        filtered = filtered.squeeze(0)
        mutual_mask = mutual_mask.squeeze(0)

    return filtered, mutual_mask
