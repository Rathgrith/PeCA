import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from peca.utils import move_data_to_device
from peca.utils.matching import build_valid_mask_from_colors, mutual_nearest_filter


def _ensure_three_channels(images):
    if images is None:
        return images
    if images.dim() == 5:
        c = images.shape[2]
        if c == 3:
            return images
        if c > 3:
            return images[:, :, :3]
        repeat = 3 // c + (1 if 3 % c else 0)
        return images.repeat(1, 1, repeat, 1, 1)[:, :, :3]
    if images.dim() == 4:
        c = images.shape[1]
        if c == 3:
            return images
        if c > 3:
            return images[:, :3]
        repeat = 3 // c + (1 if 3 % c else 0)
        return images.repeat(1, repeat, 1, 1)[:, :3]
    if images.dim() == 3:
        c = images.shape[0]
        if c == 3:
            return images
        if c > 3:
            return images[:3]
        repeat = 3 // c + (1 if 3 % c else 0)
        return images.repeat(repeat, 1, 1)[:3]
    return images


def _ensure_batch_inputs(line_image, seg_image, seg_num):
    if line_image.dim() == 3:
        line_image = line_image.unsqueeze(0)
    if seg_image.dim() == 2:
        seg_image = seg_image.unsqueeze(0)
    if seg_num.dim() == 0:
        seg_num = seg_num.unsqueeze(0)
    return line_image, seg_image, seg_num


def _ensure_bs_inputs(line_image, seg_image, seg_num):
    line_image, seg_image, seg_num = _ensure_batch_inputs(line_image, seg_image, seg_num)
    line_image = line_image.unsqueeze(1)
    seg_image = seg_image.unsqueeze(1)
    if seg_num.dim() == 1:
        seg_num = seg_num.unsqueeze(1)
    return line_image, seg_image, seg_num


def _ensure_b1_inputs(line_images, seg_images, seg_nums):
    if line_images.dim() == 4:
        line_images = line_images.unsqueeze(1)
    if seg_images.dim() == 3:
        seg_images = seg_images.unsqueeze(1)
    if seg_nums.dim() == 1:
        seg_nums = seg_nums.unsqueeze(1)
    return line_images, seg_images, seg_nums


class RefAugmenter:
    def __init__(
        self,
        flip_p=0.5,
        vflip_p=0.0,
        rotate90_p=0.0,
        affine_p=0.0,
        affine_deg=0.0,
        affine_translate=(0.0, 0.0),
        affine_scale=(1.0, 1.0),
        fill_line=1.0,
        fill_color=1.0,
        fill_seg=0.0,
    ):
        self.flip_p = flip_p
        self.vflip_p = vflip_p
        self.rotate90_p = rotate90_p
        self.affine_p = affine_p
        self.affine_deg = float(affine_deg)
        self.affine_translate = tuple(affine_translate)
        self.affine_scale = tuple(affine_scale)
        self.fill_line = float(fill_line)
        self.fill_color = float(fill_color)
        self.fill_seg = float(fill_seg)

    def _sample_affine(self, height, width, device):
        angle = 0.0
        if self.affine_deg > 0:
            angle = (torch.rand(1, device=device).item() * 2.0 - 1.0) * self.affine_deg
        translate = (0, 0)
        if self.affine_translate and (self.affine_translate[0] > 0 or self.affine_translate[1] > 0):
            max_dx = float(self.affine_translate[0]) * width
            max_dy = float(self.affine_translate[1]) * height
            dx = int(round((torch.rand(1, device=device).item() * 2.0 - 1.0) * max_dx))
            dy = int(round((torch.rand(1, device=device).item() * 2.0 - 1.0) * max_dy))
            translate = (dx, dy)
        scale = 1.0
        if self.affine_scale and (self.affine_scale[0] != 1.0 or self.affine_scale[1] != 1.0):
            min_s, max_s = float(self.affine_scale[0]), float(self.affine_scale[1])
            if max_s < min_s:
                min_s, max_s = max_s, min_s
            scale = min_s + (max_s - min_s) * torch.rand(1, device=device).item()
        return angle, translate, scale

    def __call__(self, line_image, seg_image, color_image):
        seg_dtype = seg_image.dtype
        seg_image = seg_image.float()
        seg_squeezed = False
        if seg_image.dim() == 2:
            seg_image = seg_image.unsqueeze(0)
            seg_squeezed = True

        if torch.rand(1).item() < self.flip_p:
            line_image = TF.hflip(line_image)
            seg_image = TF.hflip(seg_image)
            if color_image is not None and color_image.numel() > 0:
                color_image = TF.hflip(color_image)
        if torch.rand(1).item() < self.vflip_p:
            line_image = TF.vflip(line_image)
            seg_image = TF.vflip(seg_image)
            if color_image is not None and color_image.numel() > 0:
                color_image = TF.vflip(color_image)
        if self.rotate90_p > 0 and torch.rand(1).item() < self.rotate90_p:
            k = int(torch.randint(1, 4, (1,)).item())
            line_image = torch.rot90(line_image, k, dims=(-2, -1))
            seg_image = torch.rot90(seg_image, k, dims=(-2, -1))
            if color_image is not None and color_image.numel() > 0:
                color_image = torch.rot90(color_image, k, dims=(-2, -1))

        if self.affine_p > 0 and torch.rand(1).item() < self.affine_p:
            height, width = int(line_image.shape[-2]), int(line_image.shape[-1])
            angle, translate, scale = self._sample_affine(height, width, line_image.device)
            line_image = TF.affine(
                line_image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=self.fill_line,
            )
            seg_image = TF.affine(
                seg_image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=self.fill_seg,
            )
            if color_image is not None and color_image.numel() > 0:
                color_image = TF.affine(
                    color_image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=self.fill_color,
                )

        seg_image = seg_image.round().to(seg_dtype)
        if seg_squeezed:
            seg_image = seg_image.squeeze(0)
        return line_image, seg_image, color_image


def _color_aggregate_with_probs(sim_logits, src_colors, tau=0.1, topk_src=0):
    if sim_logits.numel() == 0 or src_colors.numel() == 0:
        return None, None, None

    unique_colors, src_color_ids = torch.unique(src_colors, dim=0, return_inverse=True)
    if topk_src and sim_logits.shape[1] > topk_src:
        topk_vals, topk_idx = torch.topk(sim_logits, k=topk_src, dim=1)
        pruned = torch.full_like(sim_logits, float("-inf"))
        pruned.scatter_(1, topk_idx, topk_vals)
        sim_logits = pruned

    if tau is None or tau <= 0:
        probs = torch.zeros_like(sim_logits)
        top_idx = torch.argmax(sim_logits, dim=1, keepdim=True)
        probs.scatter_(1, top_idx, 1.0)
    else:
        probs = torch.softmax(sim_logits / tau, dim=1)
    color_probs = torch.zeros(sim_logits.shape[0], unique_colors.shape[0], device=sim_logits.device)
    color_probs.index_add_(1, src_color_ids, probs)
    pred_color_ids = torch.argmax(color_probs, dim=1)

    rep_indices = torch.zeros(unique_colors.shape[0], dtype=torch.long, device=sim_logits.device)
    for k in range(unique_colors.shape[0]):
        rep_indices[k] = torch.nonzero(src_color_ids == k, as_tuple=False)[0]
    pred_src_indices = rep_indices[pred_color_ids]
    color_list_pred = unique_colors[pred_color_ids]
    return pred_src_indices, color_list_pred, color_probs


def _predict_colors(
    seg_sim_map,
    ref_colors,
    use_color_agg=False,
    tau=0.1,
    topk_src=0,
    apply_mutual_nn_filter_when_hard=True,
):
    valid_mask_src = build_valid_mask_from_colors(ref_colors)
    seg_sim_map = seg_sim_map[:, valid_mask_src]
    ref_colors = ref_colors[valid_mask_src]

    if use_color_agg:
        pred_src_indices, color_list_pred, color_probs = _color_aggregate_with_probs(
            seg_sim_map, ref_colors, tau=tau, topk_src=topk_src
        )
        return pred_src_indices, color_list_pred, color_probs

    if apply_mutual_nn_filter_when_hard:
        seg_sim_map, _ = mutual_nearest_filter(seg_sim_map)
    pred_src_indices = torch.argmax(seg_sim_map, dim=-1)
    color_list_pred = ref_colors[pred_src_indices]
    return pred_src_indices, color_list_pred, None


def _normalize_probs(probs, eps=1e-8):
    if probs is None or probs.numel() == 0:
        return probs
    denom = probs.sum(dim=1, keepdim=True).clamp_min(eps)
    return probs / denom


def _align_probs_to_palette(src_probs, src_palette, tgt_palette, tol=1e-5):
    if src_probs is None:
        return None
    if src_palette is None or tgt_palette is None:
        return src_probs
    if src_palette.shape[0] == tgt_palette.shape[0] and torch.allclose(
        src_palette, tgt_palette, atol=tol, rtol=0.0
    ):
        return src_probs

    diffs = (tgt_palette[:, None, :] - src_palette[None, :, :]).abs()
    match = diffs.max(dim=-1).values <= tol
    has_match = match.any(dim=1)
    idx = torch.argmax(match.float(), dim=1)
    aligned = src_probs[:, idx]
    if not has_match.all():
        aligned[:, ~has_match] = 0.0
    return aligned


def _ttc_step(
    curr_probs,
    curr_feats,
    prev_probs,
    prev_feats,
    curr_palette,
    prev_palette,
    gamma=0.3,
    eps=1e-8,
    use_cycle_consistency=True,
    palette_tol=1e-5,
):
    if (
        curr_probs is None
        or prev_probs is None
        or curr_feats is None
        or prev_feats is None
        or curr_probs.numel() == 0
        or prev_probs.numel() == 0
        or curr_feats.numel() == 0
        or prev_feats.numel() == 0
    ):
        return curr_probs

    curr_probs = _normalize_probs(curr_probs, eps=eps)
    prev_probs = _normalize_probs(prev_probs, eps=eps)

    curr_feats = F.normalize(curr_feats.float(), dim=1)
    prev_feats = F.normalize(prev_feats.float(), dim=1)
    sim = curr_feats @ prev_feats.t()

    prev_idx = torch.argmax(sim, dim=1)
    if use_cycle_consistency:
        back_idx = torch.argmax(sim, dim=0)
        stable = back_idx[prev_idx] == torch.arange(curr_feats.shape[0], device=curr_feats.device)
    else:
        stable = torch.ones(curr_feats.shape[0], dtype=torch.bool, device=curr_feats.device)

    prior = prev_probs[prev_idx]
    prior = _align_probs_to_palette(prior, prev_palette, curr_palette, tol=palette_tol)
    prior = prior.clamp_min(eps)
    if gamma != 1.0:
        prior = prior**gamma
    fused = _normalize_probs(curr_probs * prior, eps=eps)

    if stable.all():
        return fused
    updated = curr_probs.clone()
    updated[stable] = fused[stable]
    return updated


def _ttc_calibrate_clip(
    probs_list,
    feats_list,
    palettes_list,
    gamma=0.3,
    num_sweeps=1,
    bidirectional=True,
    use_cycle_consistency=True,
    eps=1e-8,
    palette_tol=1e-5,
):
    if probs_list is None or len(probs_list) < 2:
        return probs_list

    out = [p.clone() if p is not None else None for p in probs_list]
    steps = max(int(num_sweeps), 1)
    for _ in range(steps):
        for t in range(1, len(out)):
            out[t] = _ttc_step(
                out[t],
                feats_list[t],
                out[t - 1],
                feats_list[t - 1],
                palettes_list[t],
                palettes_list[t - 1],
                gamma=gamma,
                eps=eps,
                use_cycle_consistency=use_cycle_consistency,
                palette_tol=palette_tol,
            )
        if bidirectional:
            for t in range(len(out) - 2, -1, -1):
                out[t] = _ttc_step(
                    out[t],
                    feats_list[t],
                    out[t + 1],
                    feats_list[t + 1],
                    palettes_list[t],
                    palettes_list[t + 1],
                    gamma=gamma,
                    eps=eps,
                    use_cycle_consistency=use_cycle_consistency,
                    palette_tol=palette_tol,
                )
    return out


def _seg_cos_sim(seg_feats_src, seg_feats_tgt):
    seg_feats_src = F.normalize(seg_feats_src, p=2, dim=-1)
    seg_feats_tgt = F.normalize(seg_feats_tgt, p=2, dim=-1)

    B, S_src, L_src, C = seg_feats_src.shape
    seg_feats_src = seg_feats_src.view(B, S_src * L_src, C)

    B, S_tgt, L_tgt, C = seg_feats_tgt.shape
    seg_feats_tgt = seg_feats_tgt.view(B, S_tgt * L_tgt, C)

    return torch.matmul(seg_feats_tgt, seg_feats_src.transpose(-1, -2))


def _compute_dino_seg_feats(model, line_images, seg_images, seg_nums):
    line_images = _ensure_three_channels(line_images)
    dino_feats_map = model.get_dino_feats_map(line_images)
    return model.get_segment_feats(dino_feats_map, seg_images, seg_nums)


def _compute_timm_seg_feats(model, line_images, seg_images, seg_nums):
    line_images = _ensure_three_channels(line_images)
    timm_feats_map = model.get_timm_feats_map(line_images)
    return model.get_segment_feats(timm_feats_map, seg_images, seg_nums)


def _compute_sam2_seg_feats(model, line_images, seg_images, seg_nums):
    line_images = _ensure_three_channels(line_images)
    sam2_feats_map = model.get_sam2_feats_map(line_images)
    return model.get_segment_feats(sam2_feats_map, seg_images, seg_nums)


def _compute_sd_seg_feats(model, line_images, seg_images, seg_nums):
    line_images = _ensure_three_channels(line_images)
    sd_feats_map = model.get_sd_feats_map(line_images)
    return model.get_segment_feats(sd_feats_map, seg_images, seg_nums)


def _compute_match_seg_feats_batch(model, feature_source, line_images, seg_images, seg_nums):
    feature_source = str(feature_source or "dacon").lower()
    if feature_source == "dacon":
        line_images, seg_images, seg_nums = _ensure_b1_inputs(line_images, seg_images, seg_nums)
        seg_feats, _ = model._process_multi(line_images, seg_images, seg_nums)
        seg_feats = seg_feats.squeeze(1)
        return seg_feats
    if feature_source == "dino":
        line_images, seg_images, seg_nums = _ensure_b1_inputs(line_images, seg_images, seg_nums)
        seg_feats = _compute_dino_seg_feats(model, line_images, seg_images, seg_nums)
        return seg_feats.squeeze(1) if seg_feats.dim() == 4 else seg_feats
    if feature_source == "timm":
        line_images, seg_images, seg_nums = _ensure_b1_inputs(line_images, seg_images, seg_nums)
        seg_feats = _compute_timm_seg_feats(model, line_images, seg_images, seg_nums)
        return seg_feats.squeeze(1) if seg_feats.dim() == 4 else seg_feats
    if feature_source == "sam2":
        line_images, seg_images, seg_nums = _ensure_b1_inputs(line_images, seg_images, seg_nums)
        seg_feats = _compute_sam2_seg_feats(model, line_images, seg_images, seg_nums)
        return seg_feats.squeeze(1) if seg_feats.dim() == 4 else seg_feats
    if feature_source == "sd":
        line_images, seg_images, seg_nums = _ensure_b1_inputs(line_images, seg_images, seg_nums)
        seg_feats = _compute_sd_seg_feats(model, line_images, seg_images, seg_nums)
        return seg_feats.squeeze(1) if seg_feats.dim() == 4 else seg_feats
    raise ValueError(f"Unsupported feature_source for segment-level feats: {feature_source}")


def _compute_match_seg_feats_single(model, feature_source, line_image, seg_image, seg_num):
    feature_source = str(feature_source or "dacon").lower()
    if feature_source == "dacon":
        line_image, seg_image, seg_num = _ensure_batch_inputs(line_image, seg_image, seg_num)
        seg_feats, _ = model._process_single(line_image, seg_image, seg_num)
        return seg_feats[0] if seg_feats.dim() == 3 else seg_feats
    if feature_source == "dino":
        line_images, seg_images, seg_nums = _ensure_bs_inputs(line_image, seg_image, seg_num)
        seg_feats = _compute_dino_seg_feats(model, line_images, seg_images, seg_nums)
        if seg_feats.dim() == 4:
            seg_feats = seg_feats.squeeze(1)
        return seg_feats[0] if seg_feats.dim() == 3 else seg_feats
    if feature_source == "timm":
        line_images, seg_images, seg_nums = _ensure_bs_inputs(line_image, seg_image, seg_num)
        seg_feats = _compute_timm_seg_feats(model, line_images, seg_images, seg_nums)
        if seg_feats.dim() == 4:
            seg_feats = seg_feats.squeeze(1)
        return seg_feats[0] if seg_feats.dim() == 3 else seg_feats
    if feature_source == "sam2":
        line_images, seg_images, seg_nums = _ensure_bs_inputs(line_image, seg_image, seg_num)
        seg_feats = _compute_sam2_seg_feats(model, line_images, seg_images, seg_nums)
        if seg_feats.dim() == 4:
            seg_feats = seg_feats.squeeze(1)
        return seg_feats[0] if seg_feats.dim() == 3 else seg_feats
    if feature_source == "sd":
        line_images, seg_images, seg_nums = _ensure_bs_inputs(line_image, seg_image, seg_num)
        seg_feats = _compute_sd_seg_feats(model, line_images, seg_images, seg_nums)
        if seg_feats.dim() == 4:
            seg_feats = seg_feats.squeeze(1)
        return seg_feats[0] if seg_feats.dim() == 3 else seg_feats
    raise ValueError(f"Unsupported feature_source: {feature_source}")


def _compute_max_sim(target_feats, ref_feats):
    if target_feats is None or ref_feats is None:
        return None
    if target_feats.numel() == 0 or ref_feats.numel() == 0:
        return None
    tgt = F.normalize(target_feats.float(), dim=1)
    ref = F.normalize(ref_feats.float(), dim=1)
    sim = tgt @ ref.t()
    return sim.max(dim=1).values


def _score_from_max_sim(max_sim):
    if max_sim is None or max_sim.numel() == 0:
        return None
    return max_sim.mean()


def _build_target_feature_pool(
    model,
    device,
    tgt_loader,
    feature_source,
    max_frames=0,
):
    selected_indices = None
    if max_frames and max_frames > 0:
        total = None
        try:
            total = len(tgt_loader)
        except Exception:
            total = None
        if total is not None and total > max_frames:
            selected = np.linspace(0, total - 1, num=max_frames, dtype=int)
            selected_indices = set(int(i) for i in selected.tolist())
    target_feats = []
    with torch.no_grad():
        for idx, data in enumerate(tgt_loader):
            if selected_indices is not None:
                if idx not in selected_indices:
                    continue
            elif max_frames and idx >= max_frames:
                break
            data = move_data_to_device(data, device)
            seg_feats = _compute_match_seg_feats_single(
                model,
                feature_source,
                data["line_image"],
                data["seg_image"],
                data["seg_num"],
            )
            seg_colors = data["seg_colors"][0]
            valid_mask = build_valid_mask_from_colors(seg_colors)
            if valid_mask.sum().item() == 0:
                continue
            target_feats.append(seg_feats[valid_mask])
    if not target_feats:
        return None
    return torch.cat(target_feats, dim=0)


def _select_active_ref_views(
    model,
    feature_source,
    base_line,
    base_seg,
    base_color,
    seg_num_src,
    ref_augmenter,
    num_views,
    candidate_multiplier,
    target_feats,
    batch_size=0,
):
    compute_device = seg_num_src.device
    base_seg_feats = _compute_match_seg_feats_single(
        model,
        feature_source,
        base_line,
        base_seg,
        seg_num_src,
    )
    selected = [
        {
            "line": base_line,
            "seg": base_seg,
            "color": base_color,
            "feats": base_seg_feats,
        }
    ]

    if ref_augmenter is None or num_views <= 0 or target_feats is None or target_feats.numel() == 0:
        return selected, {"score": None, "selected": 0, "candidates": 0}

    curr_max = _compute_max_sim(target_feats, base_seg_feats)
    curr_score = _score_from_max_sim(curr_max)
    if curr_score is None:
        return selected, {"score": None, "selected": 0, "candidates": 0}

    multiplier = max(1, int(candidate_multiplier))
    cand_count = max(0, int(num_views) * multiplier)
    candidates = []
    for _ in range(cand_count):
        line_aug, seg_aug, color_aug = ref_augmenter(base_line, base_seg, base_color)
        # Keep candidate tensors on CPU to avoid large GPU memory spikes when
        # candidate pool is big (e.g. high-res frames + large multiplier).
        if isinstance(line_aug, torch.Tensor) and line_aug.is_cuda:
            line_aug = line_aug.cpu()
        if isinstance(seg_aug, torch.Tensor) and seg_aug.is_cuda:
            seg_aug = seg_aug.cpu()
        if isinstance(color_aug, torch.Tensor) and color_aug.is_cuda:
            color_aug = color_aug.cpu()
        candidates.append({"line": line_aug, "seg": seg_aug, "color": color_aug})

    if candidates:
        tgt_norm = F.normalize(target_feats.float(), dim=1)
        seg_num_val = seg_num_src.view(-1)[0]
        batch_size = int(batch_size) if batch_size is not None else 0
        if batch_size <= 0:
            batch_size = len(candidates)
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            line_batch = torch.stack([c["line"] for c in batch], dim=0)
            seg_batch = torch.stack([c["seg"] for c in batch], dim=0)
            seg_nums = torch.full(
                (len(batch),),
                seg_num_val,
                device=compute_device,
                dtype=seg_num_src.dtype,
            )
            if line_batch.device != compute_device:
                line_batch = line_batch.to(compute_device, non_blocking=True)
            if seg_batch.device != compute_device:
                seg_batch = seg_batch.to(compute_device, non_blocking=True)

            cand_seg_feats = _compute_match_seg_feats_batch(
                model,
                feature_source,
                line_batch,
                seg_batch,
                seg_nums,
            )
            cand_norm = F.normalize(cand_seg_feats.float(), dim=2)
            sim = torch.einsum("nc,blc->bnl", tgt_norm, cand_norm)
            max_sim = sim.max(dim=2).values
            for i, cand in enumerate(batch):
                cand.update(
                    {
                        "feats": cand_seg_feats[i],
                        "max_sim": max_sim[i],
                    }
                )

    candidates = [c for c in candidates if "max_sim" in c]
    selected_count = 0
    while selected_count < num_views and candidates:
        best_idx = None
        best_score = curr_score
        best_max = curr_max
        for idx, cand in enumerate(candidates):
            new_max = torch.maximum(curr_max, cand["max_sim"])
            score = _score_from_max_sim(new_max)
            if score is None:
                continue
            if best_idx is None or score > best_score:
                best_idx = idx
                best_score = score
                best_max = new_max
        if best_idx is None:
            break
        chosen = candidates.pop(best_idx)
        selected.append(
            {
                "line": chosen["line"],
                "seg": chosen["seg"],
                "color": chosen["color"],
                "feats": chosen["feats"],
            }
        )
        curr_max = best_max
        curr_score = best_score
        selected_count += 1

    stats = {
        "score": float(curr_score) if curr_score is not None else None,
        "selected": selected_count,
        "candidates": cand_count,
    }
    return selected, stats
