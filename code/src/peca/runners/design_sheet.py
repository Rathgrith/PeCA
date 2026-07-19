import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from peca.core import (
    _build_target_feature_pool,
    _compute_match_seg_feats_single,
    _predict_colors,
    _seg_cos_sim,
)
from peca.data import DACoNSingleDataset, dacon_single_pad_collate_fn
from peca.methods import (
    ReferenceViewAugmenter,
    fuse_sequence_probabilities,
    select_active_references,
)
from peca.models import DACoNModel
from peca.models.backbone_only import BackboneOnlyModel
from peca.utils import (
    colorize_target_image,
    fill_blue_line_gaps,
    get_file_count,
    get_folder_names,
    load_runtime_config,
    move_data_to_device,
    save_image_pred,
    save_json_pred,
    tensor_to_uint8_image,
)
from peca.utils.matching import build_valid_mask_from_colors
from peca.utils.metrics import calculate_accuracy

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt}"


def main(args):
    config = load_runtime_config(args.config)
    data_root = config["datasets"]["test"]["root"]
    apply_blue_fill = "pirate" in str(data_root).lower()
    ref_shot = config.get("ref_shot", 1)

    infer_cfg = config["inference"]
    save_images = bool(infer_cfg.get("save_images", True))
    save_json = bool(infer_cfg.get("save_json", True))
    save_path = infer_cfg.get("save_path", "outputs")
    os.makedirs(save_path, exist_ok=True)

    memory_cfg = config.get("memory", {})
    clip_cfg = memory_cfg.get("clip_group", {})
    clip_group_enable = bool(clip_cfg.get("enable", False))
    clip_group_size = int(clip_cfg.get("size", 0))
    if clip_group_enable and clip_group_size <= 0:
        raise ValueError("memory.clip_group.size must be > 0 when clip_group.enable is true")

    infer_use_color_agg = bool(infer_cfg.get("infer_use_color_agg", True))
    infer_tau = float(infer_cfg.get("infer_tau", 0.1))
    infer_topk_src = int(infer_cfg.get("infer_topk_src", 128))
    ttc_cfg = infer_cfg.get("temporal_calibration", infer_cfg.get("ttc", {}))
    ttc_enable = bool(ttc_cfg.get("enable", False))
    ttc_gamma = float(ttc_cfg.get("gamma", 0.3))
    ttc_bidirectional = bool(ttc_cfg.get("bidirectional", True))
    ttc_num_sweeps = int(ttc_cfg.get("num_sweeps", 1))
    ttc_cycle = bool(ttc_cfg.get("use_cycle_consistency", True))
    ttc_eps = float(ttc_cfg.get("eps", 1e-8))
    ttc_palette_tol = float(ttc_cfg.get("palette_tol", 1e-5))

    feature_source = str(memory_cfg.get("feature_source", "dacon")).lower()
    if feature_source in ("", "none", "null"):
        feature_source = "dacon"
    if feature_source not in ("dacon", "dino", "timm", "sam2", "sd"):
        raise ValueError(f"Unsupported memory.feature_source: {feature_source}")

    ref_aug_cfg = memory_cfg.get("ref_aug", {})
    ref_aug_enable = bool(ref_aug_cfg.get("enable", False))
    ref_aug_num_views = int(ref_aug_cfg.get("num_views", 0))
    ref_aug_flip_p = float(ref_aug_cfg.get("flip_p", 0.5))
    ref_aug_vflip_p = float(ref_aug_cfg.get("vflip_p", 0.0))
    ref_aug_rotate90_p = float(ref_aug_cfg.get("rotate90_p", 0.0))
    ref_aug_affine_p = float(ref_aug_cfg.get("affine_p", 0.0))
    ref_aug_affine_deg = float(ref_aug_cfg.get("affine_deg", 0.0))
    ref_aug_affine_translate = ref_aug_cfg.get("affine_translate", (0.0, 0.0))
    ref_aug_affine_scale = ref_aug_cfg.get("affine_scale", (1.0, 1.0))
    ref_aug_fill_line = float(ref_aug_cfg.get("fill_line", 1.0))
    ref_aug_fill_color = float(ref_aug_cfg.get("fill_color", 1.0))
    ref_aug_fill_seg = float(ref_aug_cfg.get("fill_seg", 0.0))
    ref_aug_log = bool(ref_aug_cfg.get("log_wandb", False))
    ref_aug_log_mode = str(ref_aug_cfg.get("log_mode", "both")).lower()
    ref_aug_log_limit = int(ref_aug_cfg.get("log_limit", 0))
    ref_aug_split_by_ref = bool(ref_aug_cfg.get("split_by_ref", True))
    if not isinstance(ref_aug_affine_translate, (list, tuple)) or len(ref_aug_affine_translate) < 2:
        ref_aug_affine_translate = (0.0, 0.0)
    if not isinstance(ref_aug_affine_scale, (list, tuple)) or len(ref_aug_affine_scale) < 2:
        ref_aug_affine_scale = (1.0, 1.0)
    ref_aug_affine_translate = (
        float(ref_aug_affine_translate[0]),
        float(ref_aug_affine_translate[1]),
    )
    ref_aug_affine_scale = (
        float(ref_aug_affine_scale[0]),
        float(ref_aug_affine_scale[1]),
    )

    active_cfg = memory_cfg.get("active_memory", {})
    active_enable = bool(active_cfg.get("enable", False))
    active_seed = active_cfg.get("seed", None)
    active_candidate_multiplier = max(1, int(active_cfg.get("candidate_multiplier", 1)))
    active_target_max_frames = int(active_cfg.get("target_max_frames", 20))
    if active_target_max_frames < 0:
        active_target_max_frames = 0
    active_batch_size = int(active_cfg.get("batch_size", 0))
    active_log = bool(active_cfg.get("log", True))

    network_config = dict(config["network"])
    timm_cfg = memory_cfg.get("timm", {})
    timm_model_type = timm_cfg.get("model")
    timm_pretrained = bool(timm_cfg.get("pretrained", True))
    timm_input_size = timm_cfg.get("input_size", network_config.get("dino_input_size", [518, 518]))
    sam2_cfg = memory_cfg.get("sam2", {})
    sam2_model_id = sam2_cfg.get("model") or sam2_cfg.get("model_id")
    sam2_input_size = sam2_cfg.get("input_size", [1024, 1024])
    sam2_amp_dtype = sam2_cfg.get("amp_dtype")
    sd_cfg = memory_cfg.get("sd", {})
    sd_model_id = sd_cfg.get("model", "sd2-community/stable-diffusion-2-1")
    sd_input_size = sd_cfg.get("input_size", [768, 768])
    sd_prompt = sd_cfg.get("prompt", "a photo of an anime character.")
    sd_timestep = sd_cfg.get("timestep", 261)
    sd_timestep_ratio = sd_cfg.get("timestep_ratio")
    sd_up_block_index = int(sd_cfg.get("up_block_index", 0))
    sd_precision = sd_cfg.get("precision", "fp16")
    need_dino_backbone = feature_source == "dino"
    need_timm_backbone = feature_source == "timm"
    need_sam2_backbone = feature_source == "sam2"
    need_sd_backbone = feature_source == "sd"

    if active_enable and active_seed is not None:
        seed = int(active_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and config["num_gpu"] > 0 else "cpu")
    if feature_source == "dacon":
        model = DACoNModel(network_config).to(device)
        checkpoint = torch.load(args.model, map_location=device)
        base_state = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith("dino.") for k in base_state.keys()):
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in base_state.items() if not k.startswith("dino.")}
            model_state.update(filtered_state)
            model.load_state_dict(model_state, strict=False)
            base_state = model_state
        else:
            model.load_state_dict(base_state, strict=False)
    else:
        dino_model_type = network_config.get("dino_model_type", "dinov2_vitl14")
        dino_repository = network_config.get("dino_repository", "facebookresearch/dinov2:main")
        dino_input_size = network_config.get("dino_input_size", [518, 518])
        segment_pool_size = network_config.get("segment_pool_size", [512, 512])
        if need_timm_backbone and not timm_model_type:
            raise ValueError("memory.timm.model is required when feature_source is timm.")
        if need_sam2_backbone and not sam2_model_id:
            raise ValueError("memory.sam2.model is required when feature_source is sam2.")
        model = BackboneOnlyModel(
            dino_model_type,
            dino_input_size,
            segment_pool_size,
            device,
            load_dino=need_dino_backbone,
            dino_repository=dino_repository,
            timm_model_type=timm_model_type,
            timm_input_size=timm_input_size,
            timm_pretrained=timm_pretrained,
            load_timm=need_timm_backbone,
            sam2_model_id=sam2_model_id,
            sam2_input_size=sam2_input_size,
            sam2_amp_dtype=sam2_amp_dtype,
            load_sam2=need_sam2_backbone,
            sd_model_id=sd_model_id,
            sd_input_size=sd_input_size,
            sd_prompt=sd_prompt,
            sd_timestep=sd_timestep,
            sd_timestep_ratio=sd_timestep_ratio,
            sd_up_block_index=sd_up_block_index,
            sd_precision=sd_precision,
            load_sd=need_sd_backbone,
        )
        base_state = None
    model.eval()

    wandb = None
    wandb_run = None
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enable", False)
    wandb_metric_prefix = os.getenv("WANDB_METRIC_PREFIX") or wandb_cfg.get("metric_prefix")
    if use_wandb:
        import wandb as wb

        wandb = wb
        run_name = (
            os.getenv("WANDB_RUN_NAME")
            or wandb_cfg.get("run_name")
            or f"peca-design-sheet-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_id = os.getenv("WANDB_RUN_ID") or wandb_cfg.get("run_id")
        resume = os.getenv("WANDB_RESUME") or wandb_cfg.get("resume")
        project = os.getenv("WANDB_PROJECT") or wandb_cfg.get("project", "peca")
        group = os.getenv("WANDB_GROUP") or wandb_cfg.get("group") or None
        entity = os.getenv("WANDB_ENTITY") or wandb_cfg.get("entity")
        tags_env = os.getenv("WANDB_TAGS")
        if tags_env is not None:
            tags = [t.strip() for t in tags_env.split(",") if t.strip()]
        else:
            tags = wandb_cfg.get("tags")
        init_kwargs = {
            "project": project,
            "name": run_name,
            "config": wandb_cfg.get("config", config),
            "group": group,
            "job_type": "peca_design_sheet",
        }
        if run_id:
            init_kwargs["id"] = run_id
        if resume:
            init_kwargs["resume"] = resume
        if entity:
            init_kwargs["entity"] = entity
        if tags:
            init_kwargs["tags"] = tags
        wandb_run = wandb.init(**init_kwargs)

    def _wandb_log(payload):
        if not wandb_run:
            return
        if wandb_metric_prefix:
            payload = {f"{wandb_metric_prefix}/{k}": v for k, v in payload.items()}
        wandb.log(payload)

    ttc_warned = False

    overall_metrics = {
        "seg_acc": 0.0,
        "seg_acc_thres": 0.0,
        "pix_acc": 0.0,
        "pix_fg_acc": 0.0,
        "pix_bg_miou": 0.0,
        "seg_bg_miou": 0.0,
        "count": 0,
    }

    for char_name in get_folder_names(data_root):
        if base_state is not None:
            model.load_state_dict(base_state, strict=False)
            model.eval()

        if ref_shot in ("max", "all"):
            ref_count = get_file_count(os.path.join(data_root, char_name, "ref", "gt"))
            ref_data_list = [[char_name, idx] for idx in range(ref_count)]
        else:
            ref_data_list = [[char_name, idx] for idx in range(int(ref_shot))]
        ref_count = len(ref_data_list)

        def _views_for_ref(ref_idx):
            if not ref_aug_split_by_ref or ref_aug_num_views <= 0 or ref_count <= 1:
                return ref_aug_num_views
            base = ref_aug_num_views // ref_count
            extra = ref_aug_num_views % ref_count
            return base + (1 if ref_idx < extra else 0)

        ref_dataset = DACoNSingleDataset(ref_data_list, data_root, is_ref=True, mode="val_kf")
        ref_loader = DataLoader(
            ref_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dacon_single_pad_collate_fn
        )

        all_seg_colors_ref = torch.empty(0, device=device)
        all_seg_feats_ref_match = torch.empty(0, device=device)

        ref_augmenter = None
        if ref_aug_enable and ref_aug_num_views > 0:
            ref_augmenter = ReferenceViewAugmenter(
                flip_p=ref_aug_flip_p,
                vflip_p=ref_aug_vflip_p,
                rotate90_p=ref_aug_rotate90_p,
                affine_p=ref_aug_affine_p,
                affine_deg=ref_aug_affine_deg,
                affine_translate=ref_aug_affine_translate,
                affine_scale=ref_aug_affine_scale,
                fill_line=ref_aug_fill_line,
                fill_color=ref_aug_fill_color,
                fill_seg=ref_aug_fill_seg,
            )

        ref_color_logs = []
        ref_line_logs = []
        ref_log_count = 0
        if not active_enable:
            with torch.no_grad():
                for ref_idx, ref_data in enumerate(ref_loader):
                    ref_data = move_data_to_device(ref_data, device)
                    base_line = ref_data["line_image"][0]
                    base_seg = ref_data["seg_image"][0]
                    base_color = ref_data["color_image"][0] if ref_data["color_image"].numel() > 0 else None

                    views = [(base_line, base_seg, base_color)]
                    if ref_augmenter is not None:
                        view_budget = _views_for_ref(ref_idx)
                        for _ in range(view_budget):
                            line_aug, seg_aug, color_aug = ref_augmenter(base_line, base_seg, base_color)
                            views.append((line_aug, seg_aug, color_aug))

                    for line_v, seg_v, color_v in views:
                        seg_feats_match = _compute_match_seg_feats_single(
                            model,
                            feature_source,
                            line_v,
                            seg_v,
                            ref_data["seg_num"],
                        )
                        seg_colors_ref = ref_data["seg_colors"]
                        all_seg_feats_ref_match = torch.cat((all_seg_feats_ref_match, seg_feats_match), dim=0)
                        all_seg_colors_ref = torch.cat((all_seg_colors_ref, seg_colors_ref[0]), dim=0)
                        if wandb_run and ref_aug_log:
                            if ref_aug_log_limit <= 0 or ref_log_count < ref_aug_log_limit:
                                caption = f"{ref_data['frame_name'][0]}_{ref_log_count}"
                                if ref_aug_log_mode in ("both", "color") and color_v is not None:
                                    img = tensor_to_uint8_image(color_v)
                                    if img is not None:
                                        ref_color_logs.append(wandb.Image(img, caption=caption))
                                if ref_aug_log_mode in ("both", "line"):
                                    img = tensor_to_uint8_image(line_v)
                                    if img is not None:
                                        ref_line_logs.append(wandb.Image(img, caption=caption))
                                ref_log_count += 1
            if wandb_run and ref_aug_log:
                log_payload = {}
                if ref_color_logs:
                    log_payload[f"peca/{char_name}/ref_color"] = ref_color_logs
                if ref_line_logs:
                    log_payload[f"peca/{char_name}/ref_line"] = ref_line_logs
                if log_payload:
                    _wandb_log(log_payload)

        tgt_count = get_file_count(os.path.join(data_root, char_name, "line"))
        tgt_data_list = [[char_name, idx] for idx in range(tgt_count)]
        tgt_dataset = DACoNSingleDataset(tgt_data_list, data_root, is_ref=False, mode="val_kf")
        if clip_group_enable:
            clip_indices_list = [
                list(range(start, min(start + clip_group_size, tgt_count)))
                for start in range(0, tgt_count, clip_group_size)
            ]
        else:
            clip_indices_list = [list(range(tgt_count))]

        total_metrics = {
            "seg_acc": 0.0,
            "seg_acc_thres": 0.0,
            "pix_acc": 0.0,
            "pix_fg_acc": 0.0,
            "pix_bg_miou": 0.0,
            "seg_bg_miou": 0.0,
            "count": 0,
        }

        with torch.no_grad():
            use_pbar = tqdm is not None and sys.stderr.isatty()
            for clip_idx, clip_indices in enumerate(clip_indices_list):
                if not clip_indices:
                    continue
                clip_dataset = Subset(tgt_dataset, clip_indices)
                tgt_loader = DataLoader(
                    clip_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=dacon_single_pad_collate_fn,
                )

                prefix_name = (
                    char_name
                    if not clip_group_enable
                    else f"{char_name} clip {clip_idx + 1}/{len(clip_indices_list)}"
                )

                if active_enable:
                    target_feats = _build_target_feature_pool(
                        model,
                        device,
                        tgt_loader,
                        feature_source,
                        max_frames=active_target_max_frames,
                    )
                    all_seg_colors_ref = torch.empty(0, device=device)
                    all_seg_feats_ref_match = torch.empty(0, device=device)
                    am_views = 0
                    am_refs = 0
                    for ref_idx, ref_data in enumerate(ref_loader):
                        ref_data = move_data_to_device(ref_data, device)
                        base_line = ref_data["line_image"][0]
                        base_seg = ref_data["seg_image"][0]
                        base_color = (
                            ref_data["color_image"][0] if ref_data["color_image"].numel() > 0 else None
                        )

                        view_budget = _views_for_ref(ref_idx)
                        views, _ = select_active_references(
                            model,
                            feature_source,
                            base_line,
                            base_seg,
                            base_color,
                            ref_data["seg_num"],
                            ref_augmenter,
                            view_budget,
                            active_candidate_multiplier,
                            target_feats,
                            batch_size=active_batch_size,
                        )
                        am_views += len(views)
                        am_refs += 1

                        for view in views:
                            line_v = view["line"]
                            seg_v = view["seg"]
                            color_v = view["color"]
                            seg_feats_match = view.get("feats")
                            if seg_feats_match is None:
                                seg_feats_match = _compute_match_seg_feats_single(
                                    model,
                                    feature_source,
                                    line_v,
                                    seg_v,
                                    ref_data["seg_num"],
                                )
                            seg_colors_ref = ref_data["seg_colors"]
                            all_seg_feats_ref_match = torch.cat(
                                (all_seg_feats_ref_match, seg_feats_match), dim=0
                            )
                            all_seg_colors_ref = torch.cat((all_seg_colors_ref, seg_colors_ref[0]), dim=0)
                            if wandb_run and ref_aug_log:
                                if ref_aug_log_limit <= 0 or ref_log_count < ref_aug_log_limit:
                                    caption = f"{ref_data['frame_name'][0]}_{ref_log_count}"
                                    if ref_aug_log_mode in ("both", "color") and color_v is not None:
                                        img = tensor_to_uint8_image(color_v)
                                        if img is not None:
                                            ref_color_logs.append(wandb.Image(img, caption=caption))
                                    if ref_aug_log_mode in ("both", "line"):
                                        img = tensor_to_uint8_image(line_v)
                                        if img is not None:
                                            ref_line_logs.append(wandb.Image(img, caption=caption))
                                    ref_log_count += 1

                    if active_log:
                        target_count = int(target_feats.shape[0]) if target_feats is not None else 0
                        print(
                            f"[ARE] {prefix_name}: target_feats={target_count} "
                            f"refs={am_refs} views={am_views}"
                        )

                iter_desc = (
                    char_name
                    if not clip_group_enable
                    else f"{char_name}[{clip_idx + 1}/{len(clip_indices_list)}]"
                )
                iter_loader = (
                    tqdm(
                        tgt_loader,
                        desc=iter_desc,
                        leave=False,
                        bar_format=PROGRESS_BAR_FORMAT,
                    )
                    if use_pbar
                    else tgt_loader
                )
                ttc_active = ttc_enable and infer_use_color_agg
                if ttc_enable and not infer_use_color_agg and not ttc_warned:
                    print("CT is enabled but PA is disabled; skipping CT.")
                    ttc_warned = True
                clip_cache = [] if ttc_active else None
                for data in iter_loader:
                    data = move_data_to_device(data, device)
                    seg_feats_match = _compute_match_seg_feats_single(
                        model,
                        feature_source,
                        data["line_image"],
                        data["seg_image"],
                        data["seg_num"],
                    )

                    ref_feats = all_seg_feats_ref_match
                    ref_colors = all_seg_colors_ref

                    ref_feats_batch = ref_feats.unsqueeze(0).unsqueeze(0)
                    seg_feats_tgt_batch = seg_feats_match.unsqueeze(0).unsqueeze(0)
                    seg_sim_map = _seg_cos_sim(ref_feats_batch, seg_feats_tgt_batch).squeeze(0)

                    pred_src_idx, color_list_pred, color_probs = _predict_colors(
                        seg_sim_map,
                        ref_colors,
                        use_color_agg=infer_use_color_agg,
                        tau=infer_tau,
                        topk_src=infer_topk_src,
                    )

                    seg_colors_tgt = data["seg_colors"][0]
                    seg_sizes_tgt = data["seg_sizes"][0]
                    valid_mask_tgt = build_valid_mask_from_colors(seg_colors_tgt)
                    seg_colors_tgt_valid = seg_colors_tgt[valid_mask_tgt]
                    seg_sizes_tgt_valid = seg_sizes_tgt[valid_mask_tgt]
                    ref_colors_valid = ref_colors[build_valid_mask_from_colors(ref_colors)]
                    pred_src_idx_valid = pred_src_idx[valid_mask_tgt]

                    if ttc_active:
                        palette_colors, src_color_ids = torch.unique(
                            ref_colors_valid, dim=0, return_inverse=True
                        )
                        rep_indices = torch.zeros(
                            palette_colors.shape[0], dtype=torch.long, device=palette_colors.device
                        )
                        for k in range(palette_colors.shape[0]):
                            rep_indices[k] = torch.nonzero(src_color_ids == k, as_tuple=False)[0]
                        clip_cache.append(
                            {
                                "frame_name": data["frame_name"][0],
                                "line_image": data["line_image"][0],
                                "seg_image": data["seg_image"][0],
                                "seg_feats": seg_feats_match.detach(),
                                "color_probs": color_probs.detach() if color_probs is not None else None,
                                "palette_colors": palette_colors.detach(),
                                "palette_rep_indices": rep_indices.detach(),
                                "pred_src_idx": pred_src_idx.detach(),
                                "color_list_pred": color_list_pred.detach(),
                                "seg_colors_tgt": seg_colors_tgt.detach(),
                                "seg_sizes_tgt": seg_sizes_tgt.detach(),
                                "valid_mask_tgt": valid_mask_tgt.detach(),
                                "ref_colors_valid": ref_colors_valid.detach(),
                            }
                        )
                    else:
                        if seg_colors_tgt_valid.numel() > 0:
                            metrics = calculate_accuracy(
                                pred_src_idx_valid,
                                ref_colors_valid,
                                seg_colors_tgt_valid,
                                seg_sizes_tgt_valid,
                            )
                            total_metrics["seg_acc"] += metrics[0]
                            total_metrics["seg_acc_thres"] += metrics[1]
                            total_metrics["pix_acc"] += metrics[2]
                            total_metrics["pix_fg_acc"] += metrics[3]
                            total_metrics["pix_bg_miou"] += metrics[4]
                            total_metrics["seg_bg_miou"] += metrics[5]
                            total_metrics["count"] += 1
                            if use_pbar:
                                count = total_metrics["count"]
                                iter_loader.set_postfix(
                                    seg_acc=f"{total_metrics['seg_acc'] / count:.4f}",
                                    pix_acc=f"{total_metrics['pix_acc'] / count:.4f}",
                                )

                        if save_images or save_json:
                            color_list_pred_u8 = color_list_pred * 255
                            if save_images:
                                image_pred = colorize_target_image(
                                    color_list_pred_u8, data["line_image"][0], data["seg_image"][0]
                                )
                                if apply_blue_fill:
                                    image_pred = fill_blue_line_gaps(image_pred, data["line_image"][0])
                                save_image_pred(image_pred, char_name, data["frame_name"][0], save_path)
                            if save_json:
                                save_json_pred(
                                    color_list_pred_u8, char_name, data["frame_name"][0], save_path
                                )

                if ttc_active and clip_cache:
                    probs_list = [entry["color_probs"] for entry in clip_cache]
                    feats_list = [entry["seg_feats"] for entry in clip_cache]
                    palettes_list = [entry["palette_colors"] for entry in clip_cache]
                    calibrated = fuse_sequence_probabilities(
                        probs_list,
                        feats_list,
                        palettes_list,
                        gamma=ttc_gamma,
                        num_sweeps=ttc_num_sweeps,
                        bidirectional=ttc_bidirectional,
                        cycle_consistency=ttc_cycle,
                        eps=ttc_eps,
                        palette_tolerance=ttc_palette_tol,
                    )
                    output_iter = (
                        tqdm(
                            range(len(clip_cache)),
                            desc=f"{iter_desc}-ct",
                            leave=False,
                            bar_format=PROGRESS_BAR_FORMAT,
                        )
                        if use_pbar
                        else range(len(clip_cache))
                    )
                    for idx in output_iter:
                        entry = clip_cache[idx]
                        probs = calibrated[idx] if calibrated is not None else None
                        if probs is None or probs.numel() == 0:
                            color_list_pred = entry["color_list_pred"]
                            pred_src_idx = entry["pred_src_idx"]
                        else:
                            pred_color_ids = torch.argmax(probs, dim=1)
                            color_list_pred = entry["palette_colors"][pred_color_ids]
                            pred_src_idx = entry["palette_rep_indices"][pred_color_ids]
                        valid_mask_tgt = entry["valid_mask_tgt"]
                        seg_colors_tgt_valid = entry["seg_colors_tgt"][valid_mask_tgt]
                        seg_sizes_tgt_valid = entry["seg_sizes_tgt"][valid_mask_tgt]
                        ref_colors_valid = entry["ref_colors_valid"]
                        pred_src_idx_valid = pred_src_idx[valid_mask_tgt]
                        if seg_colors_tgt_valid.numel() > 0:
                            metrics = calculate_accuracy(
                                pred_src_idx_valid,
                                ref_colors_valid,
                                seg_colors_tgt_valid,
                                seg_sizes_tgt_valid,
                            )
                            total_metrics["seg_acc"] += metrics[0]
                            total_metrics["seg_acc_thres"] += metrics[1]
                            total_metrics["pix_acc"] += metrics[2]
                            total_metrics["pix_fg_acc"] += metrics[3]
                            total_metrics["pix_bg_miou"] += metrics[4]
                            total_metrics["seg_bg_miou"] += metrics[5]
                            total_metrics["count"] += 1

                        if save_images or save_json:
                            color_list_pred_u8 = color_list_pred * 255
                            if save_images:
                                image_pred = colorize_target_image(
                                    color_list_pred_u8, entry["line_image"], entry["seg_image"]
                                )
                                if apply_blue_fill:
                                    image_pred = fill_blue_line_gaps(image_pred, entry["line_image"])
                                save_image_pred(image_pred, char_name, entry["frame_name"], save_path)
                            if save_json:
                                save_json_pred(color_list_pred_u8, char_name, entry["frame_name"], save_path)

        if active_enable and wandb_run and ref_aug_log:
            log_payload = {}
            if ref_color_logs:
                log_payload[f"peca/{char_name}/ref_color"] = ref_color_logs
            if ref_line_logs:
                log_payload[f"peca/{char_name}/ref_line"] = ref_line_logs
            if log_payload:
                _wandb_log(log_payload)

        if total_metrics["count"] > 0:
            count = total_metrics["count"]
            print(
                f"{char_name}: seg_acc {total_metrics['seg_acc'] / count:.4f}, "
                f"pix_acc {total_metrics['pix_acc'] / count:.4f}"
            )
            overall_metrics["seg_acc"] += total_metrics["seg_acc"]
            overall_metrics["seg_acc_thres"] += total_metrics["seg_acc_thres"]
            overall_metrics["pix_acc"] += total_metrics["pix_acc"]
            overall_metrics["pix_fg_acc"] += total_metrics["pix_fg_acc"]
            overall_metrics["pix_bg_miou"] += total_metrics["pix_bg_miou"]
            overall_metrics["seg_bg_miou"] += total_metrics["seg_bg_miou"]
            overall_metrics["count"] += total_metrics["count"]

            if wandb_run:
                _wandb_log(
                    {
                        f"peca/{char_name}/seg_acc": total_metrics["seg_acc"] / count,
                        f"peca/{char_name}/seg_acc_thres": total_metrics["seg_acc_thres"] / count,
                        f"peca/{char_name}/pix_acc": total_metrics["pix_acc"] / count,
                        f"peca/{char_name}/pix_fg_acc": total_metrics["pix_fg_acc"] / count,
                        f"peca/{char_name}/pix_bg_miou": total_metrics["pix_bg_miou"] / count,
                    }
                )

    print("PeCA inference complete.")
    if wandb_run:
        log_payload = {}
        if overall_metrics["count"] > 0:
            count = overall_metrics["count"]
            log_payload.update(
                {
                    "peca/overall_seg_acc": overall_metrics["seg_acc"] / count,
                    "peca/overall_seg_acc_thres": overall_metrics["seg_acc_thres"] / count,
                    "peca/overall_pix_acc": overall_metrics["pix_acc"] / count,
                    "peca/overall_pix_fg_acc": overall_metrics["pix_fg_acc"] / count,
                    "peca/overall_pix_bg_miou": overall_metrics["pix_bg_miou"] / count,
                }
            )
        if log_payload:
            _wandb_log(log_payload)
        wandb.finish()
    if overall_metrics["count"] > 0:
        count = overall_metrics["count"]
        metrics_percent = {
            "acc": 100.0 * overall_metrics["seg_acc"] / count,
            "acc_thresh": 100.0 * overall_metrics["seg_acc_thres"] / count,
            "pix_acc": 100.0 * overall_metrics["pix_acc"] / count,
            "pix_f_acc": 100.0 * overall_metrics["pix_fg_acc"] / count,
            "pix_b_miou": 100.0 * overall_metrics["pix_bg_miou"] / count,
        }
        with open(os.path.join(save_path, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "metrics_percent": metrics_percent,
                    "evaluated_frames": int(count),
                },
                handle,
                indent=2,
            )
        print(
            "\nOverall Metrics:\n"
            f"  Segment Accuracy: {overall_metrics['seg_acc'] / count:.4f} "
            f"(Threshold: {overall_metrics['seg_acc_thres'] / count:.4f})\n"
            f"  Pixel Accuracy: {overall_metrics['pix_acc'] / count:.4f} "
            f"(Foreground: {overall_metrics['pix_fg_acc'] / count:.4f})\n"
            f"  Pixel Background MIoU: {overall_metrics['pix_bg_miou'] / count:.4f}"
        )


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))

    parser = argparse.ArgumentParser(description="PeCA design-sheet inference.")
    parser.add_argument("--config", type=str, default="configs/runtime.yaml")
    parser.add_argument("--model", type=str, default="checkpoints/dacon_v1_1.pth")
    args = parser.parse_args()

    main(args)
