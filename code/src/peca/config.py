"""Composition, validation, and runtime translation for release configurations.

Public YAML files use the terminology from the PeCA paper and are translated
to the compact dictionary consumed by the inference runners.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

PAPER_DEFAULTS = {
    "pa_top_k": 64,
    "pa_temperature": 0.05,
    "are_num_views": 31,
    "are_candidate_multiplier": 4,
    "are_target_frames": 20,
    "ct_gamma": 1.0,
}


BACKBONES: dict[str, dict[str, Any]] = {
    "dinov2_vitl14": {
        "feature_source": "dino",
        "model_id": "facebookresearch/dinov2:dinov2_vitl14",
        "repository": "facebookresearch/dinov2",
        "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "input_size": [518, 518],
    },
    "dacon_v1_1": {
        "feature_source": "dacon",
        "model_id": "dacon_v1_1",
        "input_size": [518, 518],
    },
    "sam2_1_large": {
        "feature_source": "sam2",
        "model_id": "facebook/sam2.1-hiera-large",
        "input_size": [512, 512],
    },
    "dinov3_convnext_l": {
        "feature_source": "timm",
        "model_id": "hf-hub:timm/convnext_large.dinov3_lvd1689m",
        "input_size": [512, 512],
    },
    "siglip2_vit_b16": {
        "feature_source": "timm",
        "model_id": "hf-hub:timm/vit_base_patch16_siglip_512.v2_webli",
        "input_size": [512, 512],
    },
    "stable_diffusion_2_1": {
        "feature_source": "sd",
        "model_id": "sd2-community/stable-diffusion-2-1",
        "input_size": [768, 768],
    },
}

COMPONENT_CHOICES = {
    "backbone": tuple(BACKBONES),
    "dataset": ("anita_pirate", "pbc3d", "pbc_real"),
    "protocol": (
        "design_sheet_1shot",
        "design_sheet_5shot",
        "design_sheet_maxshot",
        "first_frame",
        "inbetween_anita_pirate",
        "inbetween_pbc3d",
    ),
    "method": ("base", "peca"),
}

COMPONENT_DIRECTORIES = {
    "backbone": "backbones",
    "dataset": "datasets",
    "protocol": "protocols",
    "method": "methods",
}


class ConfigError(ValueError):
    """Raised when a release configuration is invalid."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_parent(current: Path, value: str) -> Path:
    candidate = (current.parent / value).resolve()
    if candidate.exists():
        return candidate
    # Allows nested run configs to refer to top-level reusable components.
    for parent in current.parents:
        if parent.name == "configs":
            candidate = (parent / value).resolve()
            if candidate.exists():
                return candidate
            break
    raise ConfigError(f"Extended config does not exist: {value!r} (from {current})")


def _load_config_raw(path: str | Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ConfigError(f"Cyclic config inheritance: {chain}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")

    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(f"'extends' must be a string or list: {config_path}")

    composed: dict[str, Any] = {}
    for parent in parents:
        if not isinstance(parent, str):
            raise ConfigError(f"Invalid parent entry {parent!r}: {config_path}")
        parent_path = _resolve_parent(config_path, parent)
        composed = deep_merge(
            composed,
            _load_config_raw(parent_path, (*stack, config_path)),
        )
    composed = deep_merge(composed, raw)
    return composed


def load_config(
    path: str | Path,
    *,
    components: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Compose a config and validate only the fully resolved document.

    Component files intentionally contain partial mappings (for example, only
    ``dataset`` or ``backbone``), so validating each parent in isolation would
    reject otherwise valid experiment configs.
    """

    config_path = Path(path).expanduser().resolve()
    composed = _load_config_raw(config_path)
    config_root = next((parent for parent in config_path.parents if parent.name == "configs"), None)
    for component, name in (components or {}).items():
        if name is None:
            continue
        if component not in COMPONENT_DIRECTORIES:
            raise ConfigError(f"Unknown config component: {component}")
        if name not in COMPONENT_CHOICES[component]:
            choices = ", ".join(COMPONENT_CHOICES[component])
            raise ConfigError(f"Unknown {component} {name!r}; choose one of: {choices}")
        if config_root is None:
            raise ConfigError("Component switches require the run config to be inside configs/")
        component_path = config_root / COMPONENT_DIRECTORIES[component] / f"{name}.yaml"
        partial = _load_config_raw(component_path)
        if component not in partial:
            raise ConfigError(f"Missing {component!r} section in {component_path}")
        composed[component] = deepcopy(partial[component])
    composed.setdefault("_meta", {})["config_path"] = str(config_path)
    return validate_config(composed)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(dict(config))
    errors: list[str] = []

    dataset = cfg.get("dataset", {})
    protocol = cfg.get("protocol", {})
    backbone = cfg.get("backbone", {})
    method = cfg.get("method", {})

    if not dataset.get("name"):
        errors.append("dataset.name is required")
    if not dataset.get("root"):
        errors.append("dataset.root is required")

    protocol_name = str(protocol.get("name", "")).lower()
    if protocol_name not in {"design_sheet", "first_frame", "inbetween"}:
        errors.append("protocol.name must be design_sheet, first_frame, or inbetween")
    if protocol_name == "design_sheet":
        shots = protocol.get("ref_shots", 1)
        if not (shots in ("max", "all") or isinstance(shots, int) and shots > 0):
            errors.append("protocol.ref_shots must be a positive integer or 'max'")

    backbone_name = str(backbone.get("name", "")).lower()
    if backbone_name not in BACKBONES:
        errors.append("backbone.name must be one of: " + ", ".join(sorted(BACKBONES)))

    method_name = str(method.get("name", "")).lower()
    if method_name not in {"base", "peca"}:
        errors.append("method.name must be 'base' or 'peca'")

    if errors:
        raise ConfigError("Invalid PeCA config:\n- " + "\n- ".join(errors))

    protocol_label = protocol_name
    if protocol_name == "design_sheet":
        shots = protocol.get("ref_shots", 1)
        protocol_label = (
            "design_sheet_maxshot" if shots in ("max", "all") else f"design_sheet_{int(shots)}shot"
        )

    experiment = cfg.setdefault("experiment", {})
    experiment.setdefault(
        "name",
        f"{dataset['name']}_{protocol_label}_{backbone_name}_{method_name}",
    )
    experiment.setdefault("seed", None)
    experiment.setdefault("num_gpus", 1)

    cfg.setdefault("output", {}).setdefault("dir", f"outputs/{experiment['name']}")
    cfg["output"].setdefault("save_images", True)
    cfg["output"].setdefault("save_json", True)
    cfg.setdefault("wandb", {}).setdefault("enable", False)
    return cfg


def _method_runtime(method: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    name = str(method["name"]).lower()
    enabled = name == "peca"
    are = method.get("are", {})
    pa = method.get("pa", {})
    ct = method.get("ct", {})

    inference = {
        "infer_use_color_agg": bool(pa.get("enable", enabled)),
        "infer_tau": float(pa.get("temperature", PAPER_DEFAULTS["pa_temperature"])),
        "infer_topk_src": int(pa.get("top_k", PAPER_DEFAULTS["pa_top_k"])),
        "temporal_calibration": {
            "enable": bool(ct.get("enable", enabled)),
            "gamma": float(ct.get("gamma", PAPER_DEFAULTS["ct_gamma"])),
            "bidirectional": bool(ct.get("bidirectional", True)),
            "num_sweeps": int(ct.get("num_sweeps", 1)),
            "use_cycle_consistency": bool(ct.get("cycle_consistency", True)),
            "eps": float(ct.get("eps", 1.0e-8)),
            "palette_tol": float(ct.get("palette_tolerance", 1.0e-5)),
        },
    }
    memory = {
        "clip_group": {
            "enable": True,
            "size": 20,
        },
        "active_memory": {
            "enable": bool(are.get("enable", enabled)),
            "seed": are.get("seed"),
            "candidate_multiplier": int(
                are.get("candidate_multiplier", PAPER_DEFAULTS["are_candidate_multiplier"])
            ),
            "target_max_frames": int(are.get("target_frames", PAPER_DEFAULTS["are_target_frames"])),
            "batch_size": int(are.get("batch_size", 1)),
            "log": bool(are.get("log", True)),
        },
        "ref_aug": {
            "enable": bool(are.get("enable", enabled)),
            # B is the number of selected augmented views.  The original
            # reference is retained in addition to these B views.
            "num_views": int(are.get("num_views", PAPER_DEFAULTS["are_num_views"])),
            "split_by_ref": bool(are.get("split_across_references", True)),
            "flip_p": float(are.get("horizontal_flip_probability", 0.5)),
            "vflip_p": float(are.get("vertical_flip_probability", 0.1)),
            "rotate90_p": float(are.get("rotate90_probability", 0.2)),
            "affine_p": float(are.get("affine_probability", 1.0)),
            "affine_deg": float(are.get("affine_degrees", 30.0)),
            "affine_translate": list(are.get("affine_translate", [0.5, 0.5])),
            "affine_scale": list(are.get("affine_scale", [0.5, 2.0])),
            "fill_line": 1.0,
            "fill_color": 1.0,
            "fill_seg": 0.0,
            "log_wandb": bool(are.get("log_visualizations", False)),
            "log_mode": str(are.get("visualization_mode", "both")),
            "log_limit": int(are.get("visualization_limit", 0)),
        },
    }
    return inference, memory


def to_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = validate_config(config)
    dataset = cfg["dataset"]
    protocol = cfg["protocol"]
    backbone_cfg = cfg["backbone"]
    backbone_name = str(backbone_cfg["name"]).lower()
    backbone = deep_merge(BACKBONES[backbone_name], backbone_cfg)
    experiment = cfg["experiment"]
    output = cfg["output"]
    wandb = cfg.get("wandb", {})

    inference, memory = _method_runtime(cfg["method"])
    if memory["active_memory"]["seed"] is None:
        experiment_seed = experiment.get("seed")
        memory["active_memory"]["seed"] = None if experiment_seed is None else int(experiment_seed)
    inference.update(
        {
            "save_images": bool(output.get("save_images", True)),
            "save_json": bool(output.get("save_json", True)),
            "save_path": str(output["dir"]),
        }
    )

    feature_source = backbone["feature_source"]
    memory["feature_source"] = feature_source
    memory["timm"] = {
        "model": backbone["model_id"] if feature_source == "timm" else "",
        "pretrained": True,
        "input_size": list(backbone["input_size"]),
    }
    memory["sam2"] = {
        "model": backbone["model_id"] if feature_source == "sam2" else "facebook/sam2.1-hiera-large",
        "input_size": list(backbone["input_size"] if feature_source == "sam2" else [512, 512]),
        "amp_dtype": str(backbone.get("amp_dtype", "bf16")),
    }
    memory["sd"] = {
        "model": backbone["model_id"] if feature_source == "sd" else "sd2-community/stable-diffusion-2-1",
        "input_size": list(backbone["input_size"] if feature_source == "sd" else [768, 768]),
        "prompt": str(backbone.get("prompt", "a photo of an anime character.")),
        "timestep": int(backbone.get("timestep", 261)),
        "timestep_ratio": float(backbone.get("timestep_ratio", 0.261)),
        "up_block_index": int(backbone.get("up_block_index", 0)),
        "precision": str(backbone.get("precision", "fp16")),
    }

    protocol_name = str(protocol["name"]).lower()
    clip_length = protocol.get("clip_length", 20)
    if isinstance(clip_length, str) and clip_length.lower() in {"max", "all"}:
        clip_length = "max"
    else:
        clip_length = int(clip_length)
    if protocol_name in {"first_frame", "inbetween"}:
        memory["consecutive_ref_mode"] = "first" if protocol_name == "first_frame" else "inbetween"
        memory["clip_group"]["enable"] = False
        inference["infer_hard_use_mnn"] = False
    else:
        inference["infer_hard_use_mnn"] = True

    runtime = {
        "num_gpu": int(experiment.get("num_gpus", 1)),
        "ref_shot": protocol.get("ref_shots", 1),
        "wandb": {
            "enable": bool(wandb.get("enable", False)),
            "project": str(wandb.get("project", "peca")),
            "entity": wandb.get("entity"),
            "run_name": str(wandb.get("run_name", experiment["name"])),
            "group": wandb.get("group", protocol_name),
            "tags": list(wandb.get("tags", ["PeCA", backbone_name, protocol_name])),
            "config": {
                key: deepcopy(cfg[key])
                for key in ("experiment", "dataset", "protocol", "backbone", "method", "output")
            },
        },
        "datasets": {
            "test": {"root": str(dataset["root"])},
            "clip_interval": clip_length,
        },
        "network": {
            "feats_dim": 128,
            "dino_repository": (
                f"{backbone.get('repository', 'facebookresearch/dinov2')}:{backbone.get('revision', 'main')}"
            ),
            "dino_model_type": "dinov2_vitl14",
            "dino_input_size": list(backbone["input_size"] if feature_source == "dino" else [518, 518]),
            "unet_input_size": [512, 512],
            "unet_hidden_dim_list": [64, 128, 256, 512],
            "segment_pool_size": [512, 512],
        },
        "inference": inference,
        "memory": memory,
    }
    return runtime


def runner_name(config: Mapping[str, Any]) -> str:
    protocol = str(config["protocol"]["name"]).lower()
    return "design_sheet" if protocol == "design_sheet" else "same_video"


def checkpoint_path(config: Mapping[str, Any]) -> str | None:
    backbone = config.get("backbone", {})
    if str(backbone.get("name", "")).lower() != "dacon_v1_1":
        return None
    value = backbone.get("checkpoint", "checkpoints/dacon_v1_1.pth")
    return str(value)


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)
