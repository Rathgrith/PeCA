"""Internal launcher used by the public run.sh entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .config import (
    BACKBONES,
    COMPONENT_CHOICES,
    ConfigError,
    checkpoint_path,
    dump_yaml,
    load_config,
    runner_name,
    to_runtime_config,
    validate_config,
)
from .data.check import DatasetError, check_dataset


def _component_overrides(args: argparse.Namespace) -> dict[str, str]:
    return {
        component: value
        for component in COMPONENT_CHOICES
        if (value := getattr(args, component, None)) is not None
    }


def resolve_run_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config, components=_component_overrides(args))
    if args.dataset_root:
        config["dataset"]["root"] = args.dataset_root
    if args.output:
        config["output"]["dir"] = args.output
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.checkpoint:
        if config["backbone"]["name"] != "dacon_v1_1":
            raise ConfigError("--checkpoint is only used with --backbone dacon_v1_1")
        config["backbone"]["checkpoint"] = args.checkpoint
    if args.wandb:
        config.setdefault("wandb", {})["enable"] = True
    if args.save_images is not None:
        config["output"]["save_images"] = args.save_images
    if args.save_json is not None:
        config["output"]["save_json"] = args.save_json
    return validate_config(config)


def _runner_command(config: dict, runtime_path: Path) -> list[str]:
    runtime = to_runtime_config(config)
    runner = runner_name(config)
    command = [sys.executable, "-m", f"peca.runners.{runner}", "--config", str(runtime_path)]
    checkpoint = checkpoint_path(config)
    if checkpoint:
        command.extend(["--model", checkpoint])
    if runner == "same_video":
        command.extend(
            [
                "--data-root",
                str(runtime["datasets"]["test"]["root"]),
                "--out-dir",
                str(runtime["inference"]["save_path"]),
                "--clip-interval",
                str(runtime["datasets"]["clip_interval"]),
            ]
        )
    return command


def run(args: argparse.Namespace) -> int:
    config = resolve_run_config(args)
    runtime = to_runtime_config(config)
    output_dir = Path(str(runtime["inference"]["save_path"]))

    with tempfile.TemporaryDirectory(prefix="peca-") as temporary:
        runtime_path = Path(temporary) / "config.yaml"
        dump_yaml(runtime, runtime_path)
        command = _runner_command(config, runtime_path)

        if args.dry_run:
            summary = {
                "experiment": config["experiment"],
                "dataset": config["dataset"],
                "protocol": config["protocol"],
                "backbone": config["backbone"],
                "method": config["method"],
                "output": config["output"],
                "wandb": config.get("wandb", {}),
            }
            print(yaml.safe_dump(summary, sort_keys=False))
            return 0

        if not args.skip_data_check:
            report = check_dataset(config)
            print(f"[data] {report['dataset']}: {report['sequences']} sequences, {report['frames']} frames")

        checkpoint = checkpoint_path(config)
        if checkpoint and not Path(checkpoint).is_file():
            raise ConfigError(
                f"DACoN checkpoint not found: {checkpoint}. Download it from "
                "https://github.com/kzmngt/DACoN#test and pass --checkpoint."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(config, output_dir / "resolved_config.yaml")
        return int(subprocess.run(command, check=False).returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bash run.sh",
        description="Run PeCA with a reusable dataset/protocol/backbone configuration.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", choices=sorted(BACKBONES))
    parser.add_argument("--dataset", choices=COMPONENT_CHOICES["dataset"])
    parser.add_argument("--protocol", choices=COMPONENT_CHOICES["protocol"])
    parser.add_argument("--method", choices=COMPONENT_CHOICES["method"])
    parser.add_argument("--dataset-root")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-data-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ConfigError, DatasetError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
