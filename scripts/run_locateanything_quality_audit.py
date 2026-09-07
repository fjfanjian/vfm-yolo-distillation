#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow>=10.0",
#   "pydantic>=2.7",
#   "pyyaml>=6.0",
#   "torch>=2.4",
#   "transformers>=4.57",
# ]
# ///
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfm_yolo_distillation.locateanything_runner import (  # noqa: E402
    Phase,
    read_config,
    run_audit,
    run_infer,
    selected_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in Phase],
        default=Phase.ALL.value,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--prompt-mode")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    dataset_root = selected_dataset_root(config, args.dataset_root)
    phase = Phase(args.phase)
    match phase:
        case Phase.ALL:
            run_infer(config, dataset_root, args.limit, args.prompt_mode, bool(args.rebuild))
            run_audit(config, dataset_root)
        case Phase.INFER:
            run_infer(config, dataset_root, args.limit, args.prompt_mode, bool(args.rebuild))
        case Phase.AUDIT:
            run_audit(config, dataset_root)


if __name__ == "__main__":
    run()
