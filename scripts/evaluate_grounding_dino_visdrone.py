from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfm_yolo_distillation.grounding_dino_eval import GroundingDinoSettings, RunSettings, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--data", default="configs/datasets/visdrone.yaml")
    parser.add_argument("--name", default="grounding_dino_base_visdrone_val")
    parser.add_argument("--output-dir", default="runs/grounding_dino/reports")
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        RunSettings(
            name=args.name,
            data=Path(args.data),
            output_dir=Path(args.output_dir),
            limit=args.limit,
            grounding_dino=GroundingDinoSettings(
                model_id=args.model_id,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_det=args.max_det,
                device=args.device,
            ),
        )
    )


if __name__ == "__main__":
    main()
