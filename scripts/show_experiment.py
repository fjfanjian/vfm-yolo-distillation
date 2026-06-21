from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfm_yolo_distillation.config import load_experiment_config, ultralytics_train_command  # noqa: E402


def main(config_path: Path) -> None:
    config = load_experiment_config(config_path)
    line_continuation = " " + "\\" + "\n  "
    command = line_continuation.join(ultralytics_train_command(config))
    sys.stdout.write(f"{config.name}: {config.description}\n")
    sys.stdout.write(f"{command}\n")


def run(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("Usage: python scripts/show_experiment.py <config-path>\n")
        return 2
    main(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv))
