from pathlib import Path
from typing import Final, Literal, assert_never

import yaml
from pydantic import BaseModel, ConfigDict, Field

StageName = Literal[
    "supervised_baseline", "pseudo_label", "feature_distillation", "objectness_pretrain"
]
LabelBudget = Literal["full", "50pct", "25pct", "10pct", "5pct"]


class StudentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    family: Literal["ultralytics"]
    model: str


class DatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: Path
    label_budget: LabelBudget


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch: int = Field(gt=0)
    device: str | int
    workers: int = Field(ge=0)
    seed: int = Field(ge=0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: Path
    name: str


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    stage: StageName
    description: str
    student: StudentConfig
    dataset: DatasetRef
    training: TrainingConfig
    outputs: OutputConfig
    metrics: tuple[str, ...]


ROOT: Final = Path(__file__).resolve().parents[2]


class ConfigLoadError(RuntimeError):
    pass


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read config: {path}") from exc
    if not isinstance(loaded, dict):
        raise ConfigLoadError(f"Config must be a YAML mapping: {path}")
    return loaded


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = _read_yaml_mapping(path)
    try:
        return ExperimentConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigLoadError(f"Invalid experiment config: {path}") from exc


def training_dataset_config_path(config: ExperimentConfig) -> Path:
    match config.dataset.label_budget:
        case "full":
            return config.dataset.config
        case "50pct" | "25pct" | "10pct" | "5pct" as budget:
            source = config.dataset.config
            return source.with_stem(f"{source.stem}_{budget}")
        case unreachable:
            assert_never(unreachable)


def ultralytics_train_command(config: ExperimentConfig) -> tuple[str, ...]:
    data_path = training_dataset_config_path(config).as_posix()
    return (
        "yolo",
        "detect",
        "train",
        f"model={config.student.model}",
        f"data={data_path}",
        f"imgsz={config.training.image_size}",
        f"epochs={config.training.epochs}",
        f"batch={config.training.batch}",
        f"device={config.training.device}",
        f"workers={config.training.workers}",
        f"seed={config.training.seed}",
        f"project={config.outputs.project.as_posix()}",
        f"name={config.outputs.name}",
    )
