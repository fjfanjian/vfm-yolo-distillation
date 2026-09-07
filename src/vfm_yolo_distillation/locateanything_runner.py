from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict

from vfm_yolo_distillation.locateanything_model_loading import load_locateanything_model
from vfm_yolo_distillation.locateanything_quality import (
    LocateAnythingParseError,
    LocateAnythingParseStats,
    LocateAnythingRawResult,
    class_agnostic_boxes,
    class_agnostic_candidates,
    parse_raw_results,
)
from vfm_yolo_distillation.pseudo_label_quality import (
    PseudoCandidate,
    ThresholdPolicy,
    audit_candidates,
    load_yolo_boxes,
    read_image_source,
    write_candidates_csv,
    write_quality_summary,
)


class Phase(StrEnum):
    ALL = "all"
    INFER = "infer"
    AUDIT = "audit"


class LocateAnythingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    device: str
    generation_mode: str
    max_new_tokens: int
    max_image_side: int | None = None
    temperature: float
    top_p: float
    prompt_modes: dict[str, str]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_data: Path
    labeled_data: Path
    dataset_root_override: Path | None = None
    output_dir: Path
    raw_output_jsonl: Path
    parsed_candidates_csv: Path
    report_dir: Path
    locateanything: LocateAnythingConfig


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    root: Path
    train_source: Path
    names: tuple[str, ...]


def read_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Config must be a YAML mapping: {path}"
        raise LocateAnythingParseError(message)
    return ExperimentConfig.model_validate(payload)


def load_layout(data_yaml: Path, dataset_root: Path | None) -> DatasetLayout:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Dataset config must be a YAML mapping: {data_yaml}"
        raise LocateAnythingParseError(message)
    root = dataset_root if dataset_root is not None else Path(str(payload["path"]))
    names = tuple(str(name) for name in payload["names"])
    return DatasetLayout(root=root, train_source=root / str(payload["train"]), names=names)


def selected_dataset_root(config: ExperimentConfig, override_text: str | None) -> Path | None:
    if override_text is not None:
        return Path(override_text)
    return config.dataset_root_override


def selected_prompt_modes(config: ExperimentConfig, prompt_mode: str | None) -> tuple[str, ...]:
    if prompt_mode is None:
        return tuple(config.locateanything.prompt_modes)
    if prompt_mode not in config.locateanything.prompt_modes:
        message = f"Unknown prompt mode: {prompt_mode}"
        raise LocateAnythingParseError(message)
    return (prompt_mode,)


def selected_images(layout: DatasetLayout, limit: int | None) -> tuple[Path, ...]:
    images = read_image_source(layout.train_source, layout.root)
    if limit is None:
        return images
    return images[:limit]


def run_infer(
    config: ExperimentConfig,
    dataset_root: Path | None,
    limit: int | None,
    prompt_mode: str | None,
    rebuild: bool,
) -> None:
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    layout = load_layout(config.labeled_data, dataset_root)
    prompt_modes = selected_prompt_modes(config, prompt_mode)
    images = selected_images(layout, limit)
    completed_keys: set[tuple[str, str]] = set()
    if config.raw_output_jsonl.exists() and not rebuild:
        completed_keys = read_completed_raw_keys(config.raw_output_jsonl)
        expected = {(str(image), mode) for image in images for mode in prompt_modes}
        if expected.issubset(completed_keys):
            print(f"raw_output_jsonl=complete path={config.raw_output_jsonl}", flush=True)
            return
    config.raw_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.locateanything.model_id,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    processor = AutoProcessor.from_pretrained(
        config.locateanything.model_id,
        trust_remote_code=True,
        fix_mistral_regex=True,
        use_fast=False,
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_locateanything_model(config.locateanything.model_id, dtype)
    model = model.to(config.locateanything.device).eval()
    write_inference_outputs(
        config,
        model,
        processor,
        tokenizer,
        prompt_modes,
        images,
        completed_keys,
    )


def write_inference_outputs(
    config: ExperimentConfig,
    model,
    processor,
    tokenizer,
    prompt_modes: tuple[str, ...],
    images: tuple[Path, ...],
    completed_keys: set[tuple[str, str]],
) -> None:
    count = len(completed_keys)
    output_mode = "a" if completed_keys else "w"
    with config.raw_output_jsonl.open(output_mode, encoding="utf-8") as file:
        for image_path in images:
            with Image.open(image_path) as raw_image:
                image = resize_image_for_locateanything(
                    raw_image.convert("RGB"),
                    config.locateanything.max_image_side,
                )
            for mode in prompt_modes:
                key = (str(image_path), mode)
                if key in completed_keys:
                    continue
                query = config.locateanything.prompt_modes[mode]
                error = None
                try:
                    answer = infer_one(config, model, processor, tokenizer, image, query)
                except RuntimeError as exc:
                    if not is_cuda_oom_error(exc):
                        raise
                    clear_cuda_cache()
                    answer = "no objects found."
                    error = exc.__class__.__name__
                    print(
                        f"locateanything_oom_image={image_path.name} prompt_mode={mode}",
                        flush=True,
                    )
                finally:
                    clear_cuda_cache()
                payload = {
                    "image": str(image_path),
                    "prompt_mode": mode,
                    "query": query,
                    "answer": answer,
                }
                if error is not None:
                    payload["error"] = error
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                file.flush()
                count += 1
                print(f"locateanything_processed={count}", flush=True)


def read_completed_raw_keys(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            image = payload.get("image") or payload.get("image_path")
            prompt_mode = payload.get("prompt_mode")
            if isinstance(image, str) and isinstance(prompt_mode, str):
                completed.add((image, prompt_mode))
    return completed


def resize_image_for_locateanything(image: Image.Image, max_side: int | None) -> Image.Image:
    if max_side is None:
        return image
    width, height = image.size
    current_max_side = max(width, height)
    if current_max_side <= max_side:
        return image
    scale = max_side / float(current_max_side)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


def is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or exc.__class__.__name__ == "OutOfMemoryError"


def clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_one(
    config: ExperimentConfig,
    model,
    processor,
    tokenizer,
    image: Image.Image,
    query: str,
) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": query}],
        }
    ]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(
        config.locateanything.device
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    response = model.generate(
        pixel_values=inputs["pixel_values"].to(dtype),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=tokenizer,
        max_new_tokens=config.locateanything.max_new_tokens,
        use_cache=True,
        generation_mode=config.locateanything.generation_mode,
        temperature=config.locateanything.temperature,
        do_sample=True,
        top_p=config.locateanything.top_p,
        repetition_penalty=1.1,
        verbose=False,
    )
    if isinstance(response, tuple):
        return str(response[0])
    return str(response)


def run_audit(config: ExperimentConfig, dataset_root: Path | None) -> None:
    layout = load_layout(config.labeled_data, dataset_root)
    records = read_raw_results(config.raw_output_jsonl)
    train_images = read_image_source(layout.train_source, layout.root)
    ground_truths = tuple(box for image in train_images for box in load_yolo_boxes(image))
    config.report_dir.mkdir(parents=True, exist_ok=True)
    for mode in sorted({record.prompt_mode or "unknown" for record in records}):
        mode_records = tuple(
            record for record in records if (record.prompt_mode or "unknown") == mode
        )
        candidates, stats = parse_raw_results(mode_records, layout.names)
        write_candidates_csv(
            config.report_dir / f"locateanything_{mode}_candidates.csv",
            candidates,
        )
        write_summary(config.report_dir / f"locateanything_{mode}_parse_summary.json", stats)
        write_mode_quality(config, mode, candidates, ground_truths)


def read_raw_results(path: Path) -> tuple[LocateAnythingRawResult, ...]:
    with path.open("r", encoding="utf-8") as file:
        return tuple(
            LocateAnythingRawResult.model_validate_json(line) for line in file if line.strip()
        )


def write_summary(path: Path, stats: LocateAnythingParseStats) -> None:
    path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def write_mode_quality(
    config: ExperimentConfig,
    mode: str,
    candidates: tuple[PseudoCandidate, ...],
    ground_truths,
) -> None:
    policy = ThresholdPolicy.uniform(0.0)
    class_aware = audit_candidates(
        f"locateanything_{mode}_class_aware",
        candidates,
        ground_truths,
        policy,
        0.5,
    )
    class_agnostic = audit_candidates(
        f"locateanything_{mode}_class_agnostic",
        class_agnostic_candidates(candidates),
        class_agnostic_boxes(ground_truths),
        policy,
        0.5,
    )
    write_quality_summary(
        config.report_dir / f"locateanything_{mode}_class_aware_quality_summary.json",
        class_aware,
    )
    write_quality_summary(
        config.report_dir / f"locateanything_{mode}_class_agnostic_quality_summary.json",
        class_agnostic,
    )
    print_quality(mode, candidates, class_aware.overall.fn_rate, class_aware.overall.fp_rate)


def print_quality(
    mode: str,
    candidates: tuple[PseudoCandidate, ...],
    fn_rate: float,
    fp_rate: float,
) -> None:
    print(
        f"locateanything_audit mode={mode} candidates={len(candidates)} "
        f"class_aware_fn={fn_rate:.4f} class_aware_fp={fp_rate:.4f}",
        flush=True,
    )
