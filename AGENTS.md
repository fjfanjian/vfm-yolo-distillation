Always respond in Chinese-simplified.

This repository studies VFM-assisted YOLO distillation for low-label aerial small-object detection.

Project rules:
- Keep DINOv3, Grounding DINO, and YOLO-World as training-time teachers or pseudo-label sources by default.
- Keep deployment-time student models YOLO-only unless an experiment explicitly studies teacher-in-the-loop inference.
- Prefer configuration-driven experiments under `configs/`.
- Do not commit datasets, checkpoints, experiment runs, or generated pseudo-label files.
- Use `uv` for Python dependency management.
- Keep scripts small and reproducible; put reusable logic under `src/vfm_yolo_distillation/`.
