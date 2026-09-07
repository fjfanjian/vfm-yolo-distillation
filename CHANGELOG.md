# Changelog

## Unreleased

### Added
- Add target-focused DINOv3 objectness auxiliary modes for small-GT weighted soft targets and peak-only ignore-aware targets.
- Add seed42 screening configs, target construction tests, and a versioned VisDrone YOLO26n experiment report with target-focus results.
- Add center-focused and background-contrast objectness targets with multi-seed experiment configurations.
- Add Grounding DINO, YOLO-World, LocateAnything, pseudo-label quality, and small-object diagnostic workflows.
- Add YOLO26/P2 model configurations, experiment runners, coverage tests, and research reports.

### Changed
- Improve relation-distillation checkpoint handling by preserving and restoring forward hooks.
- Update small-tile iteration gate wording to cover target-focused experiment branches.

### Fixed
- Extend Ruff configuration for the new scripts, reusable modules, and tests.
