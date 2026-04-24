# Code Structure Overview

This document groups the repository by responsibility so the codebase is easier to navigate.

## Core Entry Points

- `train.py`: training entry.
- `val.py`: validation and evaluation entry.
- `predict.py`: pseudo-label generation entry.
- `vis_pred.py`: visualization entry.

## Model and Training Logic

- `models/`: network definitions, detection backbone, YOLOX extensions, and EFM modules.
- `modules/`: higher-level task modules such as detection, tracking, data wiring, and pseudo labeling.
- `callbacks/`: visualization, detection callbacks, grad flow, and custom training hooks.
- `loggers/`: logging helpers.
- `nerv/`: local utility package dependency used by the project.

## Data and Evaluation

- `data/`: dataset pipelines, event representations, augmentations, splits, and datapipe helpers.
- `datasets/`: local dataset storage root.
- `utils/evaluation/`: Prophesee-format evaluation, metrics, IO, and visualization helpers.
- `utils/`: shared helpers such as preprocessing, bbox handling, timing, and padding.

## Configuration

- `config/`: Hydra configuration root.
- `config/model/`: model variants and detection settings.
- `config/dataset/`: dataset presets and sampling variants.
- `config/experiment/`: experiment presets grouped by dataset.

## Assets and Research Material

- `assets/`: README media and demo assets.
- `paper_assets/`: thesis or paper figures, previews, and diagrams.
- `docs/`: centralized documentation and references.

## Runtime Outputs

These directories are generated during runs and are usually not part of the source code structure itself.

- `checkpoint/`: training checkpoints.
- `outputs/`: Hydra runtime outputs.
- `tb_logs/`: TensorBoard logs.
- `validation_logs/`: validation logs.
- `vis/`: rendered videos and visual outputs.
- `__pycache__/`: Python cache files.

## Scripts

- `scripts/`: utility scripts for reporting and training helpers.

## Recommended Top-Level Mental Model

If you only want a quick way to read the project, use this order:

1. `README.md`
2. `docs/README.md`
3. `train.py`, `val.py`, `predict.py`
4. `config/`
5. `modules/` and `models/`
