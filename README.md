# Flumina for 3D Medical Image Segmentation

Flumina is a segmentation framework built around multi-scale feature states
and a shared Vision Transformer. This repository provides preprocessing,
training, and inference for **AMOS-2022**, **BraTS-2025-GLI**, **KiTS-2023**,
**LiTS**, and **MAMA-MIA**.

## Model Variants

| Model | Embedding width | Blocks | Attention heads |
| --- | ---: | ---: | ---: |
| [Flumina-S](models/Flumina/Flumina-S) | 288 | 8 | 6 |
| [Flumina-B](models/Flumina/Flumina-B) | 480 | 12 | 10 |
| [Flumina-M](models/Flumina/Flumina-M) | 792 | 12 | 12 |
| [Flumina-L](models/Flumina/Flumina-L) | 864 | 16 | 12 |

Each model directory contains its complete implementation.

[Inference model packages](models/Flumina-Inference/README.md) provide all four
sizes for six attention backends, with an integrated Triton decoder resize.

## Pre-requisites

### 1. Environment

The code has been verified with Python 3.14. Install the dependencies with:

```bash
conda create -n flumina python=3.14 -y
conda activate flumina
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation for GPU training.

### 2. Datasets

Place raw data under `raw/<dataset>/` and update the corresponding configuration
in `configs/preprocessing/`. For example:

```bash
python preprocessing/amos2022.py --config configs/preprocessing/amos2022.yaml
```

Processed data are written to `data/<dataset>/`, including image and label arrays,
metadata, and the fixed `train.txt`, `val.txt`, and `test.txt` lists. Existing
processed collections can be selected with `DATA_PATH`.
See the [preprocessing guides](docs/preprocessing) for all five datasets.

## Training Scripts

Each dataset has independent scripts for all four model sizes: 20 training
launchers under `scripts/<dataset>/` and matching configurations under
`configs/training/<dataset>/`.

```bash
GPU_IDS=0 bash scripts/AMOS-2022/train_flumina_s_amos2022.sh
```

Training uses AdamW with linear warmup and cosine decay. Checkpoints and logs
are saved under `runs/`. See the [training guide](docs/training.md) for other
model sizes, configuration overrides, and resuming a run.

## Inference

Set the checkpoint and matching configuration, then run the dataset launcher:

```bash
CHECKPOINT=checkpoints/AMOS-2022/Flumina-S.pt \
CONFIG=configs/training/AMOS-2022/Flumina-S.json \
GPU_IDS=0 \
  bash scripts/inference/AMOS-2022/infer_flumina_s_amos2022.sh
```

There are 20 inference scripts under `scripts/inference/`. Results are saved
under `results/`; evaluation supports DSC, IoU, HD95, and ASD.
See the [inference guide](docs/inference.md) for details.

## Documentation

- [Model architecture](docs/architecture.md)
- [Data layout](data/README.md)
- [Verification](docs/verification.md)

## Acknowledgements

We thank the contributors of the following open-source projects for sharing
their code and ideas:

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [mHC-lite](https://github.com/FFTYYY/mhc-lite)
- [surface-distance](https://github.com/google-deepmind/surface-distance)
