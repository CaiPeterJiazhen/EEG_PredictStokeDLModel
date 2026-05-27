# Remote FC/wPLI Segment SSL Result Export

This directory contains a small, Git-safe export of FC/wPLI segment-level SSL experiment results generated on the remote Windows workstation.

## Scope

- Experiment family: segment-level SSL transfer
- Segment SSL feature kind: `fc-wpli`
- Supervised feature kind: `psd-fc-wpli`
- Data scope: `all-patient`
- Historical unlabeled pretraining: `true`
- Transfer mode: `finetune`
- Pretrain epochs: `20`
- Supervised epochs: `100`
- Projection dimension: `32`
- Feature mask probability: `0.03`
- Noise std: `0.02`
- Lambda latent: `1.0`
- Lambda local: `0.1`

## Objectives And Seeds

The completed objectives are:

- `barlow`
- `vicreg`

The completed individual seeds are:

- `0`
- `1`
- `2`
- `3`
- `7`
- `13`

Both objectives also include `seedensemble6` prediction and metric CSV outputs. An intermediate interrupted `seedensemble5` output was intentionally excluded so this export represents the completed six-seed run.

## Contents

- `metrics/`: FC/wPLI segment SSL metric CSVs, seed summaries, subject error summaries, and transfer summary.
- `predictions/`: FC/wPLI segment SSL LOSO prediction CSVs.
- `ssl/`: FC/wPLI segment SSL pretraining history CSVs.
- `training_logs/`: FC/wPLI segment SSL supervised loss histories and small run logs.

The export intentionally does not include raw result directories, feature caches, checkpoints, model weights, or raw EEG data.

Excluded source locations and patterns include:

- `data/features`
- `data/processed`
- `results`
- `checkpoints`
- `*.pt`
- `*.pth`
- `*.ckpt`
- raw EEG `*.set`
- raw EEG `*.fdt`
