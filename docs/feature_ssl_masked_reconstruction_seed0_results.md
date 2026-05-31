# Masked Reconstruction Feature SSL Seed0 Pilot

Date: 2026-05-30

## Rationale

This pilot tested a negative-sample-free SSL objective inspired by masked autoencoder and denoising autoencoder EEG literature. AnySearch found directly relevant reconstruction-based EEG SSL work:

- MAEEG: Masked Auto-encoder for EEG Representation Learning: https://ar5iv.labs.arxiv.org/html/2211.02625
- GMAEEG: Self-Supervised Graph Masked Autoencoder for EEG: https://pubmed.ncbi.nlm.nih.gov/39146173/
- General self-supervised representation learning survey: https://arxiv.org/pdf/2110.09327

The implementation keeps the supervised CNN framework and the original PSD/WPLI inputs unchanged. The SSL pretext task masks standardized PSD/WPLI features, extracts CNN branch embeddings, and trains a lightweight decoder to reconstruct masked inputs. No negative pairs are used.

## Implementation

Added `masked-reconstruction` to feature-level SSL:

- `src/eeg_recovery/training/train_feature_ssl.py`
- CLI support through `scripts/07_train_feature_ssl_transfer.py`
- tests in `tests/test_feature_ssl_transfer.py`

The history CSV records `reconstruction_loss`.

## Seed0 Results

All runs use original baseline rerun features in `runs/baseline_rerun_20260529`.

| Model | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN seed0 | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.6667 | 0.6528 | 0.2760 |
| original Barlow SSL-CNN seed0 | 0.7368 | 0.7222 | 1.0000 | 0.4444 | 0.7444 | 0.6912 | 0.2007 |
| masked reconstruction, mask 0.25 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7778 | 0.8237 | 0.2254 |
| masked reconstruction, mask 0.75 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.7889 | 0.8549 | 0.2329 |

Post-hoc monitored hard negatives:

| Model | sub09 score | sub09 pred | sub14 score | sub14 pred |
|---|---:|---:|---:|---:|
| masked reconstruction, mask 0.25 | 0.9890 | 1 | 0.9722 | 1 |
| masked reconstruction, mask 0.75 | 0.8368 | 1 | 0.8956 | 1 |

## Decision

This is a mixed/negative pilot. Masked reconstruction improves ROC AUC and PR AUC over original Barlow seed0, but it loses accuracy, balanced accuracy, and Brier score. It also does not solve the repeated high-confidence false positives for sub09/sub14.

Do not run 10 seeds for this objective yet. If revisited, it should use a smaller reconstruction decoder, reconstruction loss only on modality-local masked regions, or combine reconstruction with a weak Barlow/VICReg invariance term. The current pure reconstruction objective is not strong enough as the next default SSL-CNN candidate.
