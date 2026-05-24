# Best PSD+FC-wPLI Gated CNN Model

This document records the current best CNN configuration for the `psd-fc-wpli`
feature combination. It is intended as the durable reference for model
structure, hyperparameters, parameter counts, commands, and result files.

For the stable no-SSL supervised protocol, see:

```text
docs/no_ssl_cnn_stability_results.md
```

## Model Identity

```text
model: multimodal_psd-fc-wpli_gated_cnn
architecture: multimodal
feature_kind: psd-fc-wpli
fusion: gated
encoder_kind: cnn
```

The model uses two feature branches:

```text
branch 1: PSD
branch 2: FC-wPLI
```

Each branch receives both EO and EC state features. The EO and EC tensors share
the same branch encoder weights, then a gated EO/EC fusion layer learns
sample-specific state weights.

## Inputs

```text
PSD EO input:  batch x 62 x 90
PSD EC input:  batch x 62 x 90
wPLI EO input: batch x 1891 x 6
wPLI EC input: batch x 1891 x 6
```

Fixed data facts:

```text
channels: 62
PSD bins: 90
FC edges: 1891
FC bands: 6
EO: *1.set, open eyes
EC: *2.set, closed eyes
```

## PSD CNN Encoder

Source: `src/eeg_recovery/models/encoders.py`

Input:

```text
batch x 62 x 90
```

Internal reshape:

```text
batch x 1 x 62 x 90
```

Layer sequence:

```text
Conv2d(1, 8, kernel_size=3, padding=1)
BatchNorm2d(8)
ReLU
Dropout2d(p=0.0)

Conv2d(8, 16, kernel_size=3, padding=1)
BatchNorm2d(16)
ReLU
Dropout2d(p=0.0)

Conv2d(16, 16, kernel_size=3, padding=1)
BatchNorm2d(16)
ReLU

AdaptiveAvgPool2d((1, 1))
Flatten
Linear(16, 32)
ReLU
```

Output:

```text
batch x 32
```

Trainable parameters:

```text
PSD encoder: 4,192
```

## FC-wPLI CNN Encoder

Source: `src/eeg_recovery/models/encoders.py`

Input:

```text
batch x 1891 x 6
```

Internal transpose:

```text
batch x 6 x 1891
```

Layer sequence:

```text
Conv1d(6, 16, kernel_size=5, padding=2)
BatchNorm1d(16)
ReLU
Dropout(p=0.0)

Conv1d(16, 16, kernel_size=5, padding=2)
BatchNorm1d(16)
ReLU
Dropout(p=0.0)

Conv1d(16, 16, kernel_size=5, padding=2)
BatchNorm1d(16)
ReLU

AdaptiveAvgPool1d(1)
Flatten
Linear(16, 32)
ReLU
```

Output:

```text
batch x 32
```

Trainable parameters:

```text
FC-wPLI encoder: 3,728
```

## Gated EO/EC Fusion

Source: `src/eeg_recovery/models/multimodal_model.py`

Each branch independently fuses EO and EC embeddings.

For one branch:

```text
EO embedding: 32
EC embedding: 32
concat pair: 64
gate: Linear(64, 2)
state weights: softmax(gate(pair))
branch output = w_EO * EO embedding + w_EC * EC embedding
branch output dimension: 32
```

There are two gates:

```text
PSD gate:  130 trainable parameters
wPLI gate: 130 trainable parameters
```

The gate weights are interpretable as sample-specific EO/EC state importance
inside each feature branch.

## Multimodal Classifier

After branch-level EO/EC fusion:

```text
PSD branch output:  32
wPLI branch output: 32
branch concat:      64
```

Classifier:

```text
Linear(64, 32)
ReLU
Dropout(p=0.0)
Linear(32, 1)
Sigmoid
```

Trainable parameters:

```text
classifier: 2,113
```

## Parameter Count

```text
PSD encoder:       4,192
PSD gate:            130
wPLI encoder:      3,728
wPLI gate:           130
classifier:        2,113
--------------------------------
total trainable:  10,293
```

## Training Hyperparameters

Best single-seed configuration:

```text
device: cuda
epochs: 100
early stopping patience: 100
optimizer: Adam
learning rate: 0.001
scheduler: ReduceLROnPlateau
scheduler factor: 0.5
scheduler patience: 50
loss: BCELoss
embedding_dim: 32
dropout: 0.0
seed: 2
```

The scheduler patience is derived in code as:

```text
max(1, early_stopping_patience // 2)
```

For `patience=100`, this gives scheduler patience `50`.

## Reproduction Command

Best single-seed run:

```powershell
python -B scripts\05_train_supervised_loso.py --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --device cuda --epochs 100 --patience 100 --embedding-dim 32 --dropout 0.0 --lr 0.001 --seed 2
```

The script writes default outputs:

```text
results/predictions/dl_loso_predictions.csv
results/metrics/dl_model_comparison.csv
results/training_logs/dl_loss_history_multimodal_psd_fc_wpli_gated_cnn.csv
results/figures/dl_loss_curve_multimodal_psd_fc_wpli_gated_cnn.png
```

The preserved best-accuracy outputs are:

```text
results/predictions/dl_loso_predictions_multimodal_psd_fc_wpli_gated_cnn_best_accuracy.csv
results/metrics/dl_model_comparison_multimodal_psd_fc_wpli_gated_cnn_best_accuracy.csv
results/training_logs/dl_loss_history_multimodal_psd_fc_wpli_gated_cnn_best_accuracy.csv
results/figures/dl_loss_curve_multimodal_psd_fc_wpli_gated_cnn_best_accuracy.png
```

## Best Single-Seed Result

The best single-seed result came from seed `2`.

```text
accuracy:          0.8421052631578947
balanced_accuracy: 0.8333333333333333
sensitivity:       1.0
specificity:       0.6666666666666666
precision:         0.7692307692307693
f1:                0.8695652173913043
roc_auc:           0.8222222222222223
pr_auc:            0.8157614607614607
```

This corresponds to `16/19` correct LOSO patient-level predictions.

## Seed Ensemble Result

A 10-seed probability ensemble was also evaluated using seeds:

```text
0, 1, 2, 3, 7, 13, 21, 42, 99, 123
```

Seed ensemble result:

```text
accuracy:          0.7894736842105263
balanced_accuracy: 0.7833333333333333
sensitivity:       0.9
specificity:       0.6666666666666666
precision:         0.75
f1:                0.8181818181818182
roc_auc:           0.7777777777777778
pr_auc:            0.7894336219336219
```

This corresponds to `15/19` correct LOSO patient-level predictions.

The seed ensemble outputs are:

```text
results/predictions/dl_loso_predictions_multimodal_psd_fc_wpli_gated_cnn_seedensemble_psdfcwpli_e32_d0_ep100.csv
results/metrics/dl_model_comparison_multimodal_psd_fc_wpli_gated_cnn_seedensemble_psdfcwpli_e32_d0_ep100.csv
```

## Feature-Combination Context

Focused `psd-fc-wpli` tuning showed:

```text
psd-fc-wpli concat CNN best seed accuracy: 0.6842
psd-fc-wpli gated CNN best seed accuracy:  0.8421
psd-fc-wpli gated CNN 10-seed ensemble:    0.7895
```

The main tuning summary is:

```text
results/metrics/cnn_psd_fc_wpli_tuning_summary.csv
```

## Generalization Notes

All reported training runs use patient-level LOSO evaluation. The held-out test
subject is not used for model fitting or fold-local scaling.

However, the best single seed was selected after exploratory seed scans. For
formal reporting, the 10-seed ensemble is more defensible than the single best
seed because it reduces dependence on one random initialization.

Recommended reporting language:

```text
Exploratory tuning identified psd-fc-wPLI gated CNN as the strongest PSD+FC
fusion model. The best single seed reached 0.8421 accuracy, while a 10-seed
probability ensemble reached 0.7895 accuracy under strict LOSO evaluation.
```

Avoid claiming external generalization without independent validation or nested
model-selection controls.
