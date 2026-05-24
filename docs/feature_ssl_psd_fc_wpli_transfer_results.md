# Feature-Space SSL Transfer Results For PSD+FC-wPLI CNN

This document records the feature-space self-supervised pretraining experiment
that transfers into the established `psd-fc-wpli` gated CNN.

## Model

```text
architecture: multimodal
feature_kind: psd-fc-wpli
fusion: gated
encoder_kind: cnn
embedding_dim: 32
dropout: 0.0
supervised seed: 2
```

The supervised model is the same gated CNN documented in
`docs/best_psd_fc_wpli_gated_cnn_model.md`.

## SSL Transfer Design

The earlier raw-EEG SSL model cannot be directly loaded into a PSD/FC feature
CNN. This experiment therefore adds a feature-space SSL stage using the same
PSD and wPLI tensors that the supervised CNN consumes.

For each LOSO fold:

1. Select the SSL pool for the requested data scope.
2. Exclude the current LOSO test patient from any patient-containing SSL pool.
3. Pretrain the CNN branch encoders and gated EO/EC fusion layers with a
   feature-level NT-Xent contrastive objective.
4. Initialize the supervised `psd-fc-wpli` gated CNN from the pretrained branch
   weights.
5. Fine-tune with the original supervised LOSO training protocol.

Healthy EEG has no affected hand. For feature computation, healthy records use
the no-flip convention equivalent to affected hand `右`.

## Training Command

```powershell
python -B scripts\07_train_feature_ssl_transfer.py --data-scopes supervised-baseline all-patient-baseline all-patient all-patient-health --device cuda --pretrain-epochs 50 --supervised-epochs 100 --patience 100 --embedding-dim 32 --projection-dim 16 --pretrain-batch-size 8 --pretrain-lr 0.001 --supervised-lr 0.001 --dropout 0.0 --seed 2
```

CUDA device used:

```text
NVIDIA TITAN Xp COLLECTORS EDITION
```

## Data Pools

The fold-local SSL pair counts were:

```text
supervised-baseline:  18 EO/EC pairs per fold
all-patient-baseline: 27 EO/EC pairs per fold
all-patient:          88 EO/EC pairs per fold
all-patient-health:   101 EO/EC pairs per fold
```

## Results

```text
scope                 accuracy   balanced_acc   sensitivity   specificity   roc_auc   pr_auc
supervised-baseline   0.7895     0.7833         0.9000        0.6667        0.7667    0.7983
all-patient-baseline  0.7368     0.7333         0.8000        0.6667        0.6889    0.6470
all-patient           0.8421     0.8333         1.0000        0.6667        0.7000    0.6710
all-patient-health    0.7368     0.7333         0.8000        0.6667        0.7444    0.6892
```

The best accuracy in this SSL-transfer run came from the `all-patient` SSL
pool: `0.8421` accuracy and `0.8333` balanced accuracy.

## Output Files

Summary:

```text
results/metrics/feature_ssl_psd_fc_wpli_transfer_summary.csv
```

Per-scope outputs follow this naming pattern:

```text
results/ssl/feature_ssl_pretraining_history_feature_ssl_<scope>_psd-fc-wpli_gated_cnn_seed2_pre50_sup100.csv
results/predictions/dl_loso_predictions_feature_ssl_<scope>_psd-fc-wpli_gated_cnn_seed2_pre50_sup100.csv
results/metrics/dl_model_comparison_feature_ssl_<scope>_psd-fc-wpli_gated_cnn_seed2_pre50_sup100.csv
results/training_logs/dl_loss_history_feature_ssl_<scope>_psd-fc-wpli_gated_cnn_seed2_pre50_sup100.csv
results/figures/dl_loss_curve_feature_ssl_<scope>_psd-fc-wpli_gated_cnn_seed2_pre50_sup100.png
```

## Interpretation

The `all-patient` pool matched the previous best single-seed accuracy for the
same gated CNN configuration. The result is still exploratory because the
supervised cohort has only 19 patients and seed/model choices have already been
explored. Task 12 should add confidence intervals and permutation testing before
formal reporting.

## Follow-Up Tuning

Additional SSL-transfer experiments tested:

```text
multi-seed finetune ensemble
freeze encoder vs full finetune
SSL temperature scan
feature augmentation strength scan
```

Final tuning summary:

```text
results/metrics/feature_ssl_psd_fc_wpli_transfer_tuning_summary.csv
```

Key rows:

```text
experiment                         seed             accuracy   balanced_acc   roc_auc   pr_auc
no SSL single-seed reference       2                0.8421     0.8333         0.8222    0.8158
no SSL 10-seed ensemble reference  ensemble_10      0.7895     0.7833         0.7778    0.7894
SSL finetune pre50                 0                0.5263     0.5167         0.6889    0.7377
SSL finetune pre50                 1                0.6842     0.6778         0.7667    0.8060
SSL finetune pre50                 2                0.8421     0.8333         0.8000    0.7828
SSL finetune pre50 3-seed ensemble 0,1,2            0.6842     0.6722         0.7556    0.7642
SSL freeze encoder pre50           2                0.5263     0.5222         0.6000    0.6230
SSL temp 0.1 pre20                 2                0.6842     0.6833         0.7333    0.8040
SSL temp 0.2 pre20                 2                0.7368     0.7278         0.7111    0.6734
SSL temp 0.3 pre20                 2                0.6842     0.6778         0.6556    0.6206
SSL stronger augmentation pre20    2                0.7368     0.7333         0.7667    0.7775
```

Conclusion from this sweep:

```text
Feature-space SSL still does not produce a reliable improvement over the
no-SSL gated CNN reference. Seed 2 can match the best no-SSL accuracy, but the
multi-seed SSL ensemble is worse than the previous no-SSL 10-seed ensemble.
Freezing the pretrained encoders is harmful in this setup. Stronger feature
augmentation improves over the weaker 20-epoch SSL settings but still does not
exceed the no-SSL baseline.
```

Implementation note:

```text
scripts/07_train_feature_ssl_transfer.py now supports --seeds,
--transfer-modes, multi-value --pretrain-epochs, --temperature, --noise-std,
and --feature-mask-prob. New output names include pretraining batch size via
the bs<value> token for future runs.
```
