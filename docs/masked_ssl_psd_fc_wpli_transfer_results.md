# Masked SSL Transfer Results For PSD+FC-wPLI CNN

This document records the masked modeling SSL experiment for the established
`psd-fc-wpli` gated CNN.

## Purpose

The previous contrastive SSL runs did not reliably improve the supervised
model. This experiment replaces contrastive pretraining with masked
reconstruction:

- PSD branch: mask channels, frequency bins, and sparse channel-frequency
  elements, then reconstruct only the masked PSD values.
- wPLI branch: mask graph nodes, FC edges, and frequency bands, then reconstruct
  only the masked wPLI values.

Only the PSD and wPLI CNN encoder weights are transferred into supervised
training. The SSL reconstruction decoders are discarded.

Two reconstruction heads have now been tested:

- Global embedding reconstruction: decode the full PSD/FC tensor from the
  pooled encoder embedding.
- Local pre-pooling reconstruction: decode masked values from the encoder
  feature map before adaptive pooling.

## Implementation

Code:

```text
src/eeg_recovery/training/train_masked_ssl.py
scripts/09_train_masked_ssl_transfer.py
tests/test_masked_ssl_transfer.py
tests/test_model_shapes.py
```

The transfer target remains:

```text
architecture: multimodal
feature_kind: psd-fc-wpli
fusion: gated
encoder_kind: cnn
embedding_dim: 32
dropout: 0.0
supervised seed: 2
```

For each LOSO fold:

1. Select the `all-patient` SSL pool.
2. Exclude the current LOSO test patient from the SSL pool.
3. Fit branch-specific PSD and wPLI scalers on the fold-local SSL pool.
4. Pretrain PSD and wPLI masked reconstruction autoencoders.
5. Transfer only encoder weights into the `psd-fc-wpli` gated CNN.
6. Fine-tune with the original 19-patient LOSO supervised protocol.

## Commands

Global embedding reconstruction:

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --embedding-dim 32 --decoder-hidden-dim 128 --supervised-epochs 100 --patience 100 --summary-name masked_ssl_psd_fc_wpli_transfer_seed2_psd20_fc20_summary.csv
```

Local pre-pooling reconstruction:

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --data-scope all-patient --device cuda --seed 2 --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --embedding-dim 32 --decoder-hidden-dim 128 --supervised-epochs 100 --patience 100 --summary-name masked_local_ssl_psd_fc_wpli_transfer_seed2_psd20_fc20_summary.csv
```

## Results

Reference no-SSL gated CNN:

```text
accuracy=0.8421, balanced_accuracy=0.8333, roc_auc=0.8222, pr_auc=0.8158
```

Masked SSL transfer results:

```text
run                         accuracy   balanced_acc   sensitivity   specificity   roc_auc   pr_auc
global embedding decoder    0.6316     0.6333         0.6000        0.6667        0.7111    0.7053
local pre-pooling decoder   0.7368     0.7333         0.8000        0.6667        0.7222    0.7253
```

Global decoder SSL losses:

```text
psd_masked:  first_loss=1.0126, last_loss=0.7255, min_loss=0.6523
wpli_masked: first_loss=1.0289, last_loss=0.8319, min_loss=0.7710
```

Local decoder SSL losses:

```text
psd_masked:  first_loss=0.5800, last_loss=0.2995, min_loss=0.2219
wpli_masked: first_loss=0.9782, last_loss=0.7086, min_loss=0.6362
```

## Local SSL Tuning

After the first local decoder run, lightweight tuning focused on the wPLI mask
strength and EO/EC consistency loss.

Summary file:

```text
results/metrics/masked_local_ssl_tuning_summary.csv
```

Results:

```text
setting                         fc_node   fc_edge   fc_band   consistency   accuracy   balanced_acc   roc_auc   pr_auc
default local mask              0.05      0.15      0.05      0.00          0.7368     0.7333         0.7222    0.7253
lower wPLI mask                 0.02      0.05      0.02      0.00          0.7895     0.7833         0.8111    0.7957
lower wPLI mask + consistency   0.02      0.05      0.02      0.05          0.6842     0.6778         0.6556    0.6728
very low wPLI mask              0.01      0.03      0.01      0.00          0.7368     0.7333         0.8000    0.8084
```

Best tuned local masked SSL configuration:

```text
fc_node_mask_prob=0.02
fc_edge_mask_prob=0.05
fc_band_mask_prob=0.02
eo_ec_consistency_weight=0.0
accuracy=0.7895
balanced_accuracy=0.7833
roc_auc=0.8111
pr_auc=0.7957
```

This improves over the first local masked SSL run and is close to the no-SSL
reference ROC-AUC, but still does not exceed the no-SSL reference accuracy.

## Output Files

Global decoder:

```text
results/metrics/masked_ssl_psd_fc_wpli_transfer_seed2_psd20_fc20_summary.csv
results/metrics/dl_model_comparison_masked_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/predictions/dl_loso_predictions_masked_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/ssl/feature_ssl_pretraining_history_masked_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/training_logs/dl_loss_history_masked_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/figures/dl_loss_curve_masked_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.png
```

Local pre-pooling decoder:

```text
results/metrics/masked_local_ssl_psd_fc_wpli_transfer_seed2_psd20_fc20_summary.csv
results/metrics/dl_model_comparison_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/predictions/dl_loso_predictions_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/ssl/feature_ssl_pretraining_history_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/training_logs/dl_loss_history_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.csv
results/figures/dl_loss_curve_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_bs8_sup100.png
```

Best tuned local pre-pooling decoder:

```text
results/metrics/masked_local_ssl_tuning_low_fc_mask_seed2_summary.csv
results/metrics/dl_model_comparison_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_bs8_sup100.csv
results/predictions/dl_loso_predictions_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_bs8_sup100.csv
results/ssl/feature_ssl_pretraining_history_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_bs8_sup100.csv
results/training_logs/dl_loss_history_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_bs8_sup100.csv
results/figures/dl_loss_curve_masked_local_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_bs8_sup100.png
```

## Interpretation

Local pre-pooling reconstruction is better than decoding from the global
embedding. It improved accuracy from `0.6316` to `0.7368` and produced lower
PSD reconstruction loss. This supports the hypothesis that reconstructing
after global pooling is too strong a bottleneck.

The tuning run shows that wPLI masking should be lighter than the PSD masking
in this small dataset. Reducing wPLI node/edge/band mask probabilities improved
accuracy to `0.7895`. Making the mask even lighter did not help accuracy,
although PR-AUC improved slightly.

EO/EC consistency loss with weight `0.05` was harmful in this setup. A likely
reason is that EO and EC state differences may contain useful recovery-related
information; forcing their embeddings too close can erase that signal.

The tuned local masked SSL result still does not exceed the no-SSL gated CNN
reference (`0.8421` accuracy). The remaining limitation is probably the wPLI
branch: the current FC encoder is still a Conv1D over edge order rather than an
anatomical graph encoder.

More promising next variants would be:

```text
lighter masks with longer pretraining;
joint masked reconstruction plus EO/EC state-consistency loss;
true GNN/GraphMAE encoder for FC instead of the current FC Conv1D branch.
```
