# WPLI Segment Barlow Attention MIL Seed0 Results

## 1. Why Dual PSD+WPLI Is Not The Next Priority

The seed0 pilot showed that dual PSD+WPLI Segment Barlow did not improve the main patient-level accuracy or balanced accuracy, and freezing the encoder degraded performance. The new G experiment therefore focuses on the stronger seed0 signal: WPLI Segment Barlow.

## 2. Why WPLI Segment Barlow

The WPLI Segment Barlow seed0 run had the strongest ROC-AUC among the segment SSL pilots, suggesting useful ordering information even when the fixed 0.5 threshold did not improve accuracy. MIL keeps that segment-level information instead of averaging it away before supervised training.

## 3. Why Attention MIL Fits This EEG Setting

Each patient has many baseline EO/EC WPLI segments, but the clinical recovery label is patient-level. Attention MIL treats segments as instances inside one patient bag, learns which segments carry signal, pools to one patient embedding, and computes one BCE loss per patient.

## 4. LOSO Leakage Controls

- The held-out LOSO subject is excluded from supervised fit and validation patients.
- The fold-local scaler is fit only on fit subjects, not validation or test.
- The WPLI Segment Barlow encoder is fold-specific and its checkpoint metadata must match branch, objective, fold index, test subject, excluded subject, seed, encoder kind, embedding dimension, and feature manifest hash.
- Predictions and reported metrics are patient-level only.

## 5. sub09/sub14 QC

- sub09: residual=37.000, distance_to_threshold=35.500, no objective QC outlier.
- sub14: residual=2.200, distance_to_threshold=0.700, no objective QC outlier.

## 6. Seed0 A/C/D/G Comparison

| method_id | accuracy | balanced_accuracy | roc_auc | pr_auc | brier_score | tn | fp | fn | tp | sub09_y_score | sub09_y_pred | sub14_y_score | sub14_y_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_no_ssl_cnn | 0.6842 | 0.6722 | 0.7667 | 0.8118 | 0.2287 | 4 | 5 | 1 | 9 | 0.9826 | 1 | 0.9793 | 1 |
| C_wpli_segment_barlow_patient_averaged_cnn | 0.6842 | 0.6722 | 0.7889 | 0.8027 | 0.2123 | 4 | 5 | 1 | 9 | 0.9978 | 1 | 0.9857 | 1 |
| D_dual_segment_barlow_finetune | 0.6842 | 0.6722 | 0.7333 | 0.7557 | 0.2348 | 4 | 5 | 1 | 9 | 0.9904 | 1 | 0.9988 | 1 |
| G_wpli_segment_barlow_attention_mil | 0.5263 | 0.5167 | 0.4778 | 0.5202 | 0.3906 | 3 | 6 | 3 | 7 | 0.9998 | 1 | 0.5806 | 1 |

## 7. Continue Six Seeds?

Seed0 meets only the subject-score movement criterion, not the aggregate metric criteria. A cautious continuation to seeds 1,2,3,7,13 is defensible because: sub14 y_score moved toward the correct direction vs A_no_ssl_cnn; sub14 y_score moved toward the correct direction vs C_wpli_segment_barlow_patient_averaged_cnn. Inspect attention concentration and sub09/sub14 QC before treating this as a method improvement.

## 8. Next Command If Continuing

```powershell
& 'E:\ProgramFiles\Anaconda\Anaconda3\python.exe' -B scripts/21_train_wpli_segbarlow_mil_seed0.py --config configs/paths.example.yaml --device cuda --seed 1 --stage1-epochs 20 --stage2-epochs 80 --max-segments-per-state 64 --head-lr 0.002 --encoder-lr 0.0001 --stage2-head-lr 0.001 --weight-decay 0.00001 --embedding-dim 32 --dropout 0.0 --reuse-ssl-encoders --save-ssl-encoders --freeze-bn
```

Repeat with `--seed 2`, `--seed 3`, `--seed 7`, and `--seed 13` only if the seed0 criteria justify continuation.
