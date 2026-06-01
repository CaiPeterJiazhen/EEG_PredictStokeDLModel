# qEEG-guided SSL-CNN Results

## Locked Candidate

Model: qEEG-guided graph-smoothed masked-VICReg SSL-CNN.

Primary qEEG feature: `qeeg_ec_global_slow_fast_bsi` only.

CNN input features: PSD EO/EC `62 x 90` and WPLI EO/EC `1891 x 6`.

Supervised branch: fold-local z-scored qEEG scalar clipped to `[-3, 3]`, `Linear(1,4) -> ReLU -> Linear(4,4) -> ReLU`, concatenated with the 64-dim PSD+WPLI CNN embedding.

SSL objective: masked reconstruction + VICReg + WPLI graph smoothness + `0.05 * Huber(q_pred, qEEG_target)`.

## Main 10-seed Result

- qEEG-guided SSL-CNN mean accuracy: `0.7684`; min accuracy: `0.6842`; std accuracy: `0.0421`.
- qEEG-guided SSL-CNN mean ROC AUC: `0.8100`; mean PR AUC: `0.8101`; mean Brier: `0.1846`.
- qEEG-guided SSL-CNN seedmean10 accuracy: `0.8421`; ROC AUC: `0.8444`; PR AUC: `0.8637`; Brier: `0.1632`.
- Prior locked no-SSL reference seedmean10 accuracy: `0.8421`; ROC AUC: `0.8111`; PR AUC: `0.7824`; Brier: `0.1714`.

## Required Questions

1. What is qEEG EC global slow/fast BSI?

It is an eyes-closed, global hemispheric Brain Symmetry Index computed from PSD slow-band burden relative to fast-band preservation. It summarizes left-right asymmetry in slow/fast EEG balance.

2. Why is it biologically plausible?

Post-stroke recovery is plausibly related to asymmetric slowing, impaired alpha/beta preservation, and hemispheric imbalance. The feature is low-dimensional, EEG-derived, and does not use labels.

3. Does qEEG branch improve no-SSL CNN?

No. no-SSL CNN + qEEG branch mean accuracy was `0.7263` with min accuracy `0.5789`, worse than the prior no-SSL 10-seed mean accuracy `0.7947`.

4. Does qEEG branch improve SSL-CNN?

Mixed. The qEEG-guided SSL-CNN seedmean accuracy matched the prior no-SSL seedmean accuracy and improved seedmean ROC AUC, PR AUC, and Brier, but its 10-seed per-run mean accuracy was lower.

5. Does qEEG auxiliary SSL improve over qEEG branch alone?

Only partially. It improved over no-SSL+qEEG branch in mean accuracy (`0.7684` vs `0.7263`), but did not outperform the original no-SSL reference.

6. Does it improve 10-seed mean accuracy?

No. qEEG-guided mean accuracy `0.7684` is below the prior locked no-SSL mean accuracy `0.7947`.

7. Does it improve min accuracy?

No clear claim. qEEG-guided min accuracy was `0.6842`; the prior no-SSL document did not preserve a comparable min-accuracy row in the summary file.

8. Does it reduce seed std?

It reduced variance relative to the newly run no-SSL+qEEG branch (`0.0421` vs `0.0737`), but that branch itself was weak.

9. Does it improve ROC AUC, PR AUC, Brier?

Yes at seedmean score level versus the prior locked no-SSL reference: qEEG-guided seedmean ROC AUC `0.8444` vs `0.8111`, PR AUC `0.8637` vs `0.7824`, and Brier `0.1632` vs `0.1714`. The per-run mean Brier was still worse (`0.1846`), so this is a seed-averaged score-level improvement rather than a seed-stability win.

10. Does it reduce sub09/sub14/sub05/sub13 errors?

Not enough. The watched-subject table shows persistent errors in this group; these subjects were monitored post hoc, not optimized directly.

11. Is improvement mainly qEEG, SSL, or combination?

This run did not support promoting the qEEG-guided branch as the main model. The SSL combination improved over the weak no-SSL+qEEG branch but did not outperform the active EEG-only final model package.

12. Should this be main model, supplementary model, or hypothesis-generating candidate?

Supplementary/hypothesis-generating model test. It gives a seedmean score-level improvement over the prior no-SSL reference, but not the requested 10-seed mean-accuracy or hard-negative stability improvement.

## Output Files

- Seedmean predictions: `results\predictions\seedmean_qeeg_guided_graphsmooth_mvicreg.csv`
- Model selection summary: `results/metrics/qeeg_guided_ssl_cnn_10seed_model_selection_summary.csv`
- Seed stability summary: `results/metrics/qeeg_guided_ssl_cnn_seed_stability_summary.csv`
- Statistical comparison: `results/metrics/qeeg_guided_ssl_cnn_statistical_comparison.csv`
- Subject errors: `results/metrics/qeeg_guided_ssl_cnn_subject_error_frequency.csv`
- Watched subjects: `results/metrics/sub09_sub14_qeeg_guided_scores_summary.csv`

## Caveats

- Seeds are repeated training runs, not independent patients.
- The available paired comparison against no-SSL uses the local no-SSL schemeA 6-seed ensemble file when present; the prior 10-seed no-SSL values are included as locked reference metrics.
- No clinical generalization is claimed without external validation.
