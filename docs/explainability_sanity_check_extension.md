# Explainability Sanity Check Extension

The explainability script now implements classifier-only randomization, classifier plus final projection randomization, full PSD encoder randomization, full WPLI encoder randomization, and input permutation checks.

The committed extended summary is conservative: it reuses the existing locked sanity CSV for available rows and marks newly implemented checks that require a full attribution rerun. A full rerun was not launched here because it would overwrite the locked attribution summary artifacts.

| sanity check | status | n | PSD signed rho | WPLI signed rho | PSD abs rho | WPLI abs rho |
|---|---|---:|---:|---:|---:|---:|
| classifier_only_randomization | available | 8 | 0.338 | 0.471 | 0.805 | 0.774 |
| classifier_final_projection_randomization | implemented_for_rerun_not_recomputed_in_committed_summary | 0 |  |  |  |  |
| full_psd_encoder_randomization | implemented_for_rerun_not_recomputed_in_committed_summary | 0 |  |  |  |  |
| full_wpli_encoder_randomization | implemented_for_rerun_not_recomputed_in_committed_summary | 0 |  |  |  |  |
| input_permutation | available | 8 | 0.186 | 0.168 | 0.186 | 0.168 |

Interpretation remains conservative. The prior classifier-randomization result was mixed, especially for absolute attribution structure, so final biomarker claims should rely on convergent evidence from stability, occlusion, and network-level validation rather than a single attribution table.
