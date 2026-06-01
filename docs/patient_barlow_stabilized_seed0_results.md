# Patient-level Barlow Stabilized Seed0 Results

This pilot reuses fold-specific Patient-level Barlow encoder checkpoints and changes only supervised fine-tuning.
No qEEG branch, MIL, Dual encoder, or new input feature is used.

sub09/sub14 were monitored as post-hoc repeated-error subjects, not used for optimization.

| model_group                         |   accuracy |   balanced_accuracy |   sensitivity |   specificity |   roc_auc |   pr_auc |   brier_score | source_file                                                                                                                                    |
|:------------------------------------|-----------:|--------------------:|--------------:|--------------:|----------:|---------:|--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| patient_barlow_swa_seed0            |   0.789474 |            0.777778 |           1   |      0.555556 |  0.788889 | 0.749463 |      0.175163 | dl_model_comparison_patient_barlow_swa_seed0.csv                                                                                               |
| patient_barlow_sam_seed0            |   0.526316 |            0.516667 |           0.7 |      0.333333 |  0.666667 | 0.764918 |      0.227151 | dl_model_comparison_patient_barlow_sam_seed0.csv                                                                                               |
| patient_barlow_staged_sam_swa_seed0 |   0.631579 |            0.622222 |           0.8 |      0.444444 |  0.788889 | 0.83373  |      0.191544 | dl_model_comparison_patient_barlow_staged_sam_swa_seed0.csv                                                                                    |
| prior_patient_barlow_seed0          |   0.736842 |            0.727778 |           0.9 |      0.555556 |  0.677778 | 0.677434 |    nan        | dl_model_comparison_feature_ssl_all-patient_psd-fc-wpli_gated_cnn_barlow_finetune_seed0_pre50_temp0_2_noise0_02_mask0_01_bs8_sup100_proj32.csv |
