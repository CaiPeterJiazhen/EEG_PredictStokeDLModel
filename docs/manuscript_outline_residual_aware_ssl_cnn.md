# Manuscript Outline: Residual-Aware SSL-CNN

Suggested title: **Residual-aware self-supervised multimodal EEG learning for predicting upper-limb proportional recovery after tACS in stroke: a pilot LOSO study**

## Introduction

- Motivate baseline EEG prediction for tACS recovery stratification.
- Explain why PSD and WPLI are plausible EEG biomarkers.
- State the small-cohort challenge and the need for seed-stable validation.
- Avoid claiming clinical deployment readiness.

## Materials And Methods

- Cohort: 19 supervised stroke patients.
- Outcome: residual threshold `1.5`, fixed prediction threshold `0.5`.
- Features: PSD EO/EC `62 x 90`, WPLI EO/EC `1891 x 6`.
- Validation: patient-level LOSO and 10 repeated seeds.
- Model: Patient-level Barlow SSL-CNN with residual-aware auxiliary heads and SWA.
- Baselines: updated_sub05_sub28 PSD/WPLI ML EEG and no-SSL CNN.
- SSL data scope: fold-specific test-subject exclusion; historical unlabeled pretraining if `all-patient`.

## Results

- Use `results/tables/model_performance_main_table.csv`.
- Use `results/tables/seed_stability_table.csv`.
- Present final SSL-CNN as improving seed-level stability and Brier relative to the no-SSL rerun reference, not as universally superior across all possible baseline families.

## Explainability / Neurophysiological Interpretation

- Use SmoothGrad-smoothed IG, occlusion, topomaps, WPLI connectomes, and network-level validation.
- Interpret EO PSD delta/beta and EC WPLI beta network patterns cautiously.
- Do not claim a single causal edge or causal biomarker.

## Discussion

- Emphasize residual-aware training as a way to use continuous recovery information while keeping binary classification inference.
- Discuss why the no-SSL EEG baseline remains strong.
- Discuss qEEG as supplementary biomarker evidence.

## Limitations

- n=19.
- No external validation.
- Historical SSL data scope caveat.
- Explainability is associative and model-dependent.
- LOSO seeds are repeated fits, not independent samples.

## Conclusion

- A compact residual-aware Patient-level Barlow SSL-CNN improved seed-level stability and score metrics in a pilot LOSO cohort.
- Findings are exploratory and require external validation or pre-registered replication.

## Phrases To Avoid

- clinical deployment ready
- causal biomarker
- single edge confirmed
- SSL universally better
- external generalization proven

## Preferred Framing

- exploratory
- pilot
- patient-level LOSO
- seed-stable
- hypothesis-generating biomarkers
- requires external validation
