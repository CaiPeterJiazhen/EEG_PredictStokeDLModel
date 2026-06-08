# Cohort Characteristics

Generated from the current project metadata pipeline on 2026-06-01.

Source logic:

- Supervised cohort: `load_supervised_label_table()` using the final 19-patient integrity workbook.
- All-patient EEG cohort: all patient subjects indexed from the current preprocessed patient EEG directory and matched to the clinical workbook.
- Sex labels are reported as male/female.
- Affected-hand labels are reported as left/right.

## Supervised Cohort: 19 Patients

This is the final supervised LOSO-CV training cohort.

Label definition:

- `Residual = Predicted Delta FMA - Observed Delta FMA`
- `label = 1` if `Residual <= median Residual`
- `label = 0` if `Residual > median Residual`

| Characteristic | Value |
|---|---:|
| N | 19 |
| Age, mean +/- SD | 64.7 +/- 6.5 |
| Age, median [IQR] | 65.0 [60.5, 69.0] |
| Sex | male 8, female 11 |
| Affected hand | left 11, right 8 |
| Disease duration, mean +/- SD | 36.3 +/- 17.8 |
| Disease duration, median [IQR] | 33.0 [21.5, 49.5] |
| FMA pre, mean +/- SD | 40.5 +/- 23.8 |
| FMA pre, median [IQR] | 51.0 [16.5, 61.0] |
| FMA post, mean +/- SD | 45.5 +/- 23.2 |
| FMA post, median [IQR] | 62.0 [21.0, 64.0] |
| MBI pre, mean +/- SD | 55.5 +/- 19.9 |
| MBI pre, median [IQR] | 55.0 [42.5, 70.0] |
| MBI post, mean +/- SD | 76.8 +/- 21.5 |
| MBI post, median [IQR] | 80.0 [57.5, 95.0] |
| Observed Delta FMA, mean +/- SD | 5.0 +/- 4.1 |
| Observed Delta FMA, median [IQR] | 3.0 [2.0, 6.5] |
| Predicted Delta FMA, mean +/- SD | 17.9 +/- 16.7 |
| Predicted Delta FMA, median [IQR] | 10.5 [3.5, 34.7] |
| Residual, mean +/- SD | 12.9 +/- 16.2 |
| Residual, median [IQR] | 1.5 [0.0, 30.2] |
| Label distribution | label 0: 9, label 1: 10 |

### Supervised Cohort By Label

| Characteristic | label 0, n=9 | label 1, n=10 |
|---|---:|---:|
| Age, mean +/- SD | 65.4 +/- 5.1 | 64.1 +/- 7.8 |
| Sex | male 4, female 5 | male 4, female 6 |
| Affected hand | left 5, right 4 | left 6, right 4 |
| Disease duration, mean +/- SD | 37.2 +/- 21.4 | 35.4 +/- 15.0 |
| FMA pre, mean +/- SD | 19.2 +/- 17.0 | 59.6 +/- 4.7 |
| FMA post, mean +/- SD | 24.8 +/- 17.2 | 64.1 +/- 1.2 |
| MBI pre, mean +/- SD | 41.7 +/- 15.6 | 68.0 +/- 14.6 |
| MBI post, mean +/- SD | 58.9 +/- 16.5 | 93.0 +/- 8.2 |
| Observed Delta FMA, mean +/- SD | 5.6 +/- 4.4 | 4.5 +/- 3.9 |
| Residual, mean +/- SD | 27.2 +/- 12.4 | -0.02 +/- 0.87 |

## All-Patient EEG Cohort: 28 Patients

This cohort includes every patient subject currently indexed from the preprocessed patient EEG directory and matched to the clinical workbook.

The clinical workbook also contains `sub04`, but `sub04` is not indexed in the current project EEG data and is therefore not included in the all-patient EEG cohort.

| Characteristic | Value |
|---|---:|
| N | 28 |
| Supervised cohort overlap | 19 |
| Non-supervised patients | 9 |
| Age, mean +/- SD | 63.3 +/- 8.7 |
| Age, median [IQR] | 64.0 [57.5, 69.0] |
| Sex | male 14, female 14 |
| Affected hand | left 16, right 12 |
| Disease duration, mean +/- SD | 39.4 +/- 20.6 |
| Disease duration, median [IQR] | 37.0 [21.8, 50.3] |
| FMA pre, n | 28 |
| FMA pre, mean +/- SD | 39.7 +/- 23.9 |
| FMA pre, median [IQR] | 51.0 [12.0, 61.3] |
| MBI pre, n | 28 |
| MBI pre, mean +/- SD | 57.7 +/- 20.7 |
| MBI pre, median [IQR] | 60.0 [43.8, 71.3] |
| FMA post, n | 20 |
| FMA post, mean +/- SD | 46.5 +/- 23.1 |
| FMA post, median [IQR] | 62.5 [22.5, 64.3] |
| MBI post, n | 20 |
| MBI post, mean +/- SD | 78.0 +/- 21.5 |
| MBI post, median [IQR] | 82.5 [58.8, 96.3] |
| Observed Delta FMA, n | 20 |
| Observed Delta FMA, mean +/- SD | 4.8 +/- 4.1 |
| Observed Delta FMA, median [IQR] | 3.0 [2.0, 6.3] |
| Residual, n | 20 |
| Residual, mean +/- SD | 12.2 +/- 16.1 |
| Residual, median [IQR] | 1.2 [-0.03, 29.5] |

## Non-Supervised Patients

These 9 patients are indexed in the current preprocessed patient EEG directory but are not part of the final 19-patient supervised training cohort.

| Subject ID | Supervised training cohort | Notes |
|---|---|---|
| sub02 | No | Non-supervised patient |
| sub03 | No | Non-supervised patient |
| sub06 | No | Non-supervised patient |
| sub12 | No | Non-supervised patient |
| sub19 | No | Non-supervised patient |
| sub21 | No | Non-supervised patient |
| sub23 | No | Non-supervised patient |
| sub25 | No | Non-supervised patient |
| sub26 | No | Non-supervised patient |

## Patient Membership Table

| Subject ID | Cohort status |
|---|---|
| sub01 | Supervised 19 |
| sub02 | Non-supervised patient |
| sub03 | Non-supervised patient |
| sub05 | Supervised 19 |
| sub06 | Non-supervised patient |
| sub07 | Supervised 19 |
| sub08 | Supervised 19 |
| sub09 | Supervised 19 |
| sub10 | Supervised 19 |
| sub11 | Supervised 19 |
| sub12 | Non-supervised patient |
| sub13 | Supervised 19 |
| sub14 | Supervised 19 |
| sub15 | Supervised 19 |
| sub16 | Supervised 19 |
| sub17 | Supervised 19 |
| sub18 | Supervised 19 |
| sub19 | Non-supervised patient |
| sub20 | Supervised 19 |
| sub21 | Non-supervised patient |
| sub22 | Supervised 19 |
| sub23 | Non-supervised patient |
| sub24 | Supervised 19 |
| sub25 | Non-supervised patient |
| sub26 | Non-supervised patient |
| sub27 | Supervised 19 |
| sub28 | Supervised 19 |
| sub29 | Supervised 19 |

## Interpretation Notes

- The supervised 19-patient cohort is the only cohort used for final supervised LOSO-CV model evaluation.
- The all-patient 28-patient cohort is available for unsupervised or self-supervised learning, exploratory analyses, and descriptive reporting.
- Follow-up outcome summaries in the all-patient cohort use only the 20 patients with available post-treatment FMA/MBI.
- The 9 non-supervised patients should not be assigned supervised labels unless their eligibility and follow-up completeness are explicitly redefined.
