# tACS EEG Proportional Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, reproducible Python pipeline for predicting tACS post-treatment FMA-UE proportional recovery from baseline EO/EC EEG and baseline clinical features.

**Architecture:** The project is organized as a small Python package with explicit modules for metadata, EEG IO, channel handling, feature extraction, model training, evaluation, and explainability. Every patient-level operation is keyed by normalized subject ID, and every split-sensitive transformation is fit inside LOSO training folds only.

**Tech Stack:** Python, NumPy, SciPy, pandas, scikit-learn, PyTorch, pytest, optional MNE for validation, optional MATLAB only when Python cannot read a required EEGLAB artifact.

---

## 0. Required Reading Before Any Task

Each worker must read these files before implementation:

```text
docs/project_context.md
tacs_eeg_proportional_recovery_project_design.md
```

Workers must not rely on chat history for fixed facts. The authoritative facts are in `docs/project_context.md`.

## 1. Target Project Structure

```text
configs/
  paths.example.yaml
  channel_mapping.yaml
  config_baseline.yaml
  config_supervised.yaml
  config_ssl.yaml

src/
  eeg_recovery/
    __init__.py
    config.py
    metadata/
      __init__.py
      subjects.py
      labels.py
    io/
      __init__.py
      eeglab.py
      index.py
    channels/
      __init__.py
      mapping.py
      hemisphere_flip.py
    features/
      __init__.py
      psd.py
      connectivity.py
      feature_tables.py
    models/
      __init__.py
      encoders.py
      dual_state_model.py
      fusion_3d_model.py
      clinical_mlp.py
      ssl_model.py
    training/
      __init__.py
      loso.py
      train_baselines.py
      train_supervised.py
      train_ssl.py
      metrics.py
      schedulers.py
    explainability/
      __init__.py
      psd_attribution.py
      fc_attribution.py
      clinical_validation.py
      visualization.py
    utils/
      __init__.py
      seed.py
      paths.py
      logging.py

scripts/
  01_prepare_metadata.py
  02_compute_psd.py
  03_compute_fc.py
  04_train_ml_baselines.py
  05_train_supervised_loso.py
  06_train_ssl.py
  07_run_explainability.py

tests/
  test_metadata_labels.py
  test_eeg_index.py
  test_channel_mapping.py
  test_eeglab_io.py
  test_psd_features.py
  test_connectivity_shapes.py
  test_loso_leakage.py
```

Formal generated outputs should be placed only under:

```text
data/processed/
data/features/
results/
```

Do not copy external raw EEG or Excel source files into this project.

## 2. Development Rules

- Use `docs/task_status.md` to track progress.
- Keep module ownership narrow. A subagent should own one task and the listed files only.
- Do not run multiple implementation subagents in parallel if their write sets overlap.
- Each task must include tests before or alongside implementation.
- Do not use patient names in output filenames.
- Do not use treatment-post variables as model inputs.
- Do not introduce temporary project files. If a debug file is unavoidable, write it outside the project and delete it before completion.
- This directory is currently not a git repository. If git is initialized later, commit after each completed task.

## Task 1: Project Scaffold And Configuration

**Files:**

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/paths.example.yaml`
- Create: `configs/config_baseline.yaml`
- Create: `configs/config_supervised.yaml`
- Create: `configs/config_ssl.yaml`
- Create: `src/eeg_recovery/__init__.py`
- Create: `src/eeg_recovery/config.py`
- Create: `src/eeg_recovery/utils/paths.py`
- Create: `tests/test_config.py`

- [x] **Step 1: Write tests for config loading**

Create tests that confirm path config can be loaded from YAML, required keys are validated, and missing external files are reported without creating files.

Run:

```powershell
pytest tests/test_config.py -v
```

Expected before implementation: import or function-not-found failure.

- [x] **Step 2: Implement package scaffold and config reader**

Implement a minimal dataclass-based config reader. Required config keys:

```text
patient_info_integrity_xlsx
patient_info_clinical_xlsx
patient_eeg_root
health_eeg_root
standard_1005_ced
output_root
```

- [x] **Step 3: Add `.gitignore` for clean project state**

Ignore:

```text
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.pyc
*.pyo
*.egg-info/
data/processed/
data/features/
results/
checkpoints/
*.pt
*.pth
*.ckpt
*.tmp
```

- [x] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_config.py -v
```

Expected: all tests pass.

## Task 2: Metadata Parsing And Label Generation

**Files:**

- Create: `src/eeg_recovery/metadata/__init__.py`
- Create: `src/eeg_recovery/metadata/subjects.py`
- Create: `src/eeg_recovery/metadata/labels.py`
- Create: `scripts/01_prepare_metadata.py`
- Create: `tests/test_metadata_labels.py`

- [x] **Step 1: Write tests for subject ID normalization**

Test examples:

```text
sub1 -> sub01
sub01 -> sub01
sub011 -> sub11
sub013 -> sub13
sub021 -> sub21
```

- [x] **Step 2: Write tests for fixed label table**

Use the 19-subject table in `docs/project_context.md` as expected values. Tests must verify:

```text
n_subjects == 19
median_residual == 1.5
class_counts == {0: 9, 1: 10}
sub22 residual == 1.5 and label == 1
```

- [x] **Step 3: Implement Excel readers**

Read both Excel workbooks. Extract:

```text
subject_id
age
sex
duration
affected_hand
FMA_pre
FMA_post
MBI_pre
MBI_post
```

Only `FMA_post` and `MBI_post` may be used for label derivation or later clinical validation, not model input.

- [x] **Step 4: Implement label generation**

Compute:

```text
Delta_FMA_pred = 0.7 * (66 - FMA_pre)
Delta_FMA_obs = FMA_post - FMA_pre
Residual = Delta_FMA_pred - Delta_FMA_obs
label = 1 if Residual <= median_residual else 0
```

- [x] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_metadata_labels.py -v
```

Expected: all tests pass and no files are written outside configured output paths.

## Task 3: EEG File Indexing

**Files:**

- Create: `src/eeg_recovery/io/__init__.py`
- Create: `src/eeg_recovery/io/index.py`
- Create: `tests/test_eeg_index.py`

- [x] **Step 1: Write tests for file indexing**

Tests must verify:

```text
baseline patient files include EO and EC for each supervised subject
*1.set is EO
*2.set is EC
patient stages are parsed as 基线, 即时, 阶段, 最终
health subjects are indexed separately
```

- [x] **Step 2: Implement index builder**

The index record should include:

```text
group: patient or health
subject_id
stage
state: EO or EC
set_path
fdt_path
is_supervised_subject
```

- [x] **Step 3: Validate supervised baseline coverage**

Raise a clear error if any supervised subject lacks exactly one EO and one EC baseline file.

- [x] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_eeg_index.py -v
```

Expected: all tests pass.

## Task 4: EEGLAB Metadata And Data Reader

**Files:**

- Create: `src/eeg_recovery/io/eeglab.py`
- Create: `tests/test_eeglab_io.py`

- [x] **Step 1: Write tests for `.set` metadata reading**

Use a small number of real `.set` files through configured paths. Verify:

```text
nbchan == 62
srate == 250
trials == 1
len(ch_names) == 62
M1 and M2 are absent
all tested files share identical channel order
```

- [x] **Step 2: Implement `.set` metadata reader**

Use `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)` and avoid loading `.fdt` until data is requested.

- [x] **Step 3: Implement `.fdt` data reader**

Read EEGLAB `.fdt` as float32 and return a NumPy array shaped:

```text
channels x samples
```

Validate the file length against:

```text
nbchan * pnts
```

- [x] **Step 4: Add reader validation**

If the `.fdt` length does not match expected size, raise a descriptive exception with subject, state, path, expected samples, and observed samples.

- [x] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_eeglab_io.py -v
```

Expected: all tests pass.

## Task 5: 62-Channel Mapping And Hemisphere Flip

**Files:**

- Create: `configs/channel_mapping.yaml`
- Create: `src/eeg_recovery/channels/__init__.py`
- Create: `src/eeg_recovery/channels/mapping.py`
- Create: `src/eeg_recovery/channels/hemisphere_flip.py`
- Create: `tests/test_channel_mapping.py`

- [x] **Step 1: Write tests for mapping completeness**

Tests must verify:

```text
all 62 actual channels are present
left/right pairs are symmetric
midline channels map to themselves
M1 and M2 are absent
```

- [x] **Step 2: Build 62-channel mirror map**

Include these pair families where present:

```text
FP1-FP2
AF3-AF4
F7-F8, F5-F6, F3-F4, F1-F2
FT7-FT8
FC5-FC6, FC3-FC4, FC1-FC2
T7-T8
C5-C6, C3-C4, C1-C2
TP7-TP8
CP5-CP6, CP3-CP4, CP1-CP2
P7-P8, P5-P6, P3-P4, P1-P2
PO7-PO8, PO5-PO6, PO3-PO4
CB1-CB2
O1-O2
```

Midline self-map:

```text
FPZ, FZ, FCZ, CZ, CPZ, PZ, POZ, OZ
```

- [x] **Step 3: Implement flip function**

Function behavior:

```text
affected_hand == "右": return data unchanged
affected_hand == "左": return data reordered by mirror map
```

This follows project convention: unify to right affected hand / C3 stimulation / left hemisphere as stimulation side.

- [x] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_channel_mapping.py -v
```

Expected: all tests pass.

## Task 6: PSD Feature Extraction

**Files:**

- Create: `src/eeg_recovery/features/__init__.py`
- Create: `src/eeg_recovery/features/psd.py`
- Create: `scripts/02_compute_psd.py`
- Create: `tests/test_psd_features.py`

- [x] **Step 1: Write tests for PSD shape**

Tests must verify:

```text
single-state PSD shape == (62, 90)
frequency bins span 0.5-45 Hz with 0.5 Hz resolution
EO and EC are saved separately
left-hand patients are flipped before PSD calculation
```

- [x] **Step 2: Implement PSD calculation**

Use a deterministic Welch-style PSD configuration. Store enough metadata to reproduce:

```text
subject_id
state
channel_names_after_alignment
frequency_bins
sampling_rate
source_set_path
```

- [x] **Step 3: Implement feature writer**

Write formal outputs under:

```text
data/features/psd/
```

No temporary files should remain in the project root.

- [x] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_psd_features.py -v
```

Expected: all tests pass.

## Task 7: Functional Connectivity Feature Extraction

**Files:**

- Create: `src/eeg_recovery/features/connectivity.py`
- Create: `scripts/03_compute_fc.py`
- Create: `tests/test_connectivity_shapes.py`

- [x] **Step 1: Write tests for FC shape and edge order**

Tests must verify:

```text
n_edges == 1891
wPLI shape == (1891, 6)
imaginary coherence shape == (1891, 6)
edge list is deterministic and derived from aligned 62-channel order
```

- [x] **Step 2: Implement deterministic edge list**

Use upper-triangle order:

```text
for i in range(n_channels):
    for j in range(i + 1, n_channels):
        edge = (channel_i, channel_j)
```

- [x] **Step 3: Implement wPLI and imaginary coherence**

Compute per state and per band:

```text
EO wPLI: 1891 x 6
EC wPLI: 1891 x 6
EO iCoh: 1891 x 6
EC iCoh: 1891 x 6
```

- [x] **Step 4: Implement feature writer**

Write formal outputs under:

```text
data/features/fc/
```

- [x] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_connectivity_shapes.py -v
```

Expected: all tests pass.

## Task 8: LOSO Split And Leakage Tests

**Files:**

- Create: `src/eeg_recovery/training/__init__.py`
- Create: `src/eeg_recovery/training/loso.py`
- Create: `tests/test_loso_leakage.py`

- [x] **Step 1: Write leakage tests**

Tests must verify:

```text
19 folds are produced
each fold has 1 test subject and 18 train subjects
test subject never appears in train subjects
scalers/selectors expose fit only on train fold data
```

- [x] **Step 2: Implement LOSO generator**

Fold record should include:

```text
fold_index
test_subject_id
train_subject_ids
```

- [x] **Step 3: Add transformation guard utilities**

Provide helper wrappers or conventions so scikit-learn transformers are fit only inside a fold.

- [x] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_loso_leakage.py -v
```

Expected: all tests pass.

## Task 9: Traditional ML Baselines

**Files:**

- Create: `src/eeg_recovery/features/feature_tables.py`
- Create: `src/eeg_recovery/training/train_baselines.py`
- Create: `src/eeg_recovery/training/metrics.py`
- Create: `scripts/04_train_ml_baselines.py`
- Create: `tests/test_baseline_training.py`

- [x] **Step 1: Write tests for fold-local preprocessing**

Tests must verify that standardization, PCA, and feature selection are fit on training subjects only.

- [x] **Step 2: Build tabular features**

PSD band power:

```text
state x channel x band = 2 x 62 x 6 = 744 features
```

FC features start as:

```text
state x edge x band = 2 x 1891 x 6 = 22692 features per FC metric
```

Apply fold-local feature selection or PCA.

- [x] **Step 3: Implement baseline models**

Implement:

```text
Logistic Regression L1/L2
SVM Linear
SVM RBF
Random Forest
XGBoost or LightGBM if installed
Gaussian Naive Bayes
kNN
```

If XGBoost/LightGBM is not installed, skip with a recorded reason instead of failing the full run.

- [x] **Step 4: Save patient-level predictions and metrics**

Write:

```text
results/predictions/ml_baseline_loso_predictions.csv
results/metrics/ml_baseline_model_comparison.csv
```

- [x] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_baseline_training.py -v
```

Expected: all tests pass.

## Task 10: Deep Learning Models Without SSL

**Files:**

- Create: `src/eeg_recovery/models/__init__.py`
- Create: `src/eeg_recovery/models/encoders.py`
- Create: `src/eeg_recovery/models/dual_state_model.py`
- Create: `src/eeg_recovery/models/fusion_3d_model.py`
- Create: `src/eeg_recovery/models/clinical_mlp.py`
- Create: `src/eeg_recovery/training/train_supervised.py`
- Create: `src/eeg_recovery/training/schedulers.py`
- Create: `scripts/05_train_supervised_loso.py`
- Create: `tests/test_model_shapes.py`

- [x] **Step 1: Write model shape tests**

Tests must verify:

```text
PSD dual-state input: two tensors shaped 62 x 90
FC dual-state input: two tensors shaped 1891 x 6
3D PSD input: 2 x 62 x 90
3D FC input: 2 x 1891 x 6
output probability shape: batch x 1
```

- [x] **Step 2: Implement lightweight encoders**

Implement:

```text
Shared_PSD_Encoder
Shared_FC_Encoder
ClinicalMLP
```

Keep parameter count small due to 19 supervised patients.

- [x] **Step 3: Implement fusion options**

Implement:

```text
concat fusion
gated EO/EC fusion
```

Gated fusion should expose EO/EC weights for interpretation.

Implemented extension: FC-only training supports `fc-wpli`, `fc-icoh`, and
`fc-both`; multimodal deep learning supports `psd-fc-wpli`, `psd-fc-icoh`,
and `psd-fc-both`, with wPLI and imaginary coherence kept as separate FC
branches.

- [x] **Step 4: Implement supervised LOSO training**

Use:

```text
Binary Cross Entropy
ReduceLROnPlateau
EarlyStopping with restore_best_weights behavior
patient-level mean probability aggregation
```

- [x] **Step 5: Save predictions and metrics**

Write:

```text
results/predictions/dl_loso_predictions.csv
results/metrics/dl_model_comparison.csv
```

- [x] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_model_shapes.py -v
```

Expected: all tests pass.

## Task 11: Self-Supervised Pretraining

**Files:**

- Create: `src/eeg_recovery/models/ssl_model.py`
- Create: `src/eeg_recovery/training/train_ssl.py`
- Create: `scripts/06_train_ssl.py`
- Create: `tests/test_ssl_dataset.py`

- [ ] **Step 1: Write tests for SSL subject exclusion**

Tests must verify that in strict LOSO SSL mode, the current test subject is excluded from that fold's SSL pretraining dataset.

- [ ] **Step 2: Implement SSL datasets**

Support:

```text
baseline patient EO/EC EEG
health EO/EC EEG
incomplete patient EEG where available
ceiling-effect patient EEG where available
```

- [ ] **Step 3: Implement augmentations**

Implement deterministic-seed augmentations:

```text
time crop
Gaussian noise
amplitude scaling
channel dropout
time masking
frequency masking for PSD/time-frequency inputs
```

- [ ] **Step 4: Implement SSL objectives**

Implement:

```text
NT-Xent contrastive loss
masked reconstruction MSE
combined loss with lambda in config
```

- [ ] **Step 5: Implement transfer to supervised model**

Support:

```text
frozen encoder
fine-tuned encoder with lower encoder learning rate
```

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_ssl_dataset.py -v
```

Expected: all tests pass.

## Task 12: Evaluation Metrics And Statistical Tests

**Files:**

- Modify: `src/eeg_recovery/training/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write metric tests**

Tests must cover:

```text
Accuracy
Balanced Accuracy
Sensitivity
Specificity
Precision
F1-score
ROC-AUC
PR-AUC
Confusion matrix
```

- [ ] **Step 2: Implement bootstrap CI**

Use patient-level predictions only. Report 95% CI for key metrics.

- [ ] **Step 3: Implement permutation test**

Use patient-level labels and predictions. Record random seed and number of permutations.

- [ ] **Step 4: Save outputs**

Write:

```text
results/metrics/bootstrap_ci.csv
results/metrics/permutation_test.csv
```

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_metrics.py -v
```

Expected: all tests pass.

## Task 13: Explainability

**Files:**

- Create: `src/eeg_recovery/explainability/__init__.py`
- Create: `src/eeg_recovery/explainability/psd_attribution.py`
- Create: `src/eeg_recovery/explainability/fc_attribution.py`
- Create: `src/eeg_recovery/explainability/clinical_validation.py`
- Create: `src/eeg_recovery/explainability/visualization.py`
- Create: `scripts/07_run_explainability.py`
- Create: `tests/test_explainability_shapes.py`

- [ ] **Step 1: Write attribution shape tests**

Tests must verify:

```text
PSD attribution EO shape: 62 x 90
PSD attribution EC shape: 62 x 90
FC attribution EO shape: 1891 x 6
FC attribution EC shape: 1891 x 6
state importance sums are finite and non-negative when expected
```

- [ ] **Step 2: Implement PSD attribution**

Support:

```text
Integrated Gradients
Gradient SHAP if dependency is available
Permutation importance
```

- [ ] **Step 3: Implement FC attribution**

Support:

```text
edge masking
permutation edge importance
band-level aggregation
```

- [ ] **Step 4: Implement clinical validation**

Correlate selected EEG biomarkers with:

```text
Delta_FMA_obs
Residual
FMA_pre
FMA_post
MBI_pre
duration
```

Use FDR correction for multiple tests.

- [ ] **Step 5: Save outputs**

Write:

```text
results/explainability/state_importance.csv
results/explainability/psd_attribution_summary.csv
results/explainability/fc_attribution_summary.csv
results/explainability/clinical_correlation.csv
results/figures/
```

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_explainability_shapes.py -v
```

Expected: all tests pass.

## Task 14: End-To-End Pipeline And Documentation

**Files:**

- Create: `README.md`
- Create: `scripts/run_all_baselines.py`
- Create: `scripts/run_full_experiment.py`
- Create: `tests/test_pipeline_smoke.py`
- Modify: `docs/task_status.md`

- [ ] **Step 1: Write smoke test**

Smoke test should run a minimal dry-run path on a tiny subset or mocked feature arrays and verify that the pipeline wiring works without touching external raw data.

- [ ] **Step 2: Implement pipeline runners**

Runners should expose clear stages:

```text
prepare-metadata
compute-psd
compute-fc
train-baselines
train-supervised
train-ssl
explain
```

- [ ] **Step 3: Write README**

README must document:

```text
project goal
required external data paths
62-channel fact
EO/EC filename convention
label definition
LOSO leakage rules
how to run each script
where outputs are written
```

- [ ] **Step 4: Run full test suite**

Run:

```powershell
pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Verify project cleanliness**

Run:

```powershell
Get-ChildItem -LiteralPath . -Force
```

Expected: only formal project files and directories are present; no scratch files remain.

## 3. Suggested Subagent Boundaries

Use one fresh subagent per task. Suggested ownership:

| Task | Subagent ownership |
|---|---|
| 1 | Scaffold/config only |
| 2 | Metadata/labels only |
| 3 | EEG index only |
| 4 | EEGLAB IO only |
| 5 | Channel mapping/flip only |
| 6 | PSD only |
| 7 | FC only |
| 8 | LOSO/leakage only |
| 9 | ML baseline only |
| 10 | DL supervised only |
| 11 | SSL only |
| 12 | Metrics/statistics only |
| 13 | Explainability only |
| 14 | Pipeline/README only |

Tasks 2-8 should be completed before model-training tasks. Tasks 9 and 10 may proceed independently after feature outputs are stable. Task 11 depends on EEG IO and dataset indexing. Task 13 depends on at least one trained model.

## 4. Review Gates

After each task:

1. Spec compliance review: confirm the task implements exactly what this plan requires.
2. Code quality review: confirm focused files, readable code, no leakage, no temporary artifacts.
3. Run the task-specific tests.
4. Update `docs/task_status.md`.

Before declaring the project complete:

```powershell
pytest -v
```

Then run at least one complete LOSO baseline and one complete deep learning smoke/real run, depending on available runtime.

## 5. Known Project-Specific Risks

- The design document says 64 channels; actual data has 62 channels.
- Folder names use inconsistent zero padding for subject IDs.
- `FMA_post` must never become a model input.
- FC feature dimension is large; all feature selection must be fold-local.
- Small supervised sample size makes deep learning unstable; report confidence intervals and permutation tests.
- Strict SSL and transductive SSL must be reported separately if both are used.
- MATLAB exists but requires sandbox-external invocation with `-wait -batch`.
