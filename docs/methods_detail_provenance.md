# Methods Detail Provenance

This file records which project files support the Methods details added to the manuscript. It is an internal audit trail for manuscript preparation.

| Methods detail | Evidence file | Evidence status |
|---|---|---|
| Study uses baseline EEG to predict post-tACS FMA-UE proportional recovery | `tacs_eeg_proportional_recovery_project_design.md`; `docs/project_context.md` | Project design and context |
| tACS target and dose: contralateral M1, C3/C4 by affected hand, 20 Hz, 1000 microampere, 20 min, 14 daily sessions over 2 weeks | `tacs_eeg_proportional_recovery_project_design.md` | Project design |
| Supervised cohort has 19 labelled patients | `docs/project_context.md`; `docs/cohort_characteristics.md`; `results/tables/table1_cohort_characteristics.csv` | Confirmed current output |
| All-patient EEG cohort has 28 patients, 9 non-supervised patients | `docs/cohort_characteristics.md` | Confirmed current output |
| FMA-UE endpoint formula: predicted improvement = 0.7 x (66 - baseline FMA-UE) | `docs/project_context.md`; `src/eeg_recovery/metadata/labels.py` | Confirmed in design and code |
| Residual = predicted improvement - observed improvement; median residual threshold = 1.5 | `docs/project_context.md`; `src/eeg_recovery/metadata/labels.py`; `docs/cohort_characteristics.md` | Confirmed in design, code, and output |
| Actual retained EEG channels are 62 after removing M1/M2 | `docs/project_context.md`; `src/eeg_recovery/channels/mapping.py`; `configs/channel_mapping.yaml` | Confirmed in context, code, and config |
| Sampling rate is 250 Hz in indexed `.set` files | `docs/project_context.md` | Confirmed in project context |
| Baseline `*1.set` and `*2.set` map to EO and EC | `docs/project_context.md` | Confirmed in project context |
| Supervised baseline EO/EC recording duration averaged 188.4 s and ranged from 101.0 to 247.8 s across 38 files | `results/tables/eeg_recording_metadata_audit.csv`; `docs/eeg_metadata_audit.md` | Confirmed from EEGLAB metadata audit |
| Project pipeline starts from preprocessed EEGLAB `.set/.fdt` files and reads `.set` metadata with `scipy.io.loadmat` plus float32 `.fdt` data as channels by samples | `src/eeg_recovery/io/eeglab.py`; `docs/methods_gap_resolution_from_project_files.md` | Confirmed in code |
| No additional temporal filtering, artifact rejection, channel interpolation, or bad-channel removal is performed inside the manuscript feature scripts after loading the provided preprocessed files | `src/eeg_recovery/features/psd.py`; `src/eeg_recovery/features/connectivity.py`; `docs/methods_gap_resolution_from_project_files.md` | Confirmed for current feature pipeline |
| PSD uses Welch with Hann window, 0.5 Hz resolution, 50% overlap, density scaling, constant detrending, 0.5-45 Hz bins | `src/eeg_recovery/features/psd.py` | Confirmed in code |
| WPLI and imaginary coherence use STFT with 2 s Hann windows and 50% overlap | `src/eeg_recovery/features/connectivity.py` | Confirmed in code |
| Connectivity bands are delta 1-3 Hz, theta 4-7 Hz, alpha 8-13 Hz, beta-low 13-18 Hz, beta-medium 18-21 Hz, beta-high 21-30 Hz | `src/eeg_recovery/features/connectivity.py`; `docs/project_context.md` | Confirmed in code and context |
| Affected-side alignment mirrors left-hand impairment to a common right-hand/C3 convention before PSD and FC computation | `src/eeg_recovery/channels/hemisphere_flip.py`; `docs/project_context.md`; `configs/channel_mapping.yaml` | Confirmed in code, context, and config |
| Patient-level LOSO is the validation unit; segment and seed rows are not independent patients | `docs/statistical_validation_summary.md`; `docs/project_context.md` | Confirmed in statistical summary and context |

## Details Still Requiring Author Confirmation

- Ethics approval institution and approval number.
- Informed consent language and data-sharing permissions.
- Recruitment dates and clinical inclusion/exclusion criteria.
- Stroke subtype, lesion-side definition, and timing from stroke to EEG/tACS.
- EEG acquisition hardware, electrode cap/montage description, online reference, preprocessing filters, re-reference, artifact rejection, and bad-channel handling before the current preprocessed `.set/.fdt` files.
- Whether standardized conventional rehabilitation was delivered concurrently with tACS.
