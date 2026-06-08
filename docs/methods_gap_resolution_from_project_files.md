# Methods Gap Resolution From Project Files

This audit records what could and could not be resolved from the current project files during manuscript preparation. It is designed to prevent unsupported Methods statements from entering the paper.

## Resolved From Current Files

| Methods item | Current evidence | Manuscript-ready wording |
|---|---|---|
| Analysis starts from preprocessed EEGLAB files | `docs/project_context.md`; `src/eeg_recovery/io/eeglab.py` | Analyses used project-provided preprocessed EEGLAB `.set/.fdt` files. |
| EEGLAB loading route | `src/eeg_recovery/io/eeglab.py` | `.set` metadata were read with `scipy.io.loadmat`; companion `.fdt` float32 data were loaded as channels by samples. |
| Retained channels | `docs/project_context.md`; `configs/channel_mapping.yaml`; `src/eeg_recovery/channels/mapping.py` | The retained analysis montage contained 62 channels after M1/M2 removal. |
| Data shape validation | `src/eeg_recovery/features/psd.py`; `src/eeg_recovery/features/connectivity.py` | Feature scripts required finite 62-channel continuous arrays and fixed channel order. |
| Sampling rate and recording duration | `docs/eeg_metadata_audit.md`; `results/tables/eeg_recording_metadata_audit.csv` | Supervised baseline files were sampled at 250 Hz; baseline EO/EC duration averaged 188.4 s, range 101.0-247.8 s. |
| Additional filtering/artifact handling inside the manuscript pipeline | `src/eeg_recovery/features/psd.py`; `src/eeg_recovery/features/connectivity.py` | No additional temporal filtering, artifact rejection, channel interpolation, or bad-channel removal was performed by the manuscript feature scripts after loading the provided preprocessed files. |
| PSD method | `src/eeg_recovery/features/psd.py` | Welch PSD used a Hann window, 0.5 Hz resolution, 50% overlap, density scaling, constant detrending, and 0.5-45 Hz output bins. |
| Connectivity method | `src/eeg_recovery/features/connectivity.py` | Connectivity used STFT with 2 s Hann windows, 50% overlap, constant detrending, WPLI and imaginary coherence, 1,891 upper-triangle edges, and six frequency bands. |
| Affected-side alignment | `src/eeg_recovery/channels/hemisphere_flip.py`; `configs/channel_mapping.yaml` | Left-hand impairment was mirrored to a common right-hand/C3 stimulation convention before PSD and FC extraction. |
| Label formula | `src/eeg_recovery/metadata/labels.py` | Labels used the supervised-cohort median residual from proportional-recovery expected and observed FMA-UE gain. |
| Residual-aware targets | `src/eeg_recovery/training/residual_targets.py`; `src/eeg_recovery/training/residual_aware_losses.py` | Auxiliary residual targets used signed distance from the 1.5-point median residual threshold, fold-local standardization, soft labels, regression, and pairwise ranking losses. |

## Still Unresolved

| Missing field | Why it cannot be inferred safely | Required author action |
|---|---|---|
| Ethics approval institution and approval number | No ethics/IRB field was found in project text files or workbook structure audit. | Provide institution, approval number, consent route, and trial/registry status if applicable. |
| Informed consent and data-sharing language | Current files do not contain consent text or secondary-use permission. | Provide original consent/data-sharing wording or approved access restriction text. |
| EEG acquisition hardware and electrode cap | The repository starts from preprocessed EEGLAB files and a standard coordinate file, not raw acquisition protocol. | Provide amplifier/system, cap/montage, online reference, ground, impedance criterion, and acquisition environment. |
| Raw preprocessing protocol before provided `.set/.fdt` files | The current code only loads preprocessed files and computes features. | Provide filter settings, re-reference scheme, artifact rejection/ICA/manual cleaning, bad-channel interpolation, and epoching or continuous-data handling before export. |
| Recruitment dates and eligibility criteria | Workbook structure contains limited timing/protocol columns but no protocol text. | Provide recruitment period, inclusion/exclusion criteria, stroke type, lesion criteria, and timing from stroke to EEG/tACS. |
| Concurrent conventional rehabilitation | The project design documents the tACS protocol but not whether standardized rehabilitation was delivered concurrently. | Confirm whether tACS was standalone or combined with conventional therapy and provide dose/frequency if combined. |

## Manuscript Guardrail

The Methods section may describe the current feature-extraction pipeline precisely, but it must not imply that raw EEG acquisition or preprocessing steps are verified until the author supplies the protocol. The safest wording is to state that analyses began from project-provided preprocessed EEGLAB files and that acquisition and raw preprocessing details remain author-confirmed fields before submission.
