# Local Reference Audit

This audit reviews the papers stored in the local prognostic-model reference folder and records how each paper should be used in the manuscript. Metadata were checked from DOI/Crossref where available and from PDF text extraction for local filenames. The purpose is citation discipline: a paper is added to the manuscript only when it directly supports a claim.

## References Added To The Manuscript

| Reference | DOI | Manuscript use | Support grade |
|---|---|---|---|
| Tozlu et al., 2020, Machine learning methods predict individual upper-limb motor impairment following therapy in chronic stroke | 10.1177/1545968320909796 | Supports prior machine-learning work for individual upper-limb motor impairment prediction after stroke therapy. | Strong domain support |
| AlArfaj et al., 2022, A deep learning model for stroke patients' motor function prediction | 10.1155/2022/8645165 | Supports the use of deep learning for stroke motor-function prediction. | Moderate domain support |
| Singh et al., 2024, Determining diagnostic utility of EEG for assessing stroke severity using deep learning models | 10.1016/j.bea.2024.100121 | Supports feasibility of EEG deep-learning models in stroke assessment, but not treatment-response prognosis. | Partial support |

## References Already Included

| Reference | DOI | Current manuscript role |
|---|---|---|
| Lin et al., 2022, transferable deep-learning prognosis model for stroke rehabilitation recovery | 10.1109/JBHI.2022.3205436 | Deep-learning stroke recovery prediction background |
| White et al., 2024, deep learning and explainable multimodal prediction of stroke recovery | 10.1016/j.nicl.2024.103638 | Explainable multimodal stroke recovery prediction background |
| Lassi et al., 2026, EEG and subacute data for upper-limb motor recovery prediction | 10.1063/5.0287165 | Direct EEG-plus-clinical upper-limb recovery prediction support |
| Mane et al., 2019, prognostic and monitory EEG biomarkers for BCI upper-limb stroke rehabilitation | 10.1109/TNSRE.2019.2924742 | EEG biomarker and rehabilitation-response support |

## Useful But Not Added To The Main Manuscript

| Reference | DOI | Reason for not adding now |
|---|---|---|
| Hasanzadeh et al., 2024, EEG-derived brain networks for predicting rTMS outcomes in major depressive disorder | 10.1016/j.bspc.2024.106613 | Useful analogue for EEG network-based treatment-response prediction, but disease and intervention differ from stroke tACS rehabilitation. |
| Shahabi et al., 2023, raw EEG and hybrid convolutional-recurrent networks for rTMS response in major depressive disorder | 10.1007/s11571-022-09881-4 | Useful neuromodulation-response analogue, but off-domain for the core stroke claims. |
| Zhao et al., 2025, explainable EEG/clinical machine learning for rTMS response in major depressive disorder | 10.1016/j.bpsc.2025.02.002 | Strong methods analogue for explainable EEG treatment-response prediction, but off-domain. |
| Olbrich et al., 2026, EEG deep learning for SSRI response in major depressive disorder | 10.1038/s43856-026-01394-z | Demonstrates high-impact EEG treatment-response modelling, but unrelated to stroke recovery. |
| Nielsen et al., 2018, deep learning for acute ischemic stroke tissue outcome | 10.1161/STROKEAHA.117.019740 | Stroke prediction background, but tissue outcome and imaging task do not directly support EEG rehabilitation prognosis. |
| Explainable artificial intelligence model for stroke prediction using EEG signal | 10.3390/s22249859 | Stroke/EEG classification context, but diagnosis/stroke prediction is not the present recovery-response task. |

## Citation Guardrails

- Do not use non-stroke psychiatric treatment-response papers as direct support for post-stroke motor recovery claims.
- Do not cite EEG stroke-severity classification as if it validated EEG prediction of rehabilitation response.
- Keep the main novelty claim narrow: residual-aware patient-level EEG modelling for pilot proportional-recovery prediction after tACS-context rehabilitation.
- If the target journal requests a broader related-work section, the off-domain neuromodulation-response papers can be discussed in one carefully hedged sentence.
