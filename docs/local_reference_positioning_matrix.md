# Local Prognostic-Model Reference Positioning Matrix

This matrix aligns the locally supplied prognostic-model papers with the current manuscript. It is intentionally conservative: a paper is kept out of the main manuscript when it is off-domain, supports only general AI feasibility, or could encourage overclaiming.

## Summary

- Local PDF files screened: 14.
- References already used in the manuscript: 7.
- References kept as off-domain or background-only material: 7.
- The current manuscript already cites the closest stroke recovery, upper-limb rehabilitation, EEG-biomarker, and explainable-AI comparators.
- Psychiatric rTMS/SSRI response papers are useful neural-engineering analogues but should not be used as direct evidence for post-stroke motor recovery.

## Positioning Matrix

| Reference | Domain | Modality | Outcome | Current status | Support grade | Decision |
|---|---|---|---|---|---|---|
| Islam et al., 2022 | stroke diagnosis/risk classification | EEG | stroke prediction/classification, not recovery prognosis | not in main manuscript | background only | Do not add to main text unless a broader EEG-stroke-AI background paragraph is requested. |
| AlArfaj et al., 2022 | stroke motor-function prediction | clinical/rehabilitation variables | motor function prediction | included as reference 24 | moderate support | Retain as moderate domain support in Introduction/Discussion. |
| Lassi et al., 2026 | stroke upper-limb recovery prediction | EEG plus subacute clinical data | upper-limb motor recovery | included as reference 6 | strong direct support | Retain as one of the closest direct comparators. |
| Hasanzadeh et al., 2024 | depression neuromodulation-response prediction | EEG brain networks | rTMS treatment response | not in main manuscript | analogical support | Keep as off-domain methods analogue; cite only in an expanded related-work section. |
| Li et al., 2026 | depression pharmacotherapy-response prediction | multidimensional EEG features | SSRI treatment response | not in main manuscript | analogical support | Keep out of main text; use only if reviewers ask for broader EEG treatment-response context. |
| Lin et al., 2022 | stroke rehabilitation prognosis | rehabilitation and clinical modelling | recovery under rehabilitation training | included as reference 4 | strong domain support | Retain as direct deep-learning stroke prognosis background. |
| Nielsen et al., 2018 | acute ischemic stroke imaging prognosis | MRI/CT perfusion imaging | tissue outcome and treatment effect | not in main manuscript | background only | Do not add to main text; keep as broad stroke-AI background only. |
| Olbrich et al., 2026 | depression treatment-response prediction | EEG deep learning | SSRI response and diagnosis | not in main manuscript | analogical support | Keep out of main text; useful only as high-impact off-domain precedent for EEG treatment-response modelling. |
| Mane et al., 2019 | stroke upper-limb rehabilitation EEG biomarkers | EEG | rehabilitation response and monitoring | included as reference 7 | strong direct support | Retain as direct EEG rehabilitation biomarker support. |
| Shahabi et al., 2023 | depression neuromodulation-response prediction | raw EEG | rTMS response | not in main manuscript | analogical support | Keep as off-domain neural-engineering analogue. |
| Singh et al., 2024 | stroke EEG severity assessment | EEG deep learning | stroke severity | included as reference 25 | partial support | Retain only as partial feasibility support, not as recovery-prognosis evidence. |
| Tozlu et al., 2020 | stroke therapy outcome prediction | clinical and neuroimaging variables | upper-limb motor impairment after therapy | included as reference 23 | strong domain support | Retain as strong domain support for individual-level prediction after therapy. |
| White et al., 2024 | stroke recovery prediction | multimodal deep learning and explainable AI | post-stroke recovery | included as reference 5 | strong domain support | Retain as direct comparator for explainable multimodal stroke recovery prediction. |
| Zhao et al., 2025 | depression neuromodulation-response prediction | EEG plus clinical features | rTMS treatment response | not in main manuscript | analogical support | Keep as off-domain analogue; cite only if adding a broad EEG treatment-response paragraph. |

## Manuscript Implications

1. Keep the main related-work thread centered on stroke recovery prediction, upper-limb rehabilitation, EEG biomarkers, and explainable patient-level modelling.
2. Do not cite off-domain depression rTMS/SSRI papers as evidence that EEG predicts stroke recovery. They can be mentioned only as broader treatment-response modelling analogues if a reviewer requests a wider neural-engineering context.
3. Retain the conservative Discussion language: the study is exploratory, lacks external validation, and does not establish EEG incremental value over clinical variables.
4. If the final target changes from JNE to a broader machine-learning or neuromodulation journal, add one carefully hedged sentence about EEG treatment-response modelling across neurological and psychiatric interventions, citing the off-domain papers only as analogues.

## Candidate Expanded Sentence If Needed

Recent EEG treatment-response studies in non-stroke neuromodulation and pharmacotherapy similarly indicate that neural-network and graph-based EEG features can support response stratification, but those studies address different diseases and interventions and therefore provide methodological analogy rather than direct evidence for post-stroke motor recovery.

## Guardrails

- Use Lassi et al., Mane et al., Lin et al., White et al., Tozlu et al., AlArfaj et al., and Singh et al. for stroke-specific context, with Singh et al. limited to EEG deep-learning feasibility.
- Use Hasanzadeh et al., Shahabi et al., Zhao et al., Li et al., and Olbrich et al. only as off-domain analogues.
- Use Nielsen et al. and Islam et al. only for broad stroke-AI/EEG background, not recovery-response claims.
- Do not expand the reference list unless the manuscript gains a sentence that the added paper directly supports.

## DOI Probe

| Reference | DOI used | DOI extracted from local PDF |
|---|---|---|
| Islam et al., 2022 | `10.3390/s22249859` | `10.3390/s22249859` |
| AlArfaj et al., 2022 | `10.1155/2022/8645165` | `10.1155/2022/8645165` |
| Lassi et al., 2026 | `10.1063/5.0287165` | `10.1063/5.0287165` |
| Hasanzadeh et al., 2024 | `10.1016/j.bspc.2024.106613` | `10.1016/j.bspc.2024.106613` |
| Li et al., 2026 | `10.1016/j.jad.2025.120424` | `10.1016/j.jad.2025.120424` |
| Lin et al., 2022 | `10.1109/JBHI.2022.3205436` | `10.1109/JBHI.2022.3205436` |
| Nielsen et al., 2018 | `10.1161/STROKEAHA.117.019740` | `10.1161/STROKEAHA; 10.1161/STROKEAHA.117.019740` |
| Olbrich et al., 2026 | `10.1038/s43856-026-01394-z` | `10.1038/s43856-026-01394-z; 10.60955/mnwq-sq07` |
| Mane et al., 2019 | `10.1109/TNSRE.2019.2924742` | `10.1109/TNSRE.2019.2924742` |
| Shahabi et al., 2023 | `10.1007/s11571-022-09881-4` | `10.1007/s11571-022-09881-4` |
| Singh et al., 2024 | `10.1016/j.bea.2024.100121` | `10.1016/j.bea.2024.100121` |
| Tozlu et al., 2020 | `10.1177/1545968320909796` | `10.1177/1545968320909796` |
| White et al., 2024 | `10.1016/j.nicl.2024.103638` | `10.1016/j.nicl.2024.103638` |
| Zhao et al., 2025 | `10.1016/j.bpsc.2025.02.002` | `10.1016/j.bpsc.2025.02.002` |
