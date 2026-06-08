# Manuscript Citation Map

This audit maps the main citable claims in `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md` to the current reference list. It is intended for manuscript review, not as a replacement for final journal reference formatting.

| Manuscript claim area | Current citations | Support grade | Notes |
|---|---|---|---|
| Proportional recovery varies across patients and relates to baseline impairment/corticomotor integrity | [1-3] | Strong support | These are core proportional-recovery and PREP2 references. |
| EEG is a scalable neurophysiological marker for post-stroke prognosis | [7-9] | Strong support | EEG biomarkers, BCI rehabilitation markers, and beta oscillation recovery evidence. |
| WPLI is suitable for reducing zero-lag/volume-conduction effects in EEG connectivity | [10] | Strong support | Methodological WPLI reference. |
| Machine learning, deep learning, and explainable AI have been applied to stroke recovery/prognosis or motor-function prediction | [4-6,23,24] | Strong support | Includes transferable DL prognosis, multimodal explainable prediction, therapy-related upper-limb impairment prediction, and motor-function prediction studies. |
| EEG deep-learning models have been tested for stroke-severity assessment | [25] | Partial support | Supports broader EEG/DL feasibility in stroke assessment, but not treatment-response prognosis. |
| Resting-state connectivity can predict motor recovery changes after stroke | [16] | Strong support | Nature-family Scientific Reports article; useful CNS/Nature-family support. |
| tACS can modulate post-stroke motor-network activity with frequency-specific effects | [22] | Strong support | Supports the tACS context and 10/20 Hz neuromodulation background; it does not validate this study's clinical protocol. |
| MNE-Python was used or is appropriate for EEG/MEG processing and visualization | [11,12] | Strong support | Standard MNE references. |
| Prediction-model reporting should include transparent validation, calibration, bias, and AI reporting considerations | [13-15] | Strong support | TRIPOD, TRIPOD+AI, and PROBAST. |
| Barlow Twins supports redundancy-reduction self-supervised learning | [17] | Strong support | Method reference for SSL objective. |
| Integrated gradients and SmoothGrad support model explanation workflow | [18,19] | Strong support | Attribution-method references. |
| PyTorch and scikit-learn support software methods | [20,21] | Strong support | Software citations. |

## References Requiring Author Review

- Reference 6 is a 2026 APL Bioengineering paper extracted from the local reference folder and included because it directly matches EEG plus subacute data for upper-limb recovery prediction. Confirm that the target journal accepts 2026 online/in-press citation details and that the metadata are final.
- Reference 16 is Nature-family support for connectivity and motor recovery. It is useful for the requested Nature/CNS-leaning citation layer, but it should not be overstated as direct support for this paper's exact WPLI/SSL-CNN pipeline.
- Reference 22 supports tACS stroke neuromodulation background, not the exact therapeutic efficacy of this project's 14-session intervention protocol.
- References 23 and 24 strengthen the local related-work layer from the author-provided reference folder; they support prior machine-learning/deep-learning motor-function prediction after stroke but do not validate the present tACS-response endpoint.
- Reference 25 is useful for EEG deep-learning feasibility in stroke assessment. It should not be cited as direct evidence for predicting rehabilitation response.
- Clinical-only baseline interpretation in the manuscript is supported primarily by the current project data and by proportional-recovery/PREP2 literature [1-3]. Do not add a citation that claims EEG adds incremental clinical value unless a future analysis proves that result.

## Missing Citation Opportunities

Add or verify citations if the final Methods include these details:

- Stroke EEG preprocessing choices, including filtering, artifact rejection, epoch length, and spectral estimation method.
- Calibration metrics or Brier score reporting if the target journal asks for a specific prediction-model methods citation.
- Any public dataset or reused external resource, if added later.
