from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md"
OUTPUT = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_structured.md"


JNE_ABSTRACT = """## Abstract

### Objective

Accurate prediction of upper-limb recovery after stroke could support earlier rehabilitation stratification, but small labeled cohorts limit EEG-based prognostic models. This study tested residual-aware self-supervised EEG learning for patient-level proportional-recovery prediction after stroke.

### Approach

We developed a multimodal SSL-CNN integrating baseline resting-state power spectral density and weighted phase-lag index connectivity from eyes-open and eyes-closed EEG before transcranial alternating current stimulation. The supervised cohort included 19 patients with baseline and post-treatment Fugl-Meyer Assessment of the upper extremity scores. Labels were residuals between expected and observed motor improvement. Evaluation used patient-level leave-one-subject-out cross-validation, ten seeds, bootstrap intervals, permutation testing, paired comparisons, ablations, and model explanation.

### Main results

The residual-aware SSL-CNN achieved accuracy of 84.2%, balanced accuracy of 83.3%, ROC-AUC of 0.844, PR-AUC of 0.836, and Brier score of 0.126. It showed numerically higher ROC-AUC and lower Brier score than a PSD+WPLI logistic-regression EEG baseline. Versus a matched no-SSL CNN, hard-label accuracy was unchanged, while ranking and calibration metrics improved directionally. Paired differences were not statistically definitive. Ablations identified residual-aware auxiliary supervision as the most consistent training signal; self-supervised pretraining had no stable independent gain. Explanations localized state- and frequency-specific PSD patterns and beta-band connectivity.

### Significance

Residual-aware self-supervised EEG learning provides a compact framework for using continuous recovery information while retaining binary prognostic inference. The findings are exploratory and require external validation before clinical deployment.
"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("## Abstract")
    end = text.index("## Introduction")
    output = text[:start] + JNE_ABSTRACT + "\n" + text[end:]
    OUTPUT.write_text(output, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
