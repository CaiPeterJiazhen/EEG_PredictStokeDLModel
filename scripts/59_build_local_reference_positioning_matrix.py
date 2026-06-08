from __future__ import annotations

import csv
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(r"C:\Users\HPGZZ\Desktop\预后模型相关文献")
OUTPUT_MD = ROOT / "docs" / "local_reference_positioning_matrix.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "local_reference_positioning_matrix.csv"


MANUAL_RECORDS = {
    "2025 - (PDF) Explainable Artificial Intelligence Model for Stroke Prediction Using EEG Signal.pdf": {
        "citation_key": "Islam et al., 2022",
        "title": "Explainable artificial intelligence model for stroke prediction using EEG signal",
        "doi": "10.3390/s22249859",
        "domain": "stroke diagnosis/risk classification",
        "modality": "EEG",
        "outcome": "stroke prediction/classification, not recovery prognosis",
        "current_status": "not in main manuscript",
        "positioning_decision": "Do not add to main text unless a broader EEG-stroke-AI background paragraph is requested.",
        "support_grade": "background only",
        "reason": "The paper supports feasibility of EEG plus explainable AI for stroke classification, but it does not test post-stroke motor recovery or treatment-response prognosis.",
    },
    "AlArfaj 等 - 2022 - A Deep Learning Model for Stroke Patients’ Motor Function Prediction.pdf": {
        "citation_key": "AlArfaj et al., 2022",
        "title": "A deep learning model for stroke patients' motor function prediction",
        "doi": "10.1155/2022/8645165",
        "domain": "stroke motor-function prediction",
        "modality": "clinical/rehabilitation variables",
        "outcome": "motor function prediction",
        "current_status": "included as reference 24",
        "positioning_decision": "Retain as moderate domain support in Introduction/Discussion.",
        "support_grade": "moderate support",
        "reason": "The task is stroke motor-function prediction, but it is less directly aligned than EEG recovery-response studies.",
    },
    "Enhancing upper limb motor recovery prediction after acute stroke using EEG and subacute data(科研通-ablesci.com).pdf": {
        "citation_key": "Lassi et al., 2026",
        "title": "Enhancing upper limb motor recovery prediction after acute stroke using EEG and subacute data",
        "doi": "10.1063/5.0287165",
        "domain": "stroke upper-limb recovery prediction",
        "modality": "EEG plus subacute clinical data",
        "outcome": "upper-limb motor recovery",
        "current_status": "included as reference 6",
        "positioning_decision": "Retain as one of the closest direct comparators.",
        "support_grade": "strong direct support",
        "reason": "It directly supports the use of EEG-derived features for upper-limb recovery prediction after stroke.",
    },
    "Hasanzadeh 等 - 2024 - Analysis of EEG-derived brain networks for predicting rTMS treatment outcomes in MDD patients.pdf": {
        "citation_key": "Hasanzadeh et al., 2024",
        "title": "Analysis of EEG-derived brain networks for predicting rTMS treatment outcomes in MDD patients",
        "doi": "10.1016/j.bspc.2024.106613",
        "domain": "depression neuromodulation-response prediction",
        "modality": "EEG brain networks",
        "outcome": "rTMS treatment response",
        "current_status": "not in main manuscript",
        "positioning_decision": "Keep as off-domain methods analogue; cite only in an expanded related-work section.",
        "support_grade": "analogical support",
        "reason": "It supports EEG network-based treatment-response prediction, but the disease and intervention differ from stroke tACS rehabilitation.",
    },
    "Li 等 - 2026 - Neurophysiological mechanisms and predictive modeling of SSRI treatment response in depression disor.pdf": {
        "citation_key": "Li et al., 2026",
        "title": "Neurophysiological mechanisms and predictive modeling of SSRI treatment response in depression disorder based on multidimensional EEG features",
        "doi": "10.1016/j.jad.2025.120424",
        "domain": "depression pharmacotherapy-response prediction",
        "modality": "multidimensional EEG features",
        "outcome": "SSRI treatment response",
        "current_status": "not in main manuscript",
        "positioning_decision": "Keep out of main text; use only if reviewers ask for broader EEG treatment-response context.",
        "support_grade": "analogical support",
        "reason": "It is a treatment-response EEG model, but off-domain for stroke motor recovery.",
    },
    "Lin 等 - 2022 - A Transferable Deep Learning Prognosis Model for Predicting Stroke Patients' Recovery in Different R.pdf": {
        "citation_key": "Lin et al., 2022",
        "title": "A transferable deep learning prognosis model for predicting stroke patients' recovery in different rehabilitation trainings",
        "doi": "10.1109/JBHI.2022.3205436",
        "domain": "stroke rehabilitation prognosis",
        "modality": "rehabilitation and clinical modelling",
        "outcome": "recovery under rehabilitation training",
        "current_status": "included as reference 4",
        "positioning_decision": "Retain as direct deep-learning stroke prognosis background.",
        "support_grade": "strong domain support",
        "reason": "It directly motivates treatment-specific recovery prognosis and transferability issues.",
    },
    "Nielsen 等 - 2018 - Prediction of Tissue Outcome and Assessment of Treatment Effect in Acute Ischemic Stroke Using Deep.pdf": {
        "citation_key": "Nielsen et al., 2018",
        "title": "Prediction of tissue outcome and assessment of treatment effect in acute ischemic stroke using deep learning",
        "doi": "10.1161/STROKEAHA.117.019740",
        "domain": "acute ischemic stroke imaging prognosis",
        "modality": "MRI/CT perfusion imaging",
        "outcome": "tissue outcome and treatment effect",
        "current_status": "not in main manuscript",
        "positioning_decision": "Do not add to main text; keep as broad stroke-AI background only.",
        "support_grade": "background only",
        "reason": "It is a stroke prediction paper, but the target is tissue outcome rather than EEG-based motor recovery.",
    },
    "Olbrich 等 - 2026 - Deep learning using electroencephalogram (EEG) data for diagnosing and predicting SSRI response in m.pdf": {
        "citation_key": "Olbrich et al., 2026",
        "title": "Deep learning using electroencephalogram data for diagnosing and predicting SSRI response in major depressive disorder",
        "doi": "10.1038/s43856-026-01394-z",
        "domain": "depression treatment-response prediction",
        "modality": "EEG deep learning",
        "outcome": "SSRI response and diagnosis",
        "current_status": "not in main manuscript",
        "positioning_decision": "Keep out of main text; useful only as high-impact off-domain precedent for EEG treatment-response modelling.",
        "support_grade": "analogical support",
        "reason": "High-impact EEG response-modelling precedent, but unrelated to stroke recovery or tACS.",
    },
    "Prognostic and Monitory EEG-Biomarkers for BCI Upper-Limb Stroke Rehabilitation(科研通-ablesci.com).pdf": {
        "citation_key": "Mane et al., 2019",
        "title": "Prognostic and monitory EEG-biomarkers for BCI upper-limb stroke rehabilitation",
        "doi": "10.1109/TNSRE.2019.2924742",
        "domain": "stroke upper-limb rehabilitation EEG biomarkers",
        "modality": "EEG",
        "outcome": "rehabilitation response and monitoring",
        "current_status": "included as reference 7",
        "positioning_decision": "Retain as direct EEG rehabilitation biomarker support.",
        "support_grade": "strong direct support",
        "reason": "It directly supports EEG biomarkers in upper-limb stroke rehabilitation.",
    },
    "Shahabi 等 - 2023 - Prediction of response to repetitive transcranial magnetic stimulation for major depressive disorder.pdf": {
        "citation_key": "Shahabi et al., 2023",
        "title": "Prediction of response to repetitive transcranial magnetic stimulation for major depressive disorder using raw EEG and hybrid convolutional-recurrent neural networks",
        "doi": "10.1007/s11571-022-09881-4",
        "domain": "depression neuromodulation-response prediction",
        "modality": "raw EEG",
        "outcome": "rTMS response",
        "current_status": "not in main manuscript",
        "positioning_decision": "Keep as off-domain neural-engineering analogue.",
        "support_grade": "analogical support",
        "reason": "It is relevant to EEG prediction of neuromodulation response, but not to stroke motor recovery.",
    },
    "Singh 等 - 2024 - Determining Diagnostic Utility of EEG for Assessing Stroke Severity using Deep Learning Models.pdf": {
        "citation_key": "Singh et al., 2024",
        "title": "Determining diagnostic utility of EEG for assessing stroke severity using deep learning models",
        "doi": "10.1016/j.bea.2024.100121",
        "domain": "stroke EEG severity assessment",
        "modality": "EEG deep learning",
        "outcome": "stroke severity",
        "current_status": "included as reference 25",
        "positioning_decision": "Retain only as partial feasibility support, not as recovery-prognosis evidence.",
        "support_grade": "partial support",
        "reason": "It supports EEG deep-learning feasibility in stroke, but its endpoint is severity assessment rather than recovery.",
    },
    "Tozlu 等 - 2020 - Machine Learning Methods Predict Individual Upper-Limb Motor Impairment Following Therapy in Chronic.pdf": {
        "citation_key": "Tozlu et al., 2020",
        "title": "Machine learning methods predict individual upper-limb motor impairment following therapy in chronic stroke",
        "doi": "10.1177/1545968320909796",
        "domain": "stroke therapy outcome prediction",
        "modality": "clinical and neuroimaging variables",
        "outcome": "upper-limb motor impairment after therapy",
        "current_status": "included as reference 23",
        "positioning_decision": "Retain as strong domain support for individual-level prediction after therapy.",
        "support_grade": "strong domain support",
        "reason": "It directly supports individualized post-therapy upper-limb outcome prediction in stroke.",
    },
    "White 等 - 2024 - Predicting recovery following stroke Deep learning, multimodal data and feature selection using exp.pdf": {
        "citation_key": "White et al., 2024",
        "title": "Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI",
        "doi": "10.1016/j.nicl.2024.103638",
        "domain": "stroke recovery prediction",
        "modality": "multimodal deep learning and explainable AI",
        "outcome": "post-stroke recovery",
        "current_status": "included as reference 5",
        "positioning_decision": "Retain as direct comparator for explainable multimodal stroke recovery prediction.",
        "support_grade": "strong domain support",
        "reason": "It directly supports explainable AI and feature-selection framing for stroke recovery prediction.",
    },
    "Zhao 等 - 2025 - Predicting Treatment Response of Repetitive Transcranial Magnetic Stimulation in Major Depressive Di.pdf": {
        "citation_key": "Zhao et al., 2025",
        "title": "Predicting treatment response of repetitive transcranial magnetic stimulation in major depressive disorder using an explainable machine learning model based on EEG and clinical features",
        "doi": "10.1016/j.bpsc.2025.02.002",
        "domain": "depression neuromodulation-response prediction",
        "modality": "EEG plus clinical features",
        "outcome": "rTMS treatment response",
        "current_status": "not in main manuscript",
        "positioning_decision": "Keep as off-domain analogue; cite only if adding a broad EEG treatment-response paragraph.",
        "support_grade": "analogical support",
        "reason": "It is conceptually close to EEG plus clinical treatment-response prediction, but it differs in disease, target, and intervention.",
    },
}


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(REFERENCE_DIR.rglob("*.pdf")):
        record = MANUAL_RECORDS.get(path.name)
        if record is None:
            continue
        extracted = extract_pdf_probe(path)
        row = {
            "local_file": path.name,
            "pdf_doi_probe": "; ".join(extracted["dois"]),
            "pdf_text_probe": extracted["text_probe"],
            **record,
        }
        rows.append(row)

    fieldnames = [
        "local_file",
        "citation_key",
        "title",
        "doi",
        "pdf_doi_probe",
        "domain",
        "modality",
        "outcome",
        "current_status",
        "support_grade",
        "positioning_decision",
        "reason",
        "pdf_text_probe",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)


def extract_pdf_probe(path: Path) -> dict[str, object]:
    text = ""
    try:
        reader = PdfReader(str(path))
        for page in reader.pages[:3]:
            text += "\n" + (page.extract_text() or "")
    except Exception as exc:
        return {"dois": [], "text_probe": f"PDF extraction failed: {exc}"}
    dois = sorted(set(clean_doi(match) for match in re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)))
    text_probe = " ".join(text.split())[:500]
    return {"dois": dois, "text_probe": text_probe}


def clean_doi(value: str) -> str:
    return value.rstrip(").,;")


def build_markdown(rows: list[dict[str, str]]) -> str:
    included = [row for row in rows if row["current_status"].startswith("included")]
    off_domain = [row for row in rows if "not in main" in row["current_status"]]
    lines = [
        "# Local Prognostic-Model Reference Positioning Matrix",
        "",
        "This matrix aligns the locally supplied prognostic-model papers with the current manuscript. It is intentionally conservative: a paper is kept out of the main manuscript when it is off-domain, supports only general AI feasibility, or could encourage overclaiming.",
        "",
        "## Summary",
        "",
        f"- Local PDF files screened: {len(rows)}.",
        f"- References already used in the manuscript: {len(included)}.",
        f"- References kept as off-domain or background-only material: {len(off_domain)}.",
        "- The current manuscript already cites the closest stroke recovery, upper-limb rehabilitation, EEG-biomarker, and explainable-AI comparators.",
        "- Psychiatric rTMS/SSRI response papers are useful neural-engineering analogues but should not be used as direct evidence for post-stroke motor recovery.",
        "",
        "## Positioning Matrix",
        "",
        "| Reference | Domain | Modality | Outcome | Current status | Support grade | Decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {citation_key} | {domain} | {modality} | {outcome} | {current_status} | {support_grade} | {positioning_decision} |".format(
                **{key: md_escape(str(value)) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Manuscript Implications",
            "",
            "1. Keep the main related-work thread centered on stroke recovery prediction, upper-limb rehabilitation, EEG biomarkers, and explainable patient-level modelling.",
            "2. Do not cite off-domain depression rTMS/SSRI papers as evidence that EEG predicts stroke recovery. They can be mentioned only as broader treatment-response modelling analogues if a reviewer requests a wider neural-engineering context.",
            "3. Retain the conservative Discussion language: the study is exploratory, lacks external validation, and does not establish EEG incremental value over clinical variables.",
            "4. If the final target changes from JNE to a broader machine-learning or neuromodulation journal, add one carefully hedged sentence about EEG treatment-response modelling across neurological and psychiatric interventions, citing the off-domain papers only as analogues.",
            "",
            "## Candidate Expanded Sentence If Needed",
            "",
            "Recent EEG treatment-response studies in non-stroke neuromodulation and pharmacotherapy similarly indicate that neural-network and graph-based EEG features can support response stratification, but those studies address different diseases and interventions and therefore provide methodological analogy rather than direct evidence for post-stroke motor recovery.",
            "",
            "## Guardrails",
            "",
            "- Use Lassi et al., Mane et al., Lin et al., White et al., Tozlu et al., AlArfaj et al., and Singh et al. for stroke-specific context, with Singh et al. limited to EEG deep-learning feasibility.",
            "- Use Hasanzadeh et al., Shahabi et al., Zhao et al., Li et al., and Olbrich et al. only as off-domain analogues.",
            "- Use Nielsen et al. and Islam et al. only for broad stroke-AI/EEG background, not recovery-response claims.",
            "- Do not expand the reference list unless the manuscript gains a sentence that the added paper directly supports.",
            "",
            "## DOI Probe",
            "",
            "| Reference | DOI used | DOI extracted from local PDF |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(f"| {md_escape(row['citation_key'])} | `{row['doi']}` | `{md_escape(row['pdf_doi_probe'])}` |")
    lines.append("")
    return "\n".join(lines)


def md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
