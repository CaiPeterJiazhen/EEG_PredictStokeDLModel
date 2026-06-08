# Target Journal Strategy And Submission Fit Matrix

Source check date: 2026-06-02.

This strategy uses public author-facing pages and stable indexing records available at the time of checking. CAS partition, JCR quartile, institutional APC agreements, and local reimbursement eligibility must still be confirmed through the author's institution before submission.

## Executive Recommendation

The best first target remains **Journal of Neural Engineering (JNE)**. It is the closest match to the paper's real contribution: patient-level EEG neural-signal processing, neuro-rehabilitation, neuromodulation context, and transparent prediction-model evaluation. The strongest practical backup is **IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)** if the author prefers a rehabilitation-engineering audience and is willing to convert to IEEE style.

The Nature Portfolio option, **Communications Medicine**, is a stretch target. The current package is strong on transparency, source data, code availability drafts, visual QA, and reporting audits, but the supervised cohort is still n=19 with no external validation. A Nature-family submission should wait until ethics/data-sharing fields are complete and the cover letter frames this as a rigorously audited pilot/proof-of-concept rather than a clinically deployable model.

## Recommended Submission Order

1. **Journal of Neural Engineering**: best fit and best current manuscript readiness. Use the prepared JNE structured-abstract version.
2. **IEEE TNSRE**: strong technical and rehabilitation-engineering fit, but requires IEEE conversion.
3. **NeuroImage: Clinical**: viable if framed as clinical neurophysiology and biomarker-based therapeutic-response prediction.
4. **Communications Medicine**: high-risk Nature Portfolio stretch target after author-supplied ethics/data/code sharing details are complete.
5. **Brain Stimulation**: high-impact but high-risk stretch target only if the tACS protocol, safety, blinding/rehabilitation details, and treatment-response framing are fully supplied.

## Fit Matrix

| Journal | Current source-checked requirements or scope | Fit for this manuscript | Required actions before submission | Risk |
|---|---|---|---|---|
| Journal of Neural Engineering | IOP states that JNE covers neural engineering across experimental, computational, clinical, applied, neuromodulation, neuro-rehabilitation, neural imaging, neural signal processing, computational neuroscience, and translational neuroscience. Papers are normally <=12,000 words or 14 journal pages. Abstract headings must be Objective, Approach, Main results, and Significance. Human investigations must follow Declaration of Helsinki principles; clinical trials should quote registration number at the end of the abstract. Hybrid OA is optional; subscription publication is free, and the listed OA APC is GBP 2530 / EUR 2905 / USD 3490. Source: [IOP JNE author-facing page](https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/). | High. The prepared JNE variant already uses the required structured abstract. The paper sits directly in neural signal processing and neuro-rehabilitation. | Fill ethics/Helsinki statement, trial registration status or reason not registered, conflict-of-interest statement, author list/affiliations, and final data/code availability. Keep claims conservative because JNE explicitly cautions that modest classification gains on small datasets need robust comparisons, validation, and insight. | Moderate. The paper is well aligned, but n=19 and no external validation remain central risks. |
| IEEE Transactions on Neural Systems and Rehabilitation Engineering | DOAJ lists TNSRE as an IEEE/EMBS open-access journal with biomedical engineering, human performance measurement, and rehabilitation-engineering keywords; it reports APC up to USD 2160 and OA since 2021. Source: [DOAJ TNSRE record](https://doaj.org/toc/1558-0210). | High. The model is an EEG rehabilitation-engineering prognosis pipeline with leakage-controlled LOSO, CNN/SSL, residual-aware supervision, and source-data reproducibility. | Convert to IEEE format, tighten figures/tables for IEEE style, add explicit code/data repository information, and emphasize engineering novelty over clinical efficacy. Verify final author instructions inside the IEEE/EMBS submission portal because IEEE pages can be hard to crawl and are submission-system dependent. | Moderate. Engineering reviewers may demand stronger external validation, ablation logic, and reproducibility detail. |
| NeuroImage: Clinical | ScienceDirect describes NeuroImage: Clinical as a NeuroImage companion title focused on pathology, abnormal development, and biomarker usage. Elsevier's guide notes consistent references are acceptable at submission, DOIs are recommended, and OA APC responsibility applies if accepted. Source: [ScienceDirect NeuroImage guide mentioning NeuroImage: Clinical](https://www.sciencedirect.com/journal/neuroimage/publish/guide-for-authors). | Medium-high. The EEG biomarker and clinical neurophysiology story fits, but the study is not centered on MRI/fMRI and may need clearer clinical-neuroscience positioning. | Retitle/reframe around baseline EEG biomarkers of motor recovery/tACS response. Strengthen the clinical neurophysiology rationale, repository link, acquisition details, and limitations. | Moderate-high. Small sample and no external validation are likely major reviewer concerns. |
| Communications Medicine | Nature Portfolio requires a data availability statement for original research and requires custom code central to claims to be available to editors/referees on request, with a Code availability section. Supplementary Information is sent to referees and should include data needed to evaluate claims not deposited publicly. It has no submission/page charges, but accepted papers require an APC. Current APC listed: GBP 2850 / USD 3890 / EUR 3290. Sources: [submission guidelines](https://www.nature.com/commsmed/submit/submission-guidelines) and [APC page](https://www.nature.com/commsmed/open-access). | Medium. The package now has Nature-style text, source-data workbook, code/data drafts, supplementary information, citation/metadata audits, and visual QA. | Complete ethics, consent, data-access route, repository DOI, code archive, reporting summary/checklist, competing interests, funding, author contributions, and target-journal figure/data requirements. | High. The n=19 pilot design and no external validation make this a stretch target unless the transparency and clinical need are compelling. |
| Brain Stimulation | ScienceDirect lists Brain Stimulation as open access with CiteScore 12.4 and Impact Factor 8.4 on the page checked. Its scope is basic, translational, and clinical neuromodulation, including noninvasive/invasive techniques that alter brain function. Original research has a 4,000-word body limit and a structured abstract up to 250 words. The guide reports strict screening, including about 70% initial rejection and about 10% acceptance. Source: [Brain Stimulation Guide for Authors](https://www.sciencedirect.com/journal/brain-stimulation/publish/guide-for-authors). | Medium-low for the current draft. The tACS context is relevant, but the manuscript is primarily an EEG prediction-model paper rather than a definitive neuromodulation efficacy/mechanism trial. | Only target this after fully documenting tACS montage/dose/device, concurrent rehabilitation, safety/adverse events, blinding or lack thereof, and treatment-response interpretation. Compress main text to 4,000 words and use a 250-word structured abstract. | High. Strong journal metrics but likely desk-rejection risk unless the tACS intervention and safety/protocol story are complete. |

## Target-Specific Checklist

### Journal of Neural Engineering

- Use `docs/manuscript_residual_aware_ssl_cnn_jne_structured.md` and `output/doc/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx`.
- Keep the four abstract headings exactly: Objective, Approach, Main results, Significance.
- Add Declaration of Helsinki/ethics approval statement and trial registration status.
- Add conflicts of interest, funding, author contributions, acknowledgements, and final data/code availability.
- In the cover letter, emphasize leakage-controlled patient-level evaluation, residual-aware supervision, source-data availability, and visual/metadata QA.
- Preserve the conservative limitation language about n=19, no external validation, and no proven EEG incremental value over clinical variables.

### IEEE TNSRE

- Convert the clean manuscript to IEEE template and two-column figure/table flow.
- Move wide tables to supplementary material.
- Recast the novelty as a reproducible rehabilitation-engineering pipeline: affected-side alignment, PSD/WPLI multimodal EEG, residual-aware auxiliary heads, LOSO evaluation, source-data workbook, visual QA, and model explanation.
- Check current IEEE/EMBS instructions and page/figure limits at submission time.

### NeuroImage: Clinical

- Reframe around clinical neurophysiology and baseline EEG biomarkers.
- Use the visual explanation figures to argue biological plausibility, but keep them hypothesis-generating.
- Strengthen data sharing and preprocessing provenance before submission.
- Avoid claiming tACS efficacy or EEG incremental clinical value.

### Communications Medicine

- Treat as a stretch target after all author-supplied fields are complete.
- Ensure Data availability identifies the minimum dataset needed to interpret, verify, and extend the claims.
- Make custom code available to editors/referees and provide a Code availability section.
- Complete Nature reporting summary/checklists if the manuscript is sent for review.
- Keep all claims pilot/exploratory unless an external cohort becomes available.

### Brain Stimulation

- Only target if the intervention section becomes much stronger.
- Add tACS device, electrode montage/size, current density if available, safety/adverse events, concomitant rehabilitation, blinding details, and trial registration.
- Reduce to 4,000-word body and 250-word structured abstract.
- Reframe as tACS response stratification rather than a general EEG ML paper.

## CAS/JCR/Ranking Handling

- Do not state "CAS Q1" in the manuscript or cover letter until confirmed through the author's institution or an official CAS/JCR tool.
- Public pages can support journal scope, article type, APC, and author instruction decisions, but CAS partition is a separate institutional bibliometric requirement.
- Based on public fit and likely prestige, the first practical target remains JNE, not because it is guaranteed CAS Q1 here, but because it is the best alignment between the study's methods, audience, and current evidence strength.
- If institutional CAS Q1 status is mandatory, verify JNE, TNSRE, NeuroImage: Clinical, Communications Medicine, and Brain Stimulation in the same official partition year before choosing the target.

## Prepared Target-Specific Files

- `docs/manuscript_residual_aware_ssl_cnn_jne_structured.md`: JNE structured-abstract manuscript variant.
- `output/doc/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx`: JNE Word version.
- `docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md`: JNE clean placeholder variant without author-query text.
- `output/doc/ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx`: clean JNE Word version.
- `docs/cover_letter_template.md`: editable cover letter with target-specific paragraphs.
- `docs/author_required_information_form.md` and `output/doc/Author_Required_Information_Form.docx`: fields required before any journal upload.

## Decision Rule

Choose **JNE first** if the author can supply ethics, consent, data sharing, and EEG acquisition/preprocessing details. Choose **TNSRE first** if the author prioritizes engineering readership and is comfortable with IEEE formatting. Defer **Communications Medicine** and **Brain Stimulation** until all human-participant and intervention details are complete and the author accepts the higher desk-rejection risk.
