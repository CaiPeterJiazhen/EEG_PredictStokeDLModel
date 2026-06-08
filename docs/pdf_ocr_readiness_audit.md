# PDF OCR Readiness Audit

This audit checks whether the local patient-record PDF scans can be processed automatically for author-required manuscript metadata. It reports only non-identifying counts and tool availability. Patient names, PDF filenames, OCR text, and page images are not written to this report.

## Summary

- Existing PDF text-layer audit rows: 31
- PDFs audited in text-layer report: 31
- Pages audited in text-layer report: 277
- Pages with any extractable text layer: 62
- Extracted text characters in text-layer report: 434
- PDFs requiring OCR or manual review: 31
- OCR command available: False
- Python OCR package available: False
- PDF rendering smoke test: pass_rendered_first_available_record_page_60x85px
- Automatic OCR-ready in current environment: False

## OCR Tool Availability

| Tool or package | Type | Available |
|---|---|---|
| tesseract | command | False |
| pytesseract | python_ocr_package | False |
| easyocr | python_ocr_package | False |
| cnocr | python_ocr_package | False |
| rapidocr_onnxruntime | python_ocr_package | False |
| pypdfium2 | python_render_or_pdf_package | True |
| pypdf | python_render_or_pdf_package | True |
| PIL | python_render_or_pdf_package | True |

## Interpretation

- The current environment can inspect PDF metadata and render pages, but it does not have a usable OCR engine for Chinese/English scanned clinical documents.
- The existing text-layer audit found that all record-book PDFs require OCR or manual review before they can support ethics, consent, safety, protocol, or device metadata.
- Until OCR or manual review is completed, the scanned PDFs should not be used to write submission-ready statements.
- Any future OCR output should be reduced to non-identifying cohort-level evidence before manuscript insertion.

## Manual Review Targets

| Author field | What manual review must confirm |
|---|---|
| ethics_approval | ethics committee name, approval number, approval date, and site applicability |
| informed_consent | consent route, consent provider, procedure coverage, and data-sharing scope |
| trial_or_study_registration | registry identifier or author-approved non-registration statement |
| study_site_dates_design | hospital/department, recruitment dates, follow-up window, and design |
| eligibility_stroke_timing | inclusion/exclusion criteria, stroke subtype, lesion and timing rules |
| tacs_device_electrodes | device model, electrode dimensions/materials, and protocol confirmation |
| concurrent_rehabilitation | conventional rehabilitation dose, content, and consistency |
| tacs_safety_adverse_events | adverse events, tolerability, withdrawals, and monitoring method |
| fma_assessors_timing | assessor credentials, blinding, and assessment timing |
| eeg_hardware_reference_impedance | amplifier, cap, software, online reference, ground, and impedance |
| raw_eeg_preprocessing | filters, notch, rereference, bad-channel handling, ICA/artifact handling, export rules |
| data_repository_doi_scope | repository DOI/accession, licence, public/restricted scope, and request route |

## Recommended Next Step

Install or run an OCR workflow that supports simplified Chinese and English clinical documents, then review outputs manually before any manuscript wording is changed. A suitable local route would be Tesseract with `chi_sim` and `eng` language data, PaddleOCR/RapidOCR, or institutional OCR software. The review should export only de-identified, cohort-level protocol facts and evidence-source labels.
