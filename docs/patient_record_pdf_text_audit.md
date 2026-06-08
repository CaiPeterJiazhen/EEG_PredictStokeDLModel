# Patient-Record PDF Text-Layer Audit

This audit checks whether local patient-record PDFs can be used as searchable evidence for author-required manuscript metadata. It deliberately reports only non-identifying counts and pseudonymous record IDs; patient names, PDF file names, and extracted source text are not written to this report.

## Summary

- PDFs audited: 31
- Total pages: 277
- Extracted text characters: 434
- healthy_control_record_books: 2
- patient_record_books: 29
- scan_likely_no_text_layer: 31
- ocr_or_manual_review_required: 31

## Interpretation

- The current PDFs do not provide searchable protocol, consent, ethics, device, rehabilitation, or safety text unless a text layer is detected below.
- Image-based record books require OCR or manual review before any manuscript metadata can be inferred from them.
- Even if OCR is later performed, patient-level notes should be converted into non-identifying cohort-level evidence before manuscript use.

## Non-Identifying Record Audit

| Record group | Record ID | Pages | Text chars | Pages with text | Images | Mean image coverage | Text-layer status | Metadata utility |
|---|---|---:|---:|---:|---:|---:|---|---|
| patient_record_books | patient_record_books_001 | 10 | 0 | 0 | 10 | 0.9989 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_002 | 4 | 0 | 0 | 4 | 0.9900 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_003 | 7 | 0 | 0 | 7 | 0.9989 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_004 | 6 | 0 | 0 | 6 | 0.9989 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_005 | 7 | 0 | 0 | 7 | 0.9989 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_006 | 4 | 0 | 0 | 4 | 0.9989 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_007 | 11 | 0 | 0 | 11 | 0.9989 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_008 | 10 | 0 | 0 | 10 | 0.9934 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_009 | 7 | 0 | 0 | 7 | 0.9925 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_010 | 7 | 0 | 0 | 7 | 0.9883 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_011 | 8 | 0 | 0 | 8 | 0.9986 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_012 | 16 | 112 | 16 | 32 | 0.9292 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_013 | 23 | 0 | 0 | 46 | 0.8865 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_014 | 7 | 0 | 0 | 7 | 0.9935 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_015 | 15 | 105 | 15 | 30 | 0.9380 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_016 | 7 | 0 | 0 | 7 | 0.9985 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_017 | 8 | 0 | 0 | 8 | 0.9984 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_018 | 7 | 0 | 0 | 7 | 0.9878 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_019 | 4 | 0 | 0 | 4 | 0.9946 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_020 | 14 | 98 | 14 | 28 | 0.8839 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_021 | 7 | 0 | 0 | 7 | 0.9937 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_022 | 17 | 119 | 17 | 34 | 0.8677 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_023 | 9 | 0 | 0 | 18 | 0.8823 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_024 | 7 | 0 | 0 | 7 | 0.9902 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_025 | 6 | 0 | 0 | 6 | 0.9954 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_026 | 5 | 0 | 0 | 5 | 0.9955 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_027 | 7 | 0 | 0 | 7 | 0.9963 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_028 | 7 | 0 | 0 | 7 | 0.9982 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| patient_record_books | patient_record_books_029 | 7 | 0 | 0 | 7 | 0.9958 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| healthy_control_record_books | healthy_control_record_books_001 | 11 | 0 | 0 | 22 | 0.8901 | scan_likely_no_text_layer | ocr_or_manual_review_required |
| healthy_control_record_books | healthy_control_record_books_002 | 12 | 0 | 0 | 24 | 0.9229 | scan_likely_no_text_layer | ocr_or_manual_review_required |
