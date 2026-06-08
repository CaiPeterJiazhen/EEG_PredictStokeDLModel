# Pre-submission Editorial Readiness Audit

This audit checks editorial preflight risks that are not fully captured by structural package gates: title length, abstract length, clean manuscript variants, author-query boundaries, data/code availability, citation coverage, and final submission gate status.

## Summary

- PASS: 22
- WARN: 0
- BLOCKED: 1

## Checks

| Category | Item | Observed | Target | Status | Action |
|---|---|---|---|---|---|
| title | nature_working | 7 words; 65 characters | <=20 words; searchable and bounded | PASS | Keep title concise; revise only after the final target journal is chosen. |
| abstract | nature_working | 223 words | <=250 words preferred for broad journal portability | PASS | Shorten context, comparator detail, or exploratory caveats if the target journal enforces a tighter limit. |
| main_text | nature_working | 4340 words before references | <=5000 words as a practical pre-submission target | PASS | Keep Methods concise; target-journal limits should be checked before final upload. |
| author_queries | nature_working | 2 explicit author-query lines | 0 in clean variants; working variants may retain queries | PASS | Use clean variants for upload after author metadata are inserted. |
| title | nature_clean | 7 words; 65 characters | <=20 words; searchable and bounded | PASS | Keep title concise; revise only after the final target journal is chosen. |
| abstract | nature_clean | 223 words | <=250 words preferred for broad journal portability | PASS | Shorten context, comparator detail, or exploratory caveats if the target journal enforces a tighter limit. |
| main_text | nature_clean | 4258 words before references | <=5000 words as a practical pre-submission target | PASS | Keep Methods concise; target-journal limits should be checked before final upload. |
| author_queries | nature_clean | 0 explicit author-query lines | 0 in clean variants; working variants may retain queries | PASS | Use clean variants for upload after author metadata are inserted. |
| title | jne_working | 7 words; 65 characters | <=20 words; searchable and bounded | PASS | Keep title concise; revise only after the final target journal is chosen. |
| abstract | jne_working | 230 words | <=250 words preferred for broad journal portability | PASS | Shorten context, comparator detail, or exploratory caveats if the target journal enforces a tighter limit. |
| main_text | jne_working | 4347 words before references | <=5000 words as a practical pre-submission target | PASS | Keep Methods concise; target-journal limits should be checked before final upload. |
| author_queries | jne_working | 2 explicit author-query lines | 0 in clean variants; working variants may retain queries | PASS | Use clean variants for upload after author metadata are inserted. |
| title | jne_clean | 7 words; 65 characters | <=20 words; searchable and bounded | PASS | Keep title concise; revise only after the final target journal is chosen. |
| abstract | jne_clean | 230 words | <=250 words preferred for broad journal portability | PASS | Shorten context, comparator detail, or exploratory caveats if the target journal enforces a tighter limit. |
| main_text | jne_clean | 4265 words before references | <=5000 words as a practical pre-submission target | PASS | Keep Methods concise; target-journal limits should be checked before final upload. |
| author_queries | jne_clean | 0 explicit author-query lines | 0 in clean variants; working variants may retain queries | PASS | Use clean variants for upload after author metadata are inserted. |
| structured_abstract | jne_working | Objective, Approach, Main results, Significance | Objective, Approach, Main results, and Significance present | PASS | Maintain structured headings for JNE-style submission. |
| availability | nature_working | Data availability=True; Code availability=True | Both sections present | PASS | Replace repository DOI/licence placeholders only after author-approved metadata are available. |
| references | metadata_audit | - Lookup status: PASS=25; - Match status: PASS=25 | Lookup and match checks pass | PASS | Re-run reference metadata audit after target-journal style edits or added references. |
| citations | claim_map | citation map present | Main citable claim areas mapped | PASS | Do not add citations unless a manuscript sentence requires them directly. |
| docx_visual_qa | current_package | - Page checks passed: 75; - Page checks failed: 0 | 0 failed rendered pages | PASS | Inspect final PDF previews manually before upload. |
| artifact_quality | current_package | - Manifest rows: 224; - Zip entries: 224; - Status: PASS | Package integrity passes | PASS | Warnings from working manuscripts retaining author-query text are expected until final insertion. |
| final_gate | current_package | Overall status: **BLOCKED_BY_AUTHOR_METADATA**; - PASS: 7; - WARN: 1; - BLOCKED: 2; - FAIL: 0 | No non-author FAIL; author metadata may remain blocked | BLOCKED | Complete author metadata before final upload; do not mark the manuscript ready while this remains blocked. |

## Interpretation

- The abstract-length target is a conservative portability threshold, not a substitute for the final journal's instructions.
- Clean variants are the upload candidates after author metadata are inserted; working variants intentionally retain author-query lines.
- `BLOCKED` indicates a submission-critical author metadata dependency rather than a failure of the manuscript package itself.
