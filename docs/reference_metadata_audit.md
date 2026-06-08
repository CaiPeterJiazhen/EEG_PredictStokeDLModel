# Reference Metadata Audit

This audit checks whether manuscript reference identifiers resolve through Crossref DOI metadata or stable URL status checks. It supports final journal reference formatting, but it does not replace target-journal style editing.

## Summary

- Total references: 25
- DOI references checked through Crossref: 20
- URL-only references checked by HTTP status: 5
- Lookup status: PASS=25
- Match status: PASS=25

## Results

| Ref | ID type | Identifier | Lookup | Match | Metadata year | Metadata container | Notes |
|---:|---|---|---|---|---|---|---|
| 1 | doi | 10.1177/1545968307305302 | PASS | PASS | 2008 | Neurorehabilitation and Neural Repair | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 2 | doi | 10.1002/ana.24472 | PASS | PASS | 2015 | Annals of Neurology | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 3 | doi | 10.1002/acn3.488 | PASS | PASS | 2017 | Annals of Clinical and Translational Neurology | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 4 | doi | 10.1109/jbhi.2022.3205436 | PASS | PASS | 2022 | IEEE Journal of Biomedical and Health Informatics | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 5 | doi | 10.1016/j.nicl.2024.103638 | PASS | PASS | 2024 | NeuroImage: Clinical | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 6 | doi | 10.1063/5.0287165 | PASS | PASS | 2026 | APL Bioengineering | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 7 | doi | 10.1109/tnsre.2019.2924742 | PASS | PASS | 2019 | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 8 | doi | 10.1016/j.clinph.2020.09.031 | PASS | PASS | 2021 | Clinical Neurophysiology | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 9 | doi | 10.1177/1545968320913502 | PASS | PASS | 2020 | Neurorehabilitation and Neural Repair | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 10 | doi | 10.1016/j.neuroimage.2011.01.055 | PASS | PASS | 2011 | NeuroImage | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 11 | doi | 10.3389/fnins.2013.00267 | PASS | PASS | 2013 | Frontiers in Neuroscience | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 12 | doi | 10.1016/j.neuroimage.2013.10.027 | PASS | PASS | 2014 | NeuroImage | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 13 | doi | 10.7326/m14-0697 | PASS | PASS | 2015 | Annals of Internal Medicine | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 14 | doi | 10.1136/bmj-2023-078378 | PASS | PASS | 2024 | BMJ | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 15 | doi | 10.7326/m18-1376 | PASS | PASS | 2019 | Annals of Internal Medicine | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 16 | doi | 10.1038/s41598-025-19860-4 | PASS | PASS | 2025 | Scientific Reports | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 17 | url | https://proceedings.mlr.press/v139/zbontar21a.html | PASS | PASS |  |  | URL returned HTTP 200. Title metadata not checked for non-DOI reference. |
| 18 | url | https://proceedings.mlr.press/v70/sundararajan17a.html | PASS | PASS |  |  | URL returned HTTP 200. Title metadata not checked for non-DOI reference. |
| 19 | url | https://arxiv.org/abs/1706.03825 | PASS | PASS |  |  | URL returned HTTP 200. Title metadata not checked for non-DOI reference. |
| 20 | url | https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html | PASS | PASS |  |  | URL returned HTTP 200. Title metadata not checked for non-DOI reference. |
| 21 | url | https://jmlr.org/papers/v12/pedregosa11a.html | PASS | PASS |  |  | URL returned HTTP 200. Title metadata not checked for non-DOI reference. |
| 22 | doi | 10.1109/tnsre.2022.3153353 | PASS | PASS | 2022 | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 23 | doi | 10.1177/1545968320909796 | PASS | PASS | 2020 | Neurorehabilitation and Neural Repair | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 24 | doi | 10.1155/2022/8645165 | PASS | PASS | 2022 | Applied Bionics and Biomechanics | Crossref DOI metadata title/year are consistent with manuscript entry. |
| 25 | doi | 10.1016/j.bea.2024.100121 | PASS | PASS | 2024 | Biomedical Engineering Advances | Crossref DOI metadata title/year are consistent with manuscript entry. |

## Interpretation

- `PASS` DOI rows resolved through Crossref and had title/year metadata consistent with the manuscript entry.
- `PASS` URL rows were reachable, but title-level metadata were not checked for non-DOI software or conference references.
- Any `WARN` row should be manually verified before final journal upload, especially if the target journal requires strict reference metadata.
- The current audit does not decide whether each citation is sufficient support for a claim; that mapping remains in `docs/citation_artifacts/manuscript_citation_map.md`.
