# Lin 2022 Reader Notes

## Extraction

- PDF text extraction succeeded with PyMuPDF.
- Extracted text blocks: 151.
- Rendered page assets: `assets/page_01.png` through `assets/page_09.png`.
- Extracted embedded image assets: 7.
- Source map: `source_map.json`.
- Image manifest: `image_manifest.json`.

## Scope

The user asked to use the Lin paper to understand what each section should contain and what figures/tables should be placed where. Therefore `paper.md` is a structure-aware reader for manuscript adaptation rather than a full paragraph-by-paragraph bilingual reproduction of the article.

## Copyright And Fidelity

The reader paraphrases the article's structure and uses page/source pointers instead of reproducing the full copyrighted text. The exact extracted source blocks are kept locally in `source_map.json` for audit and follow-up questions.

## Uncertainty

- Figure extraction was based on embedded image objects, not manual tight-crop annotation. The figure assignments are high confidence for Fig. 1-Fig. 6 based on source page order and captions, but not a journal-ready reproduction of Lin's figures.
- Table I and Table II are not converted into separate table assets because they are rendered as text/table layout inside PDF pages. Use the page images and source blocks if exact table values are needed.
- The reader's Chinese sections are interpretive translations of each section's function, not a full sentence-level translation.

