from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VARIANTS = (
    (
        ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md",
        ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md",
    ),
    (
        ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_structured.md",
        ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md",
    ),
)


def main() -> None:
    for source, destination in VARIANTS:
        destination.write_text(clean_text(source.read_text(encoding="utf-8")), encoding="utf-8")
        print(destination)


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    skip_author_queries = False

    for line in lines:
        stripped = line.strip()
        if stripped == "## Author queries before journal submission":
            skip_author_queries = True
            continue
        if skip_author_queries:
            continue
        if stripped.startswith("Author information required before submission:"):
            continue
        cleaned.append(line)

    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()
    return "\n".join(cleaned) + "\n"


if __name__ == "__main__":
    main()
