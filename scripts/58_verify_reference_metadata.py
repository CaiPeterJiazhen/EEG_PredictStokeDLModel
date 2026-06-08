from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md"
OUTPUT_MD = ROOT / "docs" / "reference_metadata_audit.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "reference_metadata_audit.csv"

USER_AGENT = "EEG-PredictStrokeDLModel-reference-audit/1.0 (mailto:author@example.com)"


@dataclass
class Reference:
    number: int
    raw: str
    doi: str
    url: str
    local_title: str
    local_year: str


def main() -> None:
    refs = parse_references(MANUSCRIPT.read_text(encoding="utf-8"))
    rows = [audit_reference(reference) for reference in refs]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "number",
                "identifier_type",
                "identifier",
                "local_year",
                "metadata_year",
                "local_title",
                "metadata_title",
                "metadata_container",
                "lookup_status",
                "match_status",
                "source_url",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)

    failures = [row for row in rows if row["lookup_status"] == "FAIL" or row["match_status"] == "FAIL"]
    if failures:
        raise SystemExit(f"Reference metadata audit found {len(failures)} failures")


def parse_references(text: str) -> list[Reference]:
    if "## References" not in text:
        raise ValueError("Missing References section")
    section = text.split("## References", maxsplit=1)[1]
    if "## Author queries before journal submission" in section:
        section = section.split("## Author queries before journal submission", maxsplit=1)[0]

    refs: list[Reference] = []
    for line in section.splitlines():
        line = line.strip()
        match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if not match:
            continue
        number = int(match.group(1))
        raw = match.group(2)
        doi_match = re.search(r"doi:(10\.\S+)", raw, re.I)
        url_match = re.search(r"(https?://\S+)", raw)
        doi = doi_match.group(1).rstrip(".") if doi_match else ""
        url = url_match.group(1).rstrip(".") if url_match else ""
        refs.append(
            Reference(
                number=number,
                raw=raw,
                doi=doi,
                url=url,
                local_title=extract_title(raw),
                local_year=extract_year(raw),
            )
        )
    return refs


def extract_title(raw: str) -> str:
    # Title is usually the sentence after the author block. Keep this heuristic
    # conservative; the audit uses it only for fuzzy metadata comparison.
    parts = raw.split(". ")
    for candidate in parts[1:4]:
        stripped = candidate.strip()
        if stripped and not re.match(r"^[A-Z][a-z]+ [A-Z]", stripped) and not re.match(r"^\d{4}", stripped):
            return stripped.rstrip(".")
    return parts[1].strip().rstrip(".") if len(parts) > 1 else raw


def extract_year(raw: str) -> str:
    matches = re.findall(r"\b(20\d{2}|19\d{2})\b", raw)
    return matches[0] if matches else ""


def audit_reference(reference: Reference) -> dict[str, str | int]:
    if reference.doi:
        row = audit_doi(reference)
    elif reference.url:
        row = audit_url(reference)
    else:
        row = base_row(reference)
        row.update(
            {
                "identifier_type": "none",
                "identifier": "",
                "lookup_status": "WARN",
                "match_status": "WARN",
                "source_url": "",
                "notes": "No DOI or URL present in reference entry.",
            }
        )
    time.sleep(0.2)
    return row


def audit_doi(reference: Reference) -> dict[str, str | int]:
    row = base_row(reference)
    doi = reference.doi.lower()
    source_url = "https://doi.org/" + doi
    row.update({"identifier_type": "doi", "identifier": doi, "source_url": source_url})
    api_url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        payload = request_json(api_url)
        message = payload.get("message", {})
        title = first(message.get("title", []))
        container = first(message.get("container-title", []))
        year = metadata_year(message)
        row.update(
            {
                "metadata_title": title,
                "metadata_container": container,
                "metadata_year": year,
                "lookup_status": "PASS",
            }
        )
        title_match = fuzzy_title_match(reference.local_title, title)
        year_match = not reference.local_year or not year or reference.local_year == year
        if title_match and year_match:
            row["match_status"] = "PASS"
            row["notes"] = "Crossref DOI metadata title/year are consistent with manuscript entry."
        elif title_match:
            row["match_status"] = "WARN"
            row["notes"] = "Title matches, but year differs or could not be resolved."
        else:
            row["match_status"] = "WARN"
            row["notes"] = "DOI resolved, but title similarity is weak; manually verify this reference."
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "lookup_status": "FAIL",
                "match_status": "FAIL",
                "source_url": source_url,
                "notes": f"Crossref lookup failed: {type(exc).__name__}: {exc}",
            }
        )
    return row


def audit_url(reference: Reference) -> dict[str, str | int]:
    row = base_row(reference)
    row.update(
        {
            "identifier_type": "url",
            "identifier": reference.url,
            "source_url": reference.url,
        }
    )
    try:
        status = request_status(reference.url)
        row.update(
            {
                "lookup_status": "PASS" if 200 <= status < 400 else "WARN",
                "match_status": "PASS" if 200 <= status < 400 else "WARN",
                "notes": f"URL returned HTTP {status}. Title metadata not checked for non-DOI reference.",
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "lookup_status": "FAIL",
                "match_status": "WARN",
                "notes": f"URL check failed: {type(exc).__name__}: {exc}",
            }
        )
    return row


def base_row(reference: Reference) -> dict[str, str | int]:
    return {
        "number": reference.number,
        "identifier_type": "",
        "identifier": "",
        "local_year": reference.local_year,
        "metadata_year": "",
        "local_title": reference.local_title,
        "metadata_title": "",
        "metadata_container": "",
        "lookup_status": "",
        "match_status": "",
        "source_url": "",
        "notes": "",
    }


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=25) as response:
        if response.status >= 400:
            raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
        return json.loads(response.read().decode("utf-8"))


def request_status(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405}:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=20) as response:
                return int(response.status)
        raise


def first(values: list[str] | tuple[str, ...] | None) -> str:
    if not values:
        return ""
    return str(values[0])


def metadata_year(message: dict) -> str:
    for key in ["published-print", "published-online", "published", "issued"]:
        date_parts = message.get(key, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return str(date_parts[0][0])
    return ""


def fuzzy_title_match(local: str, metadata: str) -> bool:
    local_tokens = title_tokens(local)
    metadata_tokens = title_tokens(metadata)
    if not local_tokens or not metadata_tokens:
        return False
    overlap = len(local_tokens & metadata_tokens)
    return overlap / max(1, min(len(local_tokens), len(metadata_tokens))) >= 0.6


def title_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "or",
        "the",
        "to",
        "using",
        "with",
    }
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return {token for token in normalized.split() if len(token) > 2 and token not in stop}


def build_markdown(rows: list[dict[str, str | int]]) -> str:
    lookup_counts = count(rows, "lookup_status")
    match_counts = count(rows, "match_status")
    doi_count = sum(1 for row in rows if row["identifier_type"] == "doi")
    url_count = sum(1 for row in rows if row["identifier_type"] == "url")
    lines = [
        "# Reference Metadata Audit",
        "",
        "This audit checks whether manuscript reference identifiers resolve through Crossref DOI metadata or stable URL status checks. It supports final journal reference formatting, but it does not replace target-journal style editing.",
        "",
        "## Summary",
        "",
        f"- Total references: {len(rows)}",
        f"- DOI references checked through Crossref: {doi_count}",
        f"- URL-only references checked by HTTP status: {url_count}",
        f"- Lookup status: {format_counts(lookup_counts)}",
        f"- Match status: {format_counts(match_counts)}",
        "",
        "## Results",
        "",
        "| Ref | ID type | Identifier | Lookup | Match | Metadata year | Metadata container | Notes |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                sanitize_md(str(value))
                for value in [
                    row["number"],
                    row["identifier_type"],
                    row["identifier"],
                    row["lookup_status"],
                    row["match_status"],
                    row["metadata_year"],
                    row["metadata_container"],
                    row["notes"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS` DOI rows resolved through Crossref and had title/year metadata consistent with the manuscript entry.",
            "- `PASS` URL rows were reachable, but title-level metadata were not checked for non-DOI software or conference references.",
            "- Any `WARN` row should be manually verified before final journal upload, especially if the target journal requires strict reference metadata.",
            "- The current audit does not decide whether each citation is sufficient support for a claim; that mapping remains in `docs/citation_artifacts/manuscript_citation_map.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def count(rows: list[dict[str, str | int]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return counts


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def sanitize_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    main()
