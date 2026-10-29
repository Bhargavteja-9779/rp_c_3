#!/usr/bin/env python3
"""Write FILE_MANIFEST.md: every submission file with size, SHA-256 (first 12 hex) and purpose."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "raw", "_root", "smoke", "quick", "data"}
PURPOSE = {
    "manuscript/Final_Manuscript.docx": "Anonymised manuscript for JKSU-CIS (primary submission file)",
    "manuscript/Final_Manuscript.pdf": "PDF rendering of the manuscript (for checking)",
    "manuscript/Title_Page.docx": "Separate title page (author details to be completed by the authors)",
    "manuscript/Cover_Letter.docx": "Cover letter to the JKSU-CIS editor",
    "manuscript/alternative_ESWA/Manuscript_ESWA.docx": "Alternative: Elsevier/ESWA formatting (highlights, Harvard references)",
    "supplementary/online_resources/Online_Resource_1.docx": "ESM 1: derivations",
    "supplementary/online_resources/Online_Resource_2.docx": "ESM 2: pre-registered protocol, amendments, post-hoc changes, Table S1",
    "supplementary/online_resources/Online_Resource_3.xlsx": "ESM 3: per-dataset results and Tables S2-S10",
    "code/run_all.py": "Single entry point reproducing every result, table and figure",
    "code/results/raw_results.csv": "Every metric per dataset, resample, method and condition",
    "code/results/statistics.json": "All statistical tests used in the manuscript",
    "code/results/metrics.json": "Headline medians",
    "code/results/raw_jobs.tar.gz": "Cached per-job JSON results (allows re-analysis without recomputation)",
    "REVIEW/final_audit.md": "Final consistency/reproducibility audit and READY decision",
    "REJECTION_RISK_REPORT.md": "Rejection-risk assessment",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main():
    rows = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or any(part in SKIP_DIRS for part in rel.parts[:-1]) or p.name.startswith("_") \
                or p.suffix in (".pyc",) or rel.parts[0] == "journal_selection" and "evidence" in rel.parts and p.stat().st_size > 5e6:
            continue
        rows.append((str(rel), p.stat().st_size, sha(p), PURPOSE.get(str(rel), "")))
    out = ["# File manifest", "", f"{len(rows)} files (raw per-job caches, downloaded data and test artefacts excluded; "
           "the cached jobs are shipped as `code/results/raw_jobs.tar.gz`).", "",
           "| File | Size (KB) | SHA-256 (12) | Purpose |", "|---|---|---|---|"]
    out += [f"| `{r}` | {s / 1024:,.1f} | `{h}` | {d} |" for r, s, h, d in rows]
    (ROOT / "FILE_MANIFEST.md").write_text("\n".join(out) + "\n")
    print(f"{len(rows)} files listed")


if __name__ == "__main__":
    main()
