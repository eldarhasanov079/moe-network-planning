"""Redraw paper Clos figures from an existing plans.csv (no re-plan)."""

from __future__ import annotations

import csv
from pathlib import Path

from .claim_v1 import source_bundle
from .clos_baselines import _draw, _report, _summarize, _write_json


def redraw(csv_path: Path, out: Path) -> None:
    rows = []
    with Path(csv_path).open() as handle:
        for raw in csv.DictReader(handle):
            raw["iteration_time_s"] = float(raw["iteration_time_s"])
            raw["seed"] = int(raw["seed"])
            rows.append(raw)
    sources = sorted({r["source"] for r in rows})
    bundles = [source_bundle(name) for name in sources]
    summary = _summarize(rows, bundles)
    _write_json(out / "summary.json", summary)
    _draw(out / "figures", rows, bundles)
    report = _report(summary)
    (out / "REPORT.md").write_text(report)
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "clos-baselines.md").write_text(report)
    print(report)


if __name__ == "__main__":
    root = Path("plane/output/clos-baselines")
    redraw(root / "plans.csv", root)
