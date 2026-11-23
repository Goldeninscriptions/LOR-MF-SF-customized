#!/usr/bin/env python3
"""Merge the existing K spectrum with the new Greville P/P^-1 K sweep."""

from __future__ import annotations

import csv
from pathlib import Path

from make_condition_number_excel import DEGREES, MESHES, make_workbook


WORK_DIR = Path(__file__).resolve().parent
OLD = WORK_DIR / "results" / "eps_condition_batch_2d" / "summary.tsv"
NEW = WORK_DIR / "results" / "eps_greville_condition_batch_2d" / "summary.tsv"
OUTPUT = (WORK_DIR / "results" / "eps_greville_condition_batch_2d" /
          "condition_numbers_2d_greville.xlsx")
FIELDS = [
    "p", "n", "target", "mpi_ranks", "free_dofs", "lambda_min",
    "lambda_max", "condition_number", "eps_iterations_min",
    "eps_iterations_max", "time_spectral_eps_s", "result_file",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> None:
    old_k = [row for row in read(OLD) if row["target"] == "k"]
    new = read(NEW)
    rows = [{field: row[field] for field in FIELDS} for row in old_k + new]
    expected = {
        (str(p), str(n), target)
        for p in DEGREES for n in MESHES
        for target in ("k", "p", "pinv_k")
    }
    actual = {(row["p"], row["n"], row["target"]) for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise RuntimeError(
            f"incomplete/duplicate merged results: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}")

    workbook = make_workbook(rows)
    sheet = workbook["条件数汇总"]
    for column in range(2, 29, 3):
        sheet.cell(2, column, "K（已有）")
        sheet.cell(2, column + 1, "P_G")
        sheet.cell(2, column + 2, "P_G⁻¹K")
    sheet["A1"] = "p（Greville tensor）"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
