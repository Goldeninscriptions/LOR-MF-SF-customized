#!/usr/bin/env python3
"""Create the formatted 2D EPS condition-number workbook from summary.tsv."""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


WORK_DIR = Path(__file__).resolve().parent
SOURCE = WORK_DIR / "results" / "eps_condition_batch_2d" / "summary.tsv"
OUTPUT = WORK_DIR / "results" / "eps_condition_batch_2d" / "condition_numbers_2d.xlsx"
DEGREES = [3, 4, 5, 6, 7]
MESHES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
TARGETS = [("k", "K"), ("p", "P"), ("pinv_k", "P⁻¹K")]


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    expected = {
        (str(degree), str(mesh), target)
        for degree in DEGREES
        for mesh in MESHES
        for target, _ in TARGETS
    }
    actual = {(row["p"], row["n"], row["target"]) for row in rows}
    if len(rows) != len(actual):
        raise RuntimeError("summary.tsv contains duplicate (p, n, target) rows")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"incomplete result set: missing={missing}, extra={extra}")
    return rows


def make_workbook(rows: list[dict[str, str]]) -> Workbook:
    values = {
        (int(row["p"]), int(row["n"]), row["target"]): float(row["condition_number"])
        for row in rows
    }
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "条件数汇总"

    dark_blue = PatternFill("solid", fgColor="1F4E78")
    medium_blue = PatternFill("solid", fgColor="5B9BD5")
    pale_blue = PatternFill("solid", fgColor="D9EAF7")
    white_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="A6A6A6")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    sheet.merge_cells("A1:A2")
    sheet["A1"] = "p"
    sheet["A1"].fill = dark_blue
    sheet["A1"].font = white_font
    sheet["A1"].alignment = Alignment(horizontal="center", vertical="center")

    column = 2
    for mesh in MESHES:
        start = column
        end = column + len(TARGETS) - 1
        sheet.merge_cells(start_row=1, start_column=start, end_row=1, end_column=end)
        cell = sheet.cell(1, start, f"{mesh}×{mesh}")
        cell.fill = dark_blue
        cell.font = white_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        for target, label in TARGETS:
            cell = sheet.cell(2, column, label)
            cell.fill = medium_blue
            cell.font = white_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            column += 1

    for row_index, degree in enumerate(DEGREES, start=3):
        degree_cell = sheet.cell(row_index, 1, degree)
        degree_cell.fill = pale_blue
        degree_cell.font = Font(bold=True)
        degree_cell.alignment = Alignment(horizontal="center")
        column = 2
        for mesh in MESHES:
            for target, _ in TARGETS:
                cell = sheet.cell(row_index, column, values[(degree, mesh, target)])
                cell.number_format = "0.000000E+00"
                cell.alignment = Alignment(horizontal="right")
                column += 1

    for row in sheet.iter_rows(min_row=1, max_row=2 + len(DEGREES), min_col=1, max_col=28):
        for cell in row:
            cell.border = border
    # Merged header cells do not inherit the style of the top-left cell.
    for column in range(1, 29):
        sheet.cell(1, column).border = border
        sheet.cell(2, column).border = border

    sheet.freeze_panes = "B3"
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 24
    sheet.row_dimensions[2].height = 22
    sheet.column_dimensions["A"].width = 7
    for column in range(2, 29):
        sheet.column_dimensions[get_column_letter(column)].width = 15
    raw = workbook.create_sheet("原始数据")
    raw_headers = list(rows[0])
    for column, header in enumerate(raw_headers, start=1):
        cell = raw.cell(1, column, header)
        cell.fill = dark_blue
        cell.font = white_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = border
    numeric_fields = {
        "p": int,
        "n": int,
        "mpi_ranks": int,
        "free_dofs": int,
        "lambda_min": float,
        "lambda_max": float,
        "condition_number": float,
        "eps_iterations_min": int,
        "eps_iterations_max": int,
        "time_spectral_eps_s": float,
    }
    for row_index, result in enumerate(rows, start=2):
        for column, header in enumerate(raw_headers, start=1):
            value = result[header]
            if header in numeric_fields:
                value = numeric_fields[header](value)
            cell = raw.cell(row_index, column, value)
            cell.border = border
            if header in {"lambda_min", "lambda_max", "condition_number"}:
                cell.number_format = "0.000000000000E+00"
            elif header == "time_spectral_eps_s":
                cell.number_format = "0.000"
    raw.freeze_panes = "A2"
    raw.auto_filter.ref = raw.dimensions
    raw.sheet_view.showGridLines = False
    for column, header in enumerate(raw_headers, start=1):
        width = 14
        if header == "result_file":
            width = 70
        raw.column_dimensions[get_column_letter(column)].width = width

    return workbook


def main() -> None:
    rows = load_rows()
    workbook = make_workbook(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
