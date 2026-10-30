#!/usr/bin/env python3
"""Summarize and compare Ascend ``kernel_details.csv`` profiler exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelRecord:
    operator_type: str
    name: str
    input_shapes: str
    start_us: float
    duration_us: float


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().replace("\t", "").replace(",", "")
    if not normalized or normalized.upper() in {"N/A", "NA", "NULL"}:
        return None
    try:
        result = float(normalized)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    # Some msprof exports contain a UTF-8 BOM, NUL padding, or a short command
    # transcript before the CSV header. Locate the real header defensively.
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Duration(us)" in line and "Name" in line and "Type" in line
        ),
        None,
    )
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def load_kernel_records(profile_dir: Path) -> tuple[list[KernelRecord], list[Path]]:
    files = sorted(profile_dir.rglob("kernel_details.csv"))
    records: list[KernelRecord] = []
    for path in files:
        for row in _read_csv_rows(path):
            duration = _as_float(row.get("Duration(us)"))
            start = _as_float(row.get("Start Time(us)"))
            if duration is None or duration < 0:
                continue
            records.append(
                KernelRecord(
                    operator_type=(row.get("Type") or row.get("Name") or "unknown").strip(),
                    name=(row.get("Name") or "unknown").strip(),
                    input_shapes=(row.get("Input Shapes") or "").strip(),
                    start_us=start if start is not None else 0.0,
                    duration_us=duration,
                )
            )
    return records, files


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interval_metrics(records: list[KernelRecord]) -> tuple[float, float]:
    intervals = sorted(
        (record.start_us, record.start_us + record.duration_us)
        for record in records
        if record.start_us > 0
    )
    if not intervals:
        return 0.0, 0.0
    span = max(end for _, end in intervals) - min(start for start, _ in intervals)
    busy = 0.0
    merged_start, merged_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            busy += merged_end - merged_start
            merged_start, merged_end = start, end
    busy += merged_end - merged_start
    return span, busy


def aggregate(records: list[KernelRecord]) -> tuple[list[dict[str, float | int | str]], dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.operator_type].append(record.duration_us)
    total_kernel_us = sum(record.duration_us for record in records)
    span_us, busy_us = _interval_metrics(records)
    rows = []
    for operator_type, durations in grouped.items():
        total = sum(durations)
        rows.append(
            {
                "operator_type": operator_type,
                "calls": len(durations),
                "total_us": total,
                "share_pct": 100.0 * total / total_kernel_us if total_kernel_us else 0.0,
                "mean_us": statistics.fmean(durations),
                "p50_us": _percentile(durations, 0.50),
                "p95_us": _percentile(durations, 0.95),
                "max_us": max(durations),
            }
        )
    rows.sort(key=lambda row: float(row["total_us"]), reverse=True)
    totals = {
        "kernel_records": len(records),
        "operator_types": len(grouped),
        "summed_kernel_duration_us": total_kernel_us,
        "trace_span_us": span_us,
        "device_busy_union_us": busy_us,
        "device_busy_pct_of_span": 100.0 * busy_us / span_us if span_us else 0.0,
    }
    return rows, totals


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "operator_type",
        "calls",
        "total_us",
        "share_pct",
        "mean_us",
        "p50_us",
        "p95_us",
        "max_us",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_profile(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--profile must use NAME=PATH syntax")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("profile NAME cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def _load_region_wall_us(profile_dir: Path) -> tuple[float, list[Path]]:
    files = sorted(profile_dir.rglob("region_timing.json"))
    total = 0.0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("synchronized_region_wall_us")
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            total += float(value)
    return total, files


def summarize_profiles(
    profiles: list[tuple[str, Path]], output_dir: Path, top: int
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict]] = {}
    all_totals: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}
    for name, profile_dir in profiles:
        records, files = load_kernel_records(profile_dir)
        region_wall_us, timing_files = _load_region_wall_us(profile_dir)
        if not files and not timing_files:
            raise FileNotFoundError(
                "No kernel_details.csv or region_timing.json found below "
                f"{profile_dir}"
            )
        rows, totals = aggregate(records)
        totals["synchronized_region_wall_us"] = region_wall_us
        totals["device_busy_pct_of_region_wall"] = (
            100.0 * totals["device_busy_union_us"] / region_wall_us
            if region_wall_us
            else 0.0
        )
        all_rows[name] = rows
        all_totals[name] = totals
        sources[name] = [str(path) for path in [*files, *timing_files]]
        _write_csv(output_dir / f"{name}_operator_summary.csv", rows)

    metadata = {"profiles": all_totals, "sources": sources}
    (output_dir / "profile_totals.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    operator_types = sorted(
        {str(row["operator_type"]) for rows in all_rows.values() for row in rows},
        key=lambda operator: max(
            (
                float(row["total_us"])
                for rows in all_rows.values()
                for row in rows
                if row["operator_type"] == operator
            ),
            default=0.0,
        ),
        reverse=True,
    )
    lookup = {
        name: {str(row["operator_type"]): row for row in rows}
        for name, rows in all_rows.items()
    }
    report = output_dir / "operator_comparison.md"
    with report.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# Ascend NPU operator comparison\n\n")
        output.write("## Trace totals\n\n")
        output.write(
            "| Profile | Kernel rows | Region wall ms | Sum kernel ms | "
            "Device busy ms | Trace span ms | Device/wall |\n"
        )
        output.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, _ in profiles:
            totals = all_totals[name]
            output.write(
                f"| {name} | {totals['kernel_records']} | "
                f"{totals['synchronized_region_wall_us'] / 1000:.3f} | "
                f"{totals['summed_kernel_duration_us'] / 1000:.3f} | "
                f"{totals['device_busy_union_us'] / 1000:.3f} | "
                f"{totals['trace_span_us'] / 1000:.3f} | "
                f"{totals['device_busy_pct_of_region_wall']:.2f}% |\n"
            )
        output.write(
            "\n`Sum kernel ms` can exceed wall time when kernels overlap across streams; "
            "`Device busy ms` is the union of kernel intervals. A stage with zero "
            "kernel rows but nonzero region wall time is host work or a non-kernel "
            "runtime operation such as a device copy.\n\n"
        )
        output.write(f"## Top {top} operator types\n\n")
        output.write("| Operator | " + " | ".join(name for name, _ in profiles) + " |\n")
        output.write("|---|" + "---:|" * len(profiles) + "\n")
        for operator in operator_types[:top]:
            values = []
            for name, _ in profiles:
                row = lookup[name].get(operator)
                values.append(
                    "—"
                    if row is None
                    else f"{float(row['total_us']) / 1000:.3f} ms / {row['calls']} calls"
                )
            output.write(f"| {operator} | " + " | ".join(values) + " |\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        type=_parse_profile,
        help="Named profile in NAME=PATH form; repeat to compare traces.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    report = summarize_profiles(args.profile, args.output_dir.resolve(), args.top)
    print(report)


if __name__ == "__main__":
    main()
