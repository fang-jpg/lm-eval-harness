import csv
import json

import pytest

from scripts.summarize_npu_profile import aggregate, load_kernel_records, summarize_profiles


def _write_kernel_details(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Name", "Type", "Start Time(us)", "Duration(us)", "Input Shapes"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_kernel_summary_aggregates_calls_and_overlapping_intervals(tmp_path):
    profile = tmp_path / "profile"
    _write_kernel_details(
        profile / "run" / "kernel_details.csv",
        [
            {
                "Name": "mm_1",
                "Type": "MatMul",
                "Start Time(us)": "100",
                "Duration(us)": "10",
                "Input Shapes": "1,4",
            },
            {
                "Name": "mm_2",
                "Type": "MatMul",
                "Start Time(us)": "105",
                "Duration(us)": "10",
                "Input Shapes": "1,4",
            },
            {
                "Name": "softmax",
                "Type": "Softmax",
                "Start Time(us)": "120",
                "Duration(us)": "5",
                "Input Shapes": "1,4",
            },
        ],
    )

    records, files = load_kernel_records(profile)
    rows, totals = aggregate(records)

    assert len(files) == 1
    assert totals["summed_kernel_duration_us"] == pytest.approx(25)
    assert totals["device_busy_union_us"] == pytest.approx(20)
    assert totals["trace_span_us"] == pytest.approx(25)
    assert rows[0]["operator_type"] == "MatMul"
    assert rows[0]["calls"] == 2
    assert rows[0]["total_us"] == pytest.approx(20)


def test_profile_comparison_writes_markdown_and_csv(tmp_path):
    ar = tmp_path / "ar"
    diffusion = tmp_path / "diffusion"
    for profile, duration in ((ar, "10"), (diffusion, "14")):
        _write_kernel_details(
            profile / "kernel_details.csv",
            [
                {
                    "Name": "mm",
                    "Type": "MatMul",
                    "Start Time(us)": "100",
                    "Duration(us)": duration,
                    "Input Shapes": "1,4",
                }
            ],
        )

    output = tmp_path / "summary"
    report = summarize_profiles(
        [("ar", ar), ("diffusion", diffusion)], output, top=10
    )

    assert report.exists()
    assert (output / "ar_operator_summary.csv").exists()
    assert (output / "diffusion_operator_summary.csv").exists()
    text = report.read_text(encoding="utf-8")
    assert "MatMul" in text
    assert "10.000 ms" not in text
    assert "0.010 ms" in text


def test_profile_without_kernels_keeps_synchronized_region_wall_time(tmp_path):
    profile = tmp_path / "cache_copy"
    profile.mkdir()
    (profile / "region_timing.json").write_text(
        json.dumps({"synchronized_region_wall_us": 1250.0}), encoding="utf-8"
    )

    output = tmp_path / "summary"
    report = summarize_profiles([("cache_copy", profile)], output, top=10)

    totals = json.loads((output / "profile_totals.json").read_text("utf-8"))
    assert totals["profiles"]["cache_copy"]["kernel_records"] == 0
    assert (
        totals["profiles"]["cache_copy"]["synchronized_region_wall_us"]
        == 1250.0
    )
    assert "1.250" in report.read_text(encoding="utf-8")
