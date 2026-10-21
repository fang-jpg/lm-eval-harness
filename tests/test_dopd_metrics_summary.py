import json

from scripts.summarize_dopd_metrics import (
    _latest_results_file,
    _load_json,
    _write_summary,
)


def test_dopd_summary_uses_saving_forward_count(tmp_path):
    metrics_root = tmp_path / "metrics"
    attempt_dir = metrics_root / "attempt_1"
    attempt_dir.mkdir(parents=True)
    metric_file = attempt_dir / "rank_0.json"
    metric_payload = {
        "rank": 0,
        "world_size": 1,
        "settings": {
            "decoder": "dopd-response-aligned",
            "block_length": 4,
            "denoising_steps_per_block": 4,
        },
        "forward_counts": {"prefill": 1, "denoise": 8, "saving": 1},
        "generation_requests": 1,
        "raw_generated_tokens": 8,
        "decode_seconds": 2.0,
        "requests": [
            {
                "request_index": 0,
                "generated_tokens": 8,
                "decode_seconds": 2.0,
                "block_steps": [4, 4],
                "block_allocated_steps": [4, 4],
            }
        ],
    }
    metric_file.write_text(json.dumps(metric_payload), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_file = results_dir / "results_2026-10-21T00-00-00.json"
    results_payload = {"results": {"gsm8k": {"exact_match": 0.5}}}
    results_file.write_text(json.dumps(results_payload), encoding="utf-8")

    output_path = tmp_path / "summary.log"
    _write_summary(
        output_path=output_path,
        task="gsm8k",
        model="dopd-test-model",
        source_log="source.log",
        results_file=_latest_results_file(results_dir),
        results=_load_json(results_file),
        metric_files=[metric_file],
        payloads=[metric_payload],
        metrics_root=metrics_root,
    )

    summary = output_path.read_text(encoding="utf-8")
    assert "HF d-OPD evaluation summary" in summary
    assert "results.gsm8k.exact_match=0.5" in summary
    assert "saving_forward_count=1" in summary
    assert "actual_forward_count=10" in summary
    assert "steps_histogram=4:2" in summary
