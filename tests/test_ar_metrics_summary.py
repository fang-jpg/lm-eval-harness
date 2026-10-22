import json

from scripts.summarize_ar_metrics import (
    _latest_results_file,
    _load_json,
    _write_summary,
)


def test_ar_summary_combines_accuracy_forwards_throughput_and_memory(tmp_path):
    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    metric_file = metrics_root / "rank_0.json"
    metric_payload = {
        "rank": 0,
        "world_size": 1,
        "settings": {"decoder": "huggingface-ar-generate", "batch_size": 1},
        "forward_counts": {"prefill": 2, "decode": 6, "scoring": 0},
        "generation_requests": 2,
        "generation_batches": 2,
        "raw_generated_tokens": 8,
        "generation_seconds": 3.0,
        "decode_seconds": 2.0,
        "peak_memory_allocated_bytes": 3 * 1024**3,
        "peak_memory_reserved_bytes": 4 * 1024**3,
        "requests": [
            {
                "request_index": 0,
                "batch_size": 1,
                "generated_tokens": 4,
                "prefill_forward_count": 1,
                "decode_forward_count": 3,
                "actual_forward_count": 4,
                "generation_seconds": 1.5,
                "decode_seconds": 1.0,
            }
        ],
    }
    metric_file.write_text(json.dumps(metric_payload), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_file = results_dir / "results_2026-10-22T00-00-00.json"
    results_payload = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.5959}}
    }
    results_file.write_text(json.dumps(results_payload), encoding="utf-8")

    output_path = tmp_path / "summary.log"
    _write_summary(
        output_path=output_path,
        task="gsm8k",
        model="test-model",
        source_log="source.log",
        results_file=_latest_results_file(results_dir),
        results=_load_json(results_file),
        metric_files=[metric_file],
        payloads=[metric_payload],
        metrics_root=metrics_root,
    )

    summary = output_path.read_text(encoding="utf-8")
    assert "results.gsm8k.exact_match,strict-match=0.5959" in summary
    assert "prefill_forward_count=2" in summary
    assert "decode_forward_count=6" in summary
    assert "actual_forward_count=8" in summary
    assert "actual_forwards_per_request=4.000000" in summary
    assert "decode_tok_per_s_device_average=4.000000" in summary
    assert "peak_memory_allocated_gb_max_rank=3.000000" in summary
    assert "peak_memory_reserved_gb_max_rank=4.000000" in summary
    assert "[per_generation_batch]" in summary
