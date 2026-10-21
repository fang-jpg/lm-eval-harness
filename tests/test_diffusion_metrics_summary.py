import json

from scripts.summarize_diffusion_metrics import (
    _latest_results_file,
    _load_json,
    _write_summary,
)


def test_diffusion_summary_combines_accuracy_forwards_and_block_steps(tmp_path):
    metrics_root = tmp_path / "metrics"
    attempt_dir = metrics_root / "attempt_1"
    attempt_dir.mkdir(parents=True)
    metric_file = attempt_dir / "rank_0.json"
    metric_payload = {
        "rank": 0,
        "world_size": 1,
        "settings": {
            "block_length": 4,
            "diffusion_steps": 3,
            "threshold": 0.8,
        },
        "forward_counts": {
            "prefill": 1,
            "denoise": 3,
            "causal": 1,
            "scoring": 0,
        },
        "generation_requests": 1,
        "raw_generated_tokens": 4,
        "decode_seconds": 2.0,
        "peak_memory_allocated_bytes": 3 * 1024**3,
        "peak_memory_reserved_bytes": 4 * 1024**3,
        "requests": [
            {
                "request_index": 0,
                "generated_tokens": 4,
                "decode_seconds": 2.0,
                "block_steps": [3],
                "block_allocated_steps": [3],
                "block_progress_fallback_tokens": [1],
                "block_stalled_steps": [1],
                "block_traces": [
                    {
                        "block_index": 0,
                        "response_start": 0,
                        "allocated_steps": 3,
                        "actual_steps": 2,
                        "seed_token": {
                            "block_position": 0,
                            "response_position": 0,
                            "token_id": 10,
                            "token_text": "A",
                            "confidence": 0.9,
                        },
                        "steps": [
                            {
                                "step": 1,
                                "remaining_before": 3,
                                "accepted_by_threshold": 1,
                                "progress_fallback_tokens": 0,
                                "token_probabilities": [
                                    {
                                        "block_position": 1,
                                        "response_position": 1,
                                        "proposed_token_id": 11,
                                        "proposed_token_text": "B",
                                        "probability": 0.85,
                                        "above_threshold": True,
                                        "accepted": True,
                                        "accepted_for_progress": False,
                                    }
                                ],
                            }
                        ],
                        "final_tokens": [
                            {
                                "block_position": 0,
                                "response_position": 0,
                                "token_id": 10,
                                "token_text": "A",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    metric_file.write_text(json.dumps(metric_payload), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_file = results_dir / "results_2026-10-13T00-00-00.json"
    results_payload = {"results": {"boolq-seq2seq": {"exact_match": 0.75}}}
    results_file.write_text(json.dumps(results_payload), encoding="utf-8")

    output_path = tmp_path / "summary.log"
    _write_summary(
        output_path=output_path,
        task="boolq-seq2seq",
        model="test-model",
        source_log="source.log",
        results_file=_latest_results_file(results_dir),
        results=_load_json(results_file),
        metric_files=[metric_file],
        metrics=[metric_payload],
        metrics_root=metrics_root,
    )

    summary = output_path.read_text(encoding="utf-8")
    assert "results.boolq-seq2seq.exact_match=0.75" in summary
    assert "actual_forward_count=5" in summary
    assert "actual_forwards_per_request=5.000000" in summary
    assert "decode_tok_per_s_device_average=2.000000" in summary
    assert "peak_memory_allocated_gb_max_rank=3.000000" in summary
    assert "steps_histogram=3:1" in summary
    assert "block_steps=[3]" in summary
    assert "progress_fallback_tokens=[1]" in summary
    assert "[block_decode_probability_traces]" in summary
    assert "probability=0.85" in summary
    assert "trace_block_count=1" in summary
