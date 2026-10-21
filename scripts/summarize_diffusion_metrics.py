#!/usr/bin/env python3
"""Combine lm-eval accuracy and rank-local diffusion performance metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _latest_results_file(results_dir: Path) -> Path:
    candidates = list(results_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8g}"
    return str(value)


def _metric_sections(results: dict[str, Any], task: str) -> list[tuple[str, dict]]:
    sections = []
    for section_name in ("groups", "results"):
        section = results.get(section_name, {})
        if not isinstance(section, dict):
            continue
        if task in section and isinstance(section[task], dict):
            sections.append((f"{section_name}.{task}", section[task]))

    if sections:
        return sections

    # A task alias is not always identical to the stored group name. Fall back
    # to all aggregate groups, or to individual results when there is no group.
    groups = results.get("groups", {})
    if isinstance(groups, dict) and groups:
        return [
            (f"groups.{name}", metrics)
            for name, metrics in groups.items()
            if isinstance(metrics, dict)
        ]
    task_results = results.get("results", {})
    if isinstance(task_results, dict):
        return [
            (f"results.{name}", metrics)
            for name, metrics in task_results.items()
            if isinstance(metrics, dict)
        ]
    return []


def _write_summary(
    *,
    output_path: Path,
    task: str,
    model: str,
    source_log: str,
    results_file: Path,
    results: dict[str, Any],
    metric_files: list[Path],
    metrics: list[dict[str, Any]],
    metrics_root: Path,
) -> None:
    forward_counts: Counter[str] = Counter()
    step_histogram: Counter[int] = Counter()
    steps_by_block_index: dict[int, list[int]] = defaultdict(list)
    generated_tokens = 0
    generation_requests = 0
    decode_seconds = 0.0
    progress_fallback_tokens = 0
    stalled_steps = 0
    total_blocks = 0
    rank_decode_seconds: Counter[int] = Counter()
    peak_memory_allocated = 0
    peak_memory_reserved = 0
    settings = []

    for payload in metrics:
        forward_counts.update(payload.get("forward_counts", {}))
        generated_tokens += int(payload.get("raw_generated_tokens", 0))
        generation_requests += int(payload.get("generation_requests", 0))
        payload_decode_seconds = float(payload.get("decode_seconds", 0.0))
        decode_seconds += payload_decode_seconds
        rank_decode_seconds[int(payload.get("rank", 0))] += payload_decode_seconds
        peak_memory_allocated = max(
            peak_memory_allocated,
            int(payload.get("peak_memory_allocated_bytes", 0)),
        )
        peak_memory_reserved = max(
            peak_memory_reserved,
            int(payload.get("peak_memory_reserved_bytes", 0)),
        )
        if payload.get("settings") not in settings:
            settings.append(payload.get("settings"))

        for request in payload.get("requests", []):
            block_steps = request.get("block_steps", [])
            block_progress = request.get("block_progress_fallback_tokens", [])
            block_stalled = request.get("block_stalled_steps", [])
            total_blocks += len(block_steps)
            progress_fallback_tokens += sum(int(value) for value in block_progress)
            stalled_steps += sum(int(value) for value in block_stalled)
            for block_index, value in enumerate(block_steps):
                steps = int(value)
                step_histogram[steps] += 1
                steps_by_block_index[block_index].append(steps)

    actual_forward_count = sum(forward_counts.values())
    decode_forward_count = forward_counts["denoise"] + forward_counts["causal"]
    token_device_second = (
        generated_tokens / decode_seconds if decode_seconds > 0 else 0.0
    )
    longest_rank_seconds = max(rank_decode_seconds.values(), default=0.0)
    aggregate_tok_per_s = (
        generated_tokens / longest_rank_seconds if longest_rank_seconds > 0 else 0.0
    )
    observed_step_count = sum(step_histogram.values())
    observed_step_sum = sum(
        steps * count for steps, count in step_histogram.items()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("HF diffusion evaluation summary\n")
        output.write(f"created_at={datetime.now().astimezone().isoformat()}\n")
        output.write(f"task={task}\n")
        output.write(f"model={model}\n")
        output.write(f"source_log={source_log}\n")
        output.write(f"results_json={results_file}\n")
        output.write(f"rank_metric_files={len(metric_files)}\n")
        for index, setting in enumerate(settings):
            setting_text = json.dumps(
                setting, ensure_ascii=False, sort_keys=True
            )
            output.write(f"settings[{index}]={setting_text}\n")

        output.write("\n[accuracy_and_evaluation_metrics]\n")
        metric_sections = _metric_sections(results, task)
        if not metric_sections:
            output.write("no_metrics_found=true\n")
        for section_name, section_metrics in metric_sections:
            for metric_name, value in section_metrics.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    output.write(
                        f"{section_name}.{metric_name}={_format_value(value)}\n"
                    )

        output.write("\n[performance]\n")
        output.write(f"generation_requests={generation_requests}\n")
        output.write(f"raw_generated_tokens={generated_tokens}\n")
        for kind in ("prefill", "denoise", "causal", "scoring"):
            output.write(f"{kind}_forward_count={forward_counts[kind]}\n")
        output.write(f"actual_forward_count={actual_forward_count}\n")
        output.write(f"decode_forward_count={decode_forward_count}\n")
        output.write(
            "actual_forwards_per_request="
            f"{actual_forward_count / generation_requests:.6f}\n"
            if generation_requests
            else "actual_forwards_per_request=0.000000\n"
        )
        output.write(
            "decode_forwards_per_request="
            f"{decode_forward_count / generation_requests:.6f}\n"
            if generation_requests
            else "decode_forwards_per_request=0.000000\n"
        )
        output.write(f"decode_device_seconds={decode_seconds:.9f}\n")
        output.write(f"decode_tok_per_s_device_average={token_device_second:.6f}\n")
        output.write(
            f"decode_tok_per_s_all_ranks_estimate={aggregate_tok_per_s:.6f}\n"
        )
        gibibyte = 1024**3
        output.write(
            "peak_memory_allocated_gb_max_rank="
            f"{peak_memory_allocated / gibibyte:.6f}\n"
        )
        output.write(
            "peak_memory_reserved_gb_max_rank="
            f"{peak_memory_reserved / gibibyte:.6f}\n"
        )

        output.write("\n[block_step_summary]\n")
        output.write(f"total_blocks={total_blocks}\n")
        output.write(f"progress_fallback_tokens={progress_fallback_tokens}\n")
        output.write(f"stalled_threshold_steps={stalled_steps}\n")
        if observed_step_count:
            output.write(f"steps_min={min(step_histogram)}\n")
            output.write(f"steps_max={max(step_histogram)}\n")
            output.write(
                f"steps_mean={observed_step_sum / observed_step_count:.6f}\n"
            )
        histogram_text = ",".join(
            f"{steps}:{count}" for steps, count in sorted(step_histogram.items())
        )
        output.write(f"steps_histogram={histogram_text}\n")
        for block_index, values in sorted(steps_by_block_index.items()):
            output.write(
                "block_index={index} count={count} steps_min={minimum} "
                "steps_max={maximum} steps_mean={mean:.6f}\n".format(
                    index=block_index,
                    count=len(values),
                    minimum=min(values),
                    maximum=max(values),
                    mean=sum(values) / len(values),
                )
            )

        output.write("\n[per_request_block_steps]\n")
        for metric_file, payload in zip(metric_files, metrics):
            relative_file = metric_file.relative_to(metrics_root)
            rank = int(payload.get("rank", 0))
            for request in payload.get("requests", []):
                output.write(
                    "metrics_file={file} rank={rank} request={request_index} "
                    "generated_tokens={tokens} decode_seconds={seconds:.9f} "
                    "block_steps={steps} allocated_steps={allocated} "
                    "progress_fallback_tokens={progress} "
                    "stalled_steps={stalled}\n".format(
                        file=relative_file.as_posix(),
                        rank=rank,
                        request_index=request.get("request_index", 0),
                        tokens=request.get("generated_tokens", 0),
                        seconds=float(request.get("decode_seconds", 0.0)),
                        steps=json.dumps(request.get("block_steps", [])),
                        allocated=json.dumps(
                            request.get("block_allocated_steps", [])
                        ),
                        progress=json.dumps(
                            request.get("block_progress_fallback_tokens", [])
                        ),
                        stalled=json.dumps(request.get("block_stalled_steps", [])),
                    )
                )

        output.write("\n[block_decode_probability_traces]\n")
        trace_count = 0
        for metric_file, payload in zip(metric_files, metrics):
            relative_file = metric_file.relative_to(metrics_root)
            rank = int(payload.get("rank", 0))
            for request in payload.get("requests", []):
                request_index = request.get("request_index", 0)
                for trace in request.get("block_traces", []):
                    trace_count += 1
                    output.write(
                        "trace metrics_file={file} rank={rank} request={request} "
                        "block={block} response_start={start} "
                        "allocated_steps={allocated} actual_steps={actual}\n".format(
                            file=relative_file.as_posix(),
                            rank=rank,
                            request=request_index,
                            block=trace.get("block_index", 0),
                            start=trace.get("response_start", 0),
                            allocated=trace.get("allocated_steps", 0),
                            actual=trace.get("actual_steps", 0),
                        )
                    )
                    seed = trace.get("seed_token", {})
                    output.write(
                        "  seed position={position} response_position={response} "
                        "token_id={token_id} token={token} probability={probability}\n".format(
                            position=seed.get("block_position", 0),
                            response=seed.get("response_position", 0),
                            token_id=seed.get("token_id"),
                            token=json.dumps(
                                seed.get("token_text"), ensure_ascii=False
                            ),
                            probability=seed.get("confidence"),
                        )
                    )
                    for step in trace.get("steps", []):
                        output.write(
                            "  step={step} remaining_before={remaining} "
                            "accepted_by_threshold={accepted} "
                            "progress_fallback_tokens={progress}\n".format(
                                step=step.get("step"),
                                remaining=step.get("remaining_before"),
                                accepted=step.get("accepted_by_threshold"),
                                progress=step.get("progress_fallback_tokens"),
                            )
                        )
                        for token in step.get("token_probabilities", []):
                            output.write(
                                "    position={position} response_position={response} "
                                "proposed_id={token_id} token={token_text} "
                                "probability={probability} above_threshold={above} "
                                "accepted={accepted} "
                                "accepted_for_progress={progress}\n".format(
                                    position=token.get("block_position"),
                                    response=token.get("response_position"),
                                    token_id=token.get("proposed_token_id"),
                                    token_text=json.dumps(
                                        token.get("proposed_token_text"),
                                        ensure_ascii=False,
                                    ),
                                    probability=token.get("probability"),
                                    above=token.get("above_threshold"),
                                    accepted=token.get("accepted"),
                                    progress=token.get("accepted_for_progress"),
                                )
                            )
                    final_tokens = trace.get("final_tokens", [])
                    output.write(
                        "  final_token_ids="
                        + json.dumps(
                            [token.get("token_id") for token in final_tokens]
                        )
                        + "\n"
                    )
                    output.write(
                        "  final_token_text="
                        + json.dumps(
                            [token.get("token_text") for token in final_tokens],
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        output.write(f"trace_block_count={trace_count}\n")

    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-log", default="")
    args = parser.parse_args()

    metrics_root = args.metrics_dir.resolve()
    metric_files = sorted(metrics_root.rglob("rank_*.json"))
    if not metric_files:
        raise FileNotFoundError(f"No rank_*.json found under {metrics_root}")
    metrics = [_load_json(path) for path in metric_files]
    results_file = _latest_results_file(args.results_dir.resolve())
    results = _load_json(results_file)

    _write_summary(
        output_path=args.output.resolve(),
        task=args.task,
        model=args.model,
        source_log=args.source_log,
        results_file=results_file,
        results=results,
        metric_files=metric_files,
        metrics=metrics,
        metrics_root=metrics_root,
    )
    print(f"Diffusion evaluation summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
