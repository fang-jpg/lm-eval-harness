#!/usr/bin/env python3
"""Write an accuracy and runtime summary for the independent d-OPD backend."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _latest_results_file(results_dir: Path) -> Path:
    candidates = list(results_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


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
        if isinstance(section, dict) and isinstance(section.get(task), dict):
            sections.append((f"{section_name}.{task}", section[task]))
    if sections:
        return sections
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
    payloads: list[dict[str, Any]],
    metrics_root: Path,
) -> None:
    forward_counts: Counter[str] = Counter()
    step_histogram: Counter[int] = Counter()
    steps_by_block_index: dict[int, list[int]] = defaultdict(list)
    rank_decode_seconds: Counter[int] = Counter()
    generated_tokens = 0
    generation_requests = 0
    decode_seconds = 0.0
    settings = []

    for payload in payloads:
        forward_counts.update(payload.get("forward_counts", {}))
        generated_tokens += int(payload.get("raw_generated_tokens", 0))
        generation_requests += int(payload.get("generation_requests", 0))
        elapsed = float(payload.get("decode_seconds", 0.0))
        decode_seconds += elapsed
        rank_decode_seconds[int(payload.get("rank", 0))] += elapsed
        if payload.get("settings") not in settings:
            settings.append(payload.get("settings"))
        for request in payload.get("requests", []):
            for block_index, raw_steps in enumerate(request.get("block_steps", [])):
                steps = int(raw_steps)
                step_histogram[steps] += 1
                steps_by_block_index[block_index].append(steps)

    longest_rank_seconds = max(rank_decode_seconds.values(), default=0.0)
    device_average = generated_tokens / decode_seconds if decode_seconds > 0 else 0.0
    all_ranks_estimate = (
        generated_tokens / longest_rank_seconds if longest_rank_seconds > 0 else 0.0
    )
    total_blocks = sum(step_histogram.values())
    total_block_steps = sum(
        steps * count for steps, count in step_histogram.items()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("HF d-OPD evaluation summary\n")
        output.write(f"created_at={datetime.now().astimezone().isoformat()}\n")
        output.write(f"task={task}\n")
        output.write(f"model={model}\n")
        output.write(f"source_log={source_log}\n")
        output.write(f"results_json={results_file}\n")
        output.write(f"rank_metric_files={len(metric_files)}\n")
        for index, setting in enumerate(settings):
            output.write(
                f"settings[{index}]="
                f"{json.dumps(setting, ensure_ascii=False, sort_keys=True)}\n"
            )

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
        output.write(f"prefill_forward_count={forward_counts['prefill']}\n")
        output.write(f"denoise_forward_count={forward_counts['denoise']}\n")
        output.write(f"saving_forward_count={forward_counts['saving']}\n")
        output.write(f"actual_forward_count={sum(forward_counts.values())}\n")
        output.write(f"decode_device_seconds={decode_seconds:.9f}\n")
        output.write(f"decode_tok_per_s_device_average={device_average:.6f}\n")
        output.write(
            f"decode_tok_per_s_all_ranks_estimate={all_ranks_estimate:.6f}\n"
        )

        output.write("\n[block_step_summary]\n")
        output.write(f"total_blocks={total_blocks}\n")
        if total_blocks:
            output.write(f"steps_min={min(step_histogram)}\n")
            output.write(f"steps_max={max(step_histogram)}\n")
            output.write(f"steps_mean={total_block_steps / total_blocks:.6f}\n")
        output.write(
            "steps_histogram="
            + ",".join(
                f"{steps}:{count}"
                for steps, count in sorted(step_histogram.items())
            )
            + "\n"
        )
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
        if len(metric_files) != len(payloads):
            raise ValueError("Metric file and payload counts do not match.")
        for metric_file, payload in zip(metric_files, payloads):
            relative_file = metric_file.relative_to(metrics_root)
            rank = int(payload.get("rank", 0))
            for request in payload.get("requests", []):
                output.write(
                    "metrics_file={file} rank={rank} request={request_index} "
                    "generated_tokens={tokens} decode_seconds={seconds:.9f} "
                    "block_steps={steps} allocated_steps={allocated}\n".format(
                        file=relative_file.as_posix(),
                        rank=rank,
                        request_index=request.get("request_index", 0),
                        tokens=request.get("generated_tokens", 0),
                        seconds=float(request.get("decode_seconds", 0.0)),
                        steps=json.dumps(request.get("block_steps", [])),
                        allocated=json.dumps(
                            request.get("block_allocated_steps", [])
                        ),
                    )
                )
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
    results_file = _latest_results_file(args.results_dir.resolve())
    _write_summary(
        output_path=args.output.resolve(),
        task=args.task,
        model=args.model,
        source_log=args.source_log,
        results_file=results_file,
        results=_load_json(results_file),
        metric_files=metric_files,
        payloads=[_load_json(path) for path in metric_files],
        metrics_root=metrics_root,
    )
    print(f"d-OPD evaluation summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
