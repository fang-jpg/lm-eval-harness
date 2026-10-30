"""Opt-in Ascend operator profiling for generation backends.

The profiler is disabled unless ``HF_NPU_PROFILE_DIR`` is set.  It supports
profiling either one complete generation request or one named region (for
example an AR decode forward or a diffusion denoising forward).  Restricting a
trace to one representative region keeps CANN Level1 traces small enough to
inspect while still exposing per-operator and per-kernel timings.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch


eval_logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _parse_ranks(value: str) -> set[int] | None:
    normalized = value.strip().lower()
    if normalized in {"", "all", "-1"}:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


class NPUOperatorProfiler:
    """Create at most one bounded torch_npu profiler trace per process."""

    def __init__(self, backend: str, rank: int | None = None) -> None:
        self.backend = backend
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        profile_dir = os.environ.get("HF_NPU_PROFILE_DIR", "").strip()
        self.output_root = Path(profile_dir).expanduser() if profile_dir else None
        self.scope = os.environ.get("HF_NPU_PROFILE_SCOPE", "forward").strip().lower()
        self.request_start = _env_int("HF_NPU_PROFILE_REQUEST_START", 0)
        self.request_count = _env_int("HF_NPU_PROFILE_REQUEST_COUNT", 1)
        self.target = os.environ.get("HF_NPU_PROFILE_TARGET", "").strip()
        self.target_occurrence = _env_int("HF_NPU_PROFILE_TARGET_OCCURRENCE", 0)
        self.profile_level = os.environ.get("HF_NPU_PROFILE_LEVEL", "level1").strip().lower()
        self.export_type = os.environ.get("HF_NPU_PROFILE_EXPORT_TYPE", "text").strip().lower()
        self.with_cpu = _env_bool("HF_NPU_PROFILE_WITH_CPU", True)
        self.record_shapes = _env_bool("HF_NPU_PROFILE_RECORD_SHAPES", True)
        self.profile_memory = _env_bool("HF_NPU_PROFILE_MEMORY", False)
        self.with_stack = _env_bool("HF_NPU_PROFILE_WITH_STACK", False)
        self.data_simplification = _env_bool(
            "HF_NPU_PROFILE_DATA_SIMPLIFICATION", False
        )
        ranks = _parse_ranks(os.environ.get("HF_NPU_PROFILE_RANKS", "0"))
        self.enabled = self.output_root is not None and (
            ranks is None or self.rank in ranks
        )
        self._region_occurrences: Counter[str] = Counter()
        self._current_request_index = -1
        self._profile_created = False
        self._active_profile = False

        if not self.enabled:
            return
        if self.scope not in {"request", "forward"}:
            raise ValueError("HF_NPU_PROFILE_SCOPE must be 'request' or 'forward'.")
        if self.request_start < 0 or self.request_count <= 0:
            raise ValueError(
                "HF_NPU_PROFILE_REQUEST_START must be non-negative and "
                "HF_NPU_PROFILE_REQUEST_COUNT must be positive."
            )
        if self.target_occurrence < 0:
            raise ValueError("HF_NPU_PROFILE_TARGET_OCCURRENCE must be non-negative.")
        if self.scope == "forward" and not self.target:
            raise ValueError(
                "HF_NPU_PROFILE_TARGET is required when HF_NPU_PROFILE_SCOPE=forward."
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        eval_logger.info(
            "Enabled Ascend operator profiling: backend=%s rank=%d scope=%s "
            "target=%s occurrence=%d output=%s",
            self.backend,
            self.rank,
            self.scope,
            self.target or "n/a",
            self.target_occurrence,
            self.output_root,
        )

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    @contextmanager
    def request(self, request_index: int) -> Iterator[None]:
        previous_request_index = self._current_request_index
        self._current_request_index = request_index
        if self.scope == "forward":
            # Occurrences are request-local, so a warm-up request does not
            # shift the selected decode/denoise forward in the profiled one.
            self._region_occurrences.clear()
        selected = (
            self.enabled
            and self.scope == "request"
            and self.request_start
            <= request_index
            < self.request_start + self.request_count
            and not self._profile_created
        )
        label = f"{self.backend}_request_{request_index}"
        try:
            with self._profile_context(label, selected=selected):
                yield
        finally:
            self._current_request_index = previous_request_index

    @contextmanager
    def region(self, label: str) -> Iterator[None]:
        occurrence = int(self._region_occurrences[label])
        self._region_occurrences[label] += 1
        selected = (
            self.enabled
            and self.scope == "forward"
            and self.request_start
            <= self._current_request_index
            < self.request_start + self.request_count
            and label == self.target
            and occurrence == self.target_occurrence
            and not self._profile_created
        )
        annotated_label = f"{label}_{occurrence}"
        with self._profile_context(annotated_label, selected=selected):
            yield

    @contextmanager
    def _profile_context(self, label: str, *, selected: bool) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        # A request-level trace owns the profiler. Nested regions only add
        # record_function markers to that trace.
        if self._active_profile:
            with torch.autograd.profiler.record_function(label):
                yield
            return

        if not selected:
            yield
            return

        profiler, trace_dir = self._create_profiler(label)
        self._profile_created = True
        self._synchronize()
        elapsed_us = 0.0
        self._active_profile = True
        try:
            with profiler:
                start = time.perf_counter()
                try:
                    with torch.autograd.profiler.record_function(label):
                        yield
                finally:
                    # Ensure asynchronously launched kernels finish before the
                    # profiler is stopped and exports its trace.
                    self._synchronize()
                    elapsed_us = (time.perf_counter() - start) * 1_000_000
        finally:
            self._active_profile = False
            (trace_dir / "region_timing.json").write_text(
                json.dumps(
                    {
                        "backend": self.backend,
                        "rank": self.rank,
                        "label": label,
                        "synchronized_region_wall_us": elapsed_us,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        eval_logger.info(
            "Ascend operator trace complete: backend=%s rank=%d label=%s",
            self.backend,
            self.rank,
            label,
        )

    def _create_profiler(self, label: str):
        try:
            import torch_npu.profiler as npu_profiler
        except ImportError as error:
            raise RuntimeError(
                "HF_NPU_PROFILE_DIR is set, but torch_npu.profiler is unavailable."
            ) from error

        level_map = {
            "level_none": npu_profiler.ProfilerLevel.Level_none,
            "level0": npu_profiler.ProfilerLevel.Level0,
            "level1": npu_profiler.ProfilerLevel.Level1,
            "level2": npu_profiler.ProfilerLevel.Level2,
        }
        if self.profile_level not in level_map:
            raise ValueError(
                "HF_NPU_PROFILE_LEVEL must be level_none, level0, level1, or level2."
            )
        export_map = {
            "text": npu_profiler.ExportType.Text,
            "db": npu_profiler.ExportType.Db,
        }
        if self.export_type not in export_map:
            raise ValueError("HF_NPU_PROFILE_EXPORT_TYPE must be text or db.")

        activities = [npu_profiler.ProfilerActivity.NPU]
        if self.with_cpu:
            activities.append(npu_profiler.ProfilerActivity.CPU)
        experimental_config = npu_profiler._ExperimentalConfig(
            aic_metrics=npu_profiler.AiCMetrics.PipeUtilization,
            profiler_level=level_map[self.profile_level],
            export_type=export_map[self.export_type],
            data_simplification=self.data_simplification,
        )
        trace_dir = self.output_root / self.backend / f"rank_{self.rank}" / label
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler = npu_profiler.profile(
            activities=activities,
            on_trace_ready=npu_profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            experimental_config=experimental_config,
        )
        metadata: dict[str, Any] = {
            "backend": self.backend,
            "rank": self.rank,
            "scope": self.scope,
            "label": label,
            "target": self.target,
            "target_occurrence": self.target_occurrence,
        }
        try:
            profiler.add_metadata_json("hf_npu_profile", json.dumps(metadata))
        except Exception as error:  # pragma: no cover - version-specific API
            eval_logger.warning("Could not attach profiler metadata: %s", error)
        (trace_dir / "profile_selection.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return profiler, trace_dir

    @staticmethod
    def _synchronize() -> None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()


def disabled_profile_region():
    """Compatibility helper for callers that do not own a profiler."""

    return nullcontext()


__all__ = ["NPUOperatorProfiler", "disabled_profile_region"]
