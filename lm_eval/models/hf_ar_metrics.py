"""Metrics-only wrapper for ordinary Hugging Face autoregressive generation.

Generation is delegated unchanged to :class:`HFLM`. This backend only counts
top-level model forwards, measures decode time after the prompt-prefill forward,
and records peak accelerator memory for evaluation summaries.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)


@register_model("hf-ar-metrics")
class HFARMetricsLM(HFLM):
    """Run the stock AR decoder while collecting rank-local performance data."""

    def __init__(self, *args, **kwargs) -> None:
        self._ar_metrics_dir = os.environ.get("HF_AR_METRICS_DIR")
        self._ar_perf_metrics = self._new_ar_perf_metrics()
        super().__init__(*args, **kwargs)

        if self.backend != "causal":
            raise ValueError("hf-ar-metrics only supports decoder-only AR models.")
        if self._ar_metrics_dir:
            self._reset_peak_memory_stats()
            atexit.register(self._write_ar_perf_metrics)

        eval_logger.info(
            "Initialized hf-ar-metrics; metrics_dir=%s",
            self._ar_metrics_dir,
        )

    @staticmethod
    def _new_ar_perf_metrics() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "forward_counts": {"prefill": 0, "decode": 0, "scoring": 0},
            "generation_requests": 0,
            "generation_batches": 0,
            "raw_generated_tokens": 0,
            "generation_seconds": 0.0,
            "decode_seconds": 0.0,
            "requests": [],
        }

    def _metrics_synchronize(self) -> None:
        if not getattr(self, "_ar_metrics_dir", None):
            return
        device_type = self.device.type
        if device_type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elif device_type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()

    def _reset_peak_memory_stats(self) -> None:
        try:
            if self.device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.reset_peak_memory_stats(self.device)
            elif self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        except Exception as error:
            eval_logger.warning("Could not reset peak device-memory stats: %s", error)

    def _peak_memory_stats(self) -> tuple[int, int]:
        try:
            if self.device.type == "npu" and hasattr(torch, "npu"):
                return (
                    int(torch.npu.max_memory_allocated(self.device)),
                    int(torch.npu.max_memory_reserved(self.device)),
                )
            if self.device.type == "cuda" and torch.cuda.is_available():
                return (
                    int(torch.cuda.max_memory_allocated(self.device)),
                    int(torch.cuda.max_memory_reserved(self.device)),
                )
        except Exception as error:
            eval_logger.warning("Could not read peak device-memory stats: %s", error)
        return 0, 0

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_ar_metrics_dir", None):
            self._ar_perf_metrics["forward_counts"]["scoring"] += 1
        return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        if not getattr(self, "_ar_metrics_dir", None):
            return super()._model_generate(
                context=context,
                max_length=max_length,
                stop=stop,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        forward_count = 0
        decode_start: float | None = None

        def record_top_level_forward(module, args, kwargs, output) -> None:
            del module, args, kwargs, output
            nonlocal forward_count, decode_start
            if forward_count == 0:
                self._ar_perf_metrics["forward_counts"]["prefill"] += 1
                # The first top-level forward is prompt prefill. Synchronizing
                # here gives decode-only wall time without changing generation.
                self._metrics_synchronize()
                decode_start = time.perf_counter()
            else:
                self._ar_perf_metrics["forward_counts"]["decode"] += 1
            forward_count += 1

        self._metrics_synchronize()
        generation_start = time.perf_counter()
        hook = self.model.register_forward_hook(
            record_top_level_forward,
            with_kwargs=True,
        )
        try:
            generated = super()._model_generate(
                context=context,
                max_length=max_length,
                stop=stop,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        finally:
            hook.remove()
        self._metrics_synchronize()
        generation_end = time.perf_counter()

        batch_size = int(context.shape[0])
        generated_per_sequence = max(0, int(generated.shape[1] - context.shape[1]))
        generated_tokens = batch_size * generated_per_sequence
        decode_seconds = (
            generation_end - decode_start
            if decode_start is not None
            else generation_end - generation_start
        )
        metrics = self._ar_perf_metrics
        request_index = metrics["generation_requests"]
        metrics["generation_requests"] += batch_size
        metrics["generation_batches"] += 1
        metrics["raw_generated_tokens"] += generated_tokens
        metrics["generation_seconds"] += generation_end - generation_start
        metrics["decode_seconds"] += decode_seconds
        metrics["requests"].append(
            {
                "request_index": request_index,
                "batch_size": batch_size,
                "generated_tokens": generated_tokens,
                "prefill_forward_count": 1 if forward_count else 0,
                "decode_forward_count": max(0, forward_count - 1),
                "actual_forward_count": forward_count,
                "generation_seconds": generation_end - generation_start,
                "decode_seconds": decode_seconds,
            }
        )
        return generated

    def _write_ar_perf_metrics(self) -> None:
        metrics_dir = getattr(self, "_ar_metrics_dir", None)
        metrics = getattr(self, "_ar_perf_metrics", None)
        if not metrics_dir or metrics is None:
            return
        try:
            output_dir = Path(metrics_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_synchronize()
            peak_allocated, peak_reserved = self._peak_memory_stats()
            rank = int(self.rank)
            payload = {
                **metrics,
                "rank": rank,
                "world_size": int(self.world_size),
                "peak_memory_allocated_bytes": peak_allocated,
                "peak_memory_reserved_bytes": peak_reserved,
                "settings": {
                    "decoder": "huggingface-ar-generate",
                    "batch_size": self.batch_size,
                },
            }
            decode_seconds = float(payload["decode_seconds"])
            payload["decode_tok_per_s"] = (
                payload["raw_generated_tokens"] / decode_seconds
                if decode_seconds > 0
                else 0.0
            )
            output_path = output_dir / f"rank_{rank}.json"
            temporary_path = output_dir / f"rank_{rank}.{os.getpid()}.tmp"
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            temporary_path.replace(output_path)
        except Exception as error:
            eval_logger.warning("Could not write AR performance metrics: %s", error)
