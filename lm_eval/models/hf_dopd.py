"""Independent Hugging Face backend for d-OPD response-aligned decoding.

This module intentionally does not inherit from or call the hybrid
``hf-diffusion`` evaluator.  d-OPD uses JetEngine-style start-aligned prompt
blocks, all-MASK response blocks, a fixed denoising budget per block, and a
full-visible SAVING forward for every completed response block.
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
import transformers
from packaging.version import parse as parse_version

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)


def _as_optional_int(value: int | str | None) -> int | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def _crop_cache(cache: Any, length: int) -> None:
    """Restore the persistent prefix after a speculative denoising forward."""
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for layer_index in range(len(cache.key_cache)):
        cache.key_cache[layer_index] = cache.key_cache[layer_index][:, :, :length]
        cache.value_cache[layer_index] = cache.value_cache[layer_index][:, :, :length]
    cache._seen_tokens = length


@register_model("hf-dopd")
class HFDOPDLM(HFLM):
    """Evaluate a d-OPD checkpoint with its own response-aligned decoder."""

    def __init__(
        self,
        *args,
        mask_token_id: int | str | None = None,
        block_length: int | str = 4,
        denoising_steps_per_block: int | str = 4,
        diffusion_temperature: float | str = 1.0,
        top_k: int | str = 1,
        top_p: float | str = 1.0,
        seed: int | str = 42,
        **kwargs,
    ) -> None:
        if parse_version(transformers.__version__) < parse_version("5.2.0"):
            raise RuntimeError(
                "hf-dopd requires transformers>=5.2.0 for per-layer "
                f"attention-mask mappings; found {transformers.__version__}."
            )
        self._requested_mask_token_id = _as_optional_int(mask_token_id)
        self.block_length = int(block_length)
        self.denoising_steps_per_block = int(denoising_steps_per_block)
        self.diffusion_temperature = float(diffusion_temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self._generation_calls = 0
        self._perf_metrics = self._new_perf_metrics()
        self._metrics_dir = os.environ.get("HF_DOPD_METRICS_DIR")

        self._validate_dopd_args()
        kwargs["attn_implementation"] = "eager"
        super().__init__(*args, **kwargs)

        if self.backend != "causal":
            raise ValueError(
                "hf-dopd only supports decoder-only causal-LM checkpoints."
            )
        actual_attention = getattr(self.model.config, "_attn_implementation", None)
        if actual_attention != "eager":
            raise RuntimeError(
                "hf-dopd requires eager attention, but the loaded model uses "
                f"{actual_attention!r}."
            )

        configured_mask_id = getattr(self.config, "mask_token_id", None)
        tokenizer_mask_id = getattr(self.tokenizer, "mask_token_id", None)
        self.mask_token_id = self._requested_mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = configured_mask_id
        if self.mask_token_id is None:
            self.mask_token_id = tokenizer_mask_id
        if self.mask_token_id is None:
            raise ValueError(
                "No d-OPD mask token is configured. Add mask_token_id to "
                "config.json or pass mask_token_id=... in --model_args."
            )
        self.mask_token_id = int(self.mask_token_id)
        self._validate_mask_token()

        if self._metrics_dir:
            atexit.register(self._write_perf_metrics)

        eval_logger.info(
            "Initialized hf-dopd: mask_token_id=%d, block_length=%d, "
            "denoising_steps_per_block=%d, temperature=%.4f, top_k=%d, top_p=%.4f",
            self.mask_token_id,
            self.block_length,
            self.denoising_steps_per_block,
            self.diffusion_temperature,
            self.top_k,
            self.top_p,
        )

    def _get_config(self, *args, **kwargs) -> None:
        """Load the HF config and force the 4D-mask-compatible eager backend."""
        super()._get_config(*args, **kwargs)
        self._config._attn_implementation = "eager"
        self._config.attn_implementation = "eager"
        if self._requested_mask_token_id is not None:
            self._config.mask_token_id = self._requested_mask_token_id

    def _validate_dopd_args(self) -> None:
        if self.block_length <= 0:
            raise ValueError("block_length must be positive.")
        if self.denoising_steps_per_block <= 0:
            raise ValueError("denoising_steps_per_block must be positive.")
        if self.diffusion_temperature < 0:
            raise ValueError("diffusion_temperature must be non-negative.")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")

    def _validate_mask_token(self) -> None:
        embeddings = self.model.get_input_embeddings()
        vocab_size = int(embeddings.weight.shape[0])
        if not 0 <= self.mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id={self.mask_token_id} is outside the loaded "
                f"embedding vocabulary ({vocab_size})."
            )
        if self.mask_token_id in {
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }:
            raise ValueError("The d-OPD mask token must differ from EOS and PAD.")

    def _attention_mask_mapping(
        self, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Route an explicit additive mask to every Qwen3 attention layer."""
        layer_types = getattr(self.config, "layer_types", None) or ["full_attention"]
        return {key: mask for key in set(layer_types) | {"full_attention"}}

    def _prompt_block_attention_mask(
        self, length: int, device: torch.device
    ) -> torch.Tensor:
        """Build JetEngine's start-aligned staircase prompt mask."""
        allowed = torch.zeros((length, length), dtype=torch.bool, device=device)
        for start in range(0, length, self.block_length):
            end = min(start + self.block_length, length)
            allowed[start:end, :end] = True
        mask = torch.full(
            (1, 1, length, length),
            torch.finfo(self.model.dtype).min,
            dtype=self.model.dtype,
            device=device,
        )
        mask[0, 0][allowed] = 0
        return mask

    def _full_visible_block_mask(
        self, query_length: int, key_length: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        mask = torch.zeros(
            (1, 1, query_length, key_length),
            dtype=self.model.dtype,
            device=device,
        )
        return self._attention_mask_mapping(mask)

    def _autocast_context(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        )

    def _new_perf_metrics(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "forward_counts": {"prefill": 0, "denoise": 0, "saving": 0},
            "generation_requests": 0,
            "raw_generated_tokens": 0,
            "decode_seconds": 0.0,
            "requests": [],
        }

    def _increment_forward_count(self, kind: str) -> None:
        metrics = getattr(self, "_perf_metrics", None)
        if metrics is not None:
            metrics["forward_counts"][kind] += 1

    def _metrics_synchronize(self) -> None:
        if not getattr(self, "_metrics_dir", None):
            return
        device_type = self.device.type
        if device_type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elif device_type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()

    def _record_generation_metrics(
        self,
        *,
        generated_tokens: int,
        decode_seconds: float,
        block_steps: list[int],
        block_allocated_steps: list[int],
    ) -> None:
        metrics = getattr(self, "_perf_metrics", None)
        if metrics is None or not getattr(self, "_metrics_dir", None):
            return
        request_index = metrics["generation_requests"]
        metrics["generation_requests"] += 1
        metrics["raw_generated_tokens"] += generated_tokens
        metrics["decode_seconds"] += decode_seconds
        metrics["requests"].append(
            {
                "request_index": request_index,
                "generated_tokens": generated_tokens,
                "decode_seconds": decode_seconds,
                "block_steps": block_steps,
                "block_allocated_steps": block_allocated_steps,
            }
        )

    def _write_perf_metrics(self) -> None:
        metrics_dir = getattr(self, "_metrics_dir", None)
        metrics = getattr(self, "_perf_metrics", None)
        if not metrics_dir or metrics is None:
            return
        try:
            output_dir = Path(metrics_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                **metrics,
                "rank": int(self.rank),
                "world_size": int(self.world_size),
                "settings": {
                    "decoder": "dopd-response-aligned",
                    "block_length": self.block_length,
                    "denoising_steps_per_block": self.denoising_steps_per_block,
                    "diffusion_temperature": self.diffusion_temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                },
            }
            decode_seconds = float(payload["decode_seconds"])
            payload["decode_tok_per_s"] = (
                payload["raw_generated_tokens"] / decode_seconds
                if decode_seconds > 0
                else 0.0
            )
            output_path = output_dir / f"rank_{int(self.rank)}.json"
            temporary_path = output_dir / f"rank_{int(self.rank)}.{os.getpid()}.tmp"
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            temporary_path.replace(output_path)
        except Exception as error:
            eval_logger.warning("Could not write d-OPD metrics: %s", error)

    def _filter_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k > 0:
            top_k = min(self.top_k, logits.shape[-1])
            cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative > self.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            logits = torch.full_like(logits, -torch.inf).scatter(
                -1, sorted_indices, sorted_logits
            )
        return logits

    def _sample(
        self, logits: torch.Tensor, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.float().clone()
        logits[..., self.mask_token_id] = -torch.inf
        raw_probabilities = torch.softmax(logits, dim=-1)
        if temperature == 0 or self.top_k == 1:
            proposed = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(
                self._filter_logits(logits / temperature), dim=-1
            )
            proposed = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1
            ).reshape(probabilities.shape[:-1])
        confidence = raw_probabilities.gather(
            -1, proposed.unsqueeze(-1)
        ).squeeze(-1)
        return confidence, proposed

    @staticmethod
    def _transfer_schedule(token_count: int, steps: int) -> list[int]:
        if token_count <= 0 or steps <= 0 or steps > token_count:
            raise ValueError("d-OPD transfer schedule requires 1 <= steps <= tokens.")
        base, remainder = divmod(token_count, steps)
        return [base + (1 if index < remainder else 0) for index in range(steps)]

    @staticmethod
    def _select_updates(
        confidence: torch.Tensor, mutable: torch.Tensor, quota: int
    ) -> torch.Tensor:
        candidates = torch.nonzero(mutable[0], as_tuple=False).flatten()
        quota = min(int(quota), int(candidates.numel()))
        updates = torch.zeros_like(mutable)
        if quota <= 0:
            return updates
        selected = candidates[
            torch.topk(confidence[0, candidates], k=quota, largest=True).indices
        ]
        updates[0, selected] = True
        return updates

    @torch.inference_mode()
    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        if prompt_ids.shape[0] != 1:
            raise ValueError("_generate_one expects exactly one prompt.")
        if max_new_tokens <= 0:
            return prompt_ids

        model = self.model
        prompt_length = prompt_ids.shape[1]
        prompt_positions = torch.arange(prompt_length, device=prompt_ids.device)
        prompt_mask = self._prompt_block_attention_mask(
            prompt_length, prompt_ids.device
        )
        with self._autocast_context():
            self._increment_forward_count("prefill")
            prefill = model(
                input_ids=prompt_ids,
                attention_mask=self._attention_mask_mapping(prompt_mask),
                position_ids=prompt_positions.unsqueeze(0),
                cache_position=prompt_positions,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = prefill.past_key_values
        output_ids = prompt_ids.clone()
        generated = 0
        block_steps: list[int] = []
        block_allocated_steps: list[int] = []
        generated_output = None

        self._metrics_synchronize()
        decode_start = time.perf_counter()
        num_blocks = (max_new_tokens + self.block_length - 1) // self.block_length
        for block_index in range(num_blocks):
            current_length = min(self.block_length, max_new_tokens - generated)
            block = torch.full(
                (1, current_length),
                self.mask_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            mutable = block.eq(self.mask_token_id)
            cache_length = past_key_values.get_seq_length()
            position_ids = torch.arange(
                cache_length,
                cache_length + current_length,
                device=prompt_ids.device,
            ).unsqueeze(0)
            cache_position = position_ids[0]
            allocated_steps = min(self.denoising_steps_per_block, current_length)
            schedule = self._transfer_schedule(current_length, allocated_steps)
            actual_steps = 0

            for quota in schedule:
                if not mutable.any():
                    break
                with self._autocast_context():
                    self._increment_forward_count("denoise")
                    denoise = model(
                        input_ids=block,
                        attention_mask=self._full_visible_block_mask(
                            current_length,
                            cache_length + current_length,
                            prompt_ids.device,
                        ),
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = denoise.past_key_values
                _crop_cache(past_key_values, cache_length)
                confidence, proposed = self._sample(denoise.logits, temperature)
                confidence = confidence.masked_fill(~mutable, -torch.inf)
                updates = self._select_updates(confidence, mutable, quota)
                block[updates] = proposed[updates]
                mutable &= ~updates
                actual_steps += 1

            block_steps.append(actual_steps)
            block_allocated_steps.append(allocated_steps)
            if mutable.any():
                raise RuntimeError(
                    f"d-OPD block {block_index} still contains mask tokens after "
                    f"{allocated_steps} steps."
                )

            output_ids = torch.cat([output_ids, block], dim=1)
            generated += current_length
            eos_end = None
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                eos_positions = torch.nonzero(
                    output_ids[0, prompt_length:].eq(eos_token_id), as_tuple=False
                ).flatten()
                if eos_positions.numel() > 0:
                    eos_end = prompt_length + int(eos_positions[0].item()) + 1
                    generated_output = output_ids[:, :eos_end]

            # JetEngine's inference state machine schedules every completed
            # block once in SAVING, including the final or EOS block.  The
            # completed block remains full-visible while its KV is materialized.
            with self._autocast_context():
                self._increment_forward_count("saving")
                saving = model(
                    input_ids=block,
                    attention_mask=self._full_visible_block_mask(
                        current_length,
                        cache_length + current_length,
                        prompt_ids.device,
                    ),
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = saving.past_key_values
            if eos_end is not None:
                break

        if generated_output is None:
            generated_output = output_ids
        self._metrics_synchronize()
        decode_seconds = time.perf_counter() - decode_start
        self._record_generation_metrics(
            generated_tokens=generated_output.shape[1] - prompt_length,
            decode_seconds=decode_seconds,
            block_steps=block_steps,
            block_allocated_steps=block_allocated_steps,
        )
        return generated_output

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        del stop
        max_new_tokens = max_length - context.shape[1]
        temperature = float(
            generation_kwargs.get("temperature", self.diffusion_temperature)
        )
        if generation_kwargs.get("do_sample") is False:
            temperature = 0.0

        torch.manual_seed(self.seed + self.rank * 1_000_003 + self._generation_calls)
        self._generation_calls += 1
        rows = []
        for row_index in range(context.shape[0]):
            if attention_mask is None:
                prompt = context[row_index : row_index + 1]
            else:
                prompt = context[row_index : row_index + 1][
                    :, attention_mask[row_index].bool()
                ]
            generated = self._generate_one(prompt, max_new_tokens, temperature)
            continuation = generated[:, prompt.shape[1] :]
            rows.append(
                torch.cat([context[row_index : row_index + 1], continuation], dim=1)
            )

        output_length = max(row.shape[1] for row in rows)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        padded = torch.full(
            (len(rows), output_length),
            pad_token_id,
            dtype=context.dtype,
            device=context.device,
        )
        for row_index, row in enumerate(rows):
            padded[row_index, : row.shape[1]] = row[0]
        return padded

