"""Hugging Face backend for masked block-diffusion language models.

This backend loads an ordinary Hugging Face causal-LM checkpoint, but uses
bidirectional attention while denoising masked tokens.  It is intended for
hybrid AR + diffusion checkpoints whose exported parameter names are
compatible with ``AutoModelForCausalLM`` (for example the Qwen3 diffusion
model in MindSpeed-LLM).
"""

from __future__ import annotations

import atexit
import copy
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import transformers
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


def _as_optional_int(value: int | str | None) -> int | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


@register_model("hf-diffusion")
class HFDiffusionLM(HFLM):
    """Evaluate a Hugging Face-compatible masked diffusion language model.

    Generation uses causal KV-cache updates between blocks and bidirectional
    masked-token denoising inside each block.  Multiple-choice scoring uses the
    masked Monte Carlo conditional-likelihood estimator used by LLaDA.  A
    single-token continuation is evaluated exactly with one masked forward and
    all choices sharing a context reuse that forward.
    """

    def __init__(
        self,
        *args,
        mask_token_id: int | str | None = None,
        block_length: int | str = 32,
        diffusion_steps: int | str | None = None,
        diffusion_temperature: float | str = 0.0,
        threshold: float | str = 0.0,
        top_k: int | str = 0,
        top_p: float | str = 1.0,
        mc_num: int | str = 128,
        mc_batch_size: int | str = 16,
        check_greedy: bool = False,
        seed: int | str = 42,
        **kwargs,
    ) -> None:
        if parse_version(transformers.__version__) < parse_version("5.2.0"):
            raise RuntimeError(
                "hf-diffusion requires transformers>=5.2.0 for per-layer "
                f"attention-mask mappings; found {transformers.__version__}."
            )
        self._requested_mask_token_id = _as_optional_int(mask_token_id)
        self.block_length = int(block_length)
        self.diffusion_steps = _as_optional_int(diffusion_steps)
        self.diffusion_temperature = float(diffusion_temperature)
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.mc_num = int(mc_num)
        self.mc_batch_size = int(mc_batch_size)
        self.check_greedy = bool(check_greedy)
        self.seed = int(seed)
        self._generation_calls = 0
        self._perf_metrics = self._new_perf_metrics()
        self._metrics_dir = os.environ.get("HF_DIFFUSION_METRICS_DIR")

        self._validate_diffusion_args()
        # HFLM loads its inspection config and model config separately.  This
        # kwarg must reach AutoModelForCausalLM.from_pretrained directly;
        # changing only self._config would leave the real model on SDPA.
        kwargs["attn_implementation"] = "eager"
        super().__init__(*args, **kwargs)

        if self.backend != "causal":
            raise ValueError(
                "hf-diffusion only supports decoder-only causal-LM checkpoints."
            )
        actual_attention = getattr(self.model.config, "_attn_implementation", None)
        if actual_attention != "eager":
            raise RuntimeError(
                "hf-diffusion requires eager attention, but the loaded model uses "
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
                "No diffusion mask token is configured. Add mask_token_id to "
                "config.json or pass mask_token_id=... in --model_args."
            )
        self.mask_token_id = int(self.mask_token_id)
        self._validate_mask_token()

        if self._metrics_dir:
            atexit.register(self._write_perf_metrics)

        eval_logger.info(
            "Initialized hf-diffusion: mask_token_id=%d, block_length=%d, "
            "diffusion_steps=%s, threshold=%.4f, mc_num=%d",
            self.mask_token_id,
            self.block_length,
            self.diffusion_steps,
            self.threshold,
            self.mc_num,
        )

    def _get_config(self, *args, **kwargs) -> None:
        """Load the HF config and force the 4D-mask-compatible eager backend."""
        super()._get_config(*args, **kwargs)
        self._config._attn_implementation = "eager"
        self._config.attn_implementation = "eager"
        if self._requested_mask_token_id is not None:
            self._config.mask_token_id = self._requested_mask_token_id

    def _validate_diffusion_args(self) -> None:
        if self.block_length <= 0:
            raise ValueError("block_length must be positive.")
        if self.diffusion_steps is not None and self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive when provided.")
        if self.diffusion_temperature < 0:
            raise ValueError("diffusion_temperature must be non-negative.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if self.mc_num <= 0 or self.mc_batch_size <= 0:
            raise ValueError("mc_num and mc_batch_size must be positive.")

    def _validate_mask_token(self) -> None:
        embeddings = self.model.get_input_embeddings()
        vocab_size = int(embeddings.weight.shape[0])
        if not 0 <= self.mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id={self.mask_token_id} is outside the loaded embedding "
                f"vocabulary ({vocab_size}). The exported diffusion checkpoint must "
                "contain the added mask-token embedding."
            )
        if self.mask_token_id in {
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }:
            raise ValueError("The diffusion mask token must differ from EOS and PAD.")

    def _bidirectional_mask(self) -> dict[str, None]:
        """Return a Qwen3 mask mapping that bypasses causal-mask creation.

        Transformers 5.x accepts a mapping of per-layer attention masks.  With
        eager attention, ``None`` means unrestricted attention.  Keeping this
        as a mapping is important: a bare ``None`` would make a decoder build a
        causal mask again.
        """
        layer_types = getattr(self.config, "layer_types", None) or ["full_attention"]
        return dict.fromkeys(set(layer_types) | {"full_attention"})

    def _autocast_context(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        )

    def _new_perf_metrics(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "forward_counts": {
                "prefill": 0,
                "denoise": 0,
                "causal": 0,
                "scoring": 0,
            },
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
        """Wait for queued device work only when performance metrics are enabled."""
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
        block_forced_tokens: list[int],
        block_stalled_steps: list[int],
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
                "block_forced_low_confidence_tokens": block_forced_tokens,
                "block_stalled_steps": block_stalled_steps,
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
            rank = int(self.rank)
            payload = {
                **metrics,
                "rank": rank,
                "world_size": int(self.world_size),
                "settings": {
                    "block_length": self.block_length,
                    "diffusion_steps": self.diffusion_steps,
                    "threshold": self.threshold,
                    "diffusion_temperature": self.diffusion_temperature,
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
            eval_logger.warning("Could not write diffusion metrics: %s", error)

    def _diffusion_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), self._autocast_context():
            self._increment_forward_count("scoring")
            return self.model(
                input_ids=input_ids,
                attention_mask=self._bidirectional_mask(),
                use_cache=False,
                return_dict=True,
            ).logits

    @staticmethod
    def _allocate_steps(total_steps: int, num_blocks: int) -> list[int]:
        if total_steps < num_blocks:
            raise ValueError(
                f"diffusion_steps ({total_steps}) must be at least the number "
                f"of generation blocks ({num_blocks})."
            )
        base, remainder = divmod(total_steps, num_blocks)
        return [base + (index < remainder) for index in range(num_blocks)]

    def _filter_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k > 0:
            k = min(self.top_k, logits.shape[-1])
            cutoff = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
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
        # The mask token is a transient diffusion state, never a valid decoded
        # token.  Committing it would leave an apparently completed block with
        # unresolved masks.
        logits[..., self.mask_token_id] = -torch.inf
        raw_probabilities = torch.softmax(logits, dim=-1)
        if temperature == 0 or self.top_k == 1:
            sampled = logits.argmax(dim=-1)
            confidence = raw_probabilities.gather(
                -1, sampled.unsqueeze(-1)
            ).squeeze(-1)
            return confidence, sampled

        probabilities = torch.softmax(
            self._filter_logits(logits / temperature), dim=-1
        )
        sampled = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1
        ).reshape(probabilities.shape[:-1])
        confidence = raw_probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return confidence, sampled

    def _select_updates(
        self,
        confidence: torch.Tensor,
        mutable: torch.Tensor,
        initial_count: int,
        step_index: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, int, int]:
        candidates = torch.nonzero(mutable[0], as_tuple=False).flatten()
        if candidates.numel() == 0:
            return torch.zeros_like(mutable), 0, 0

        candidate_confidence = confidence[0, candidates]
        if self.threshold > 0:
            above_threshold = candidate_confidence >= self.threshold
            selected = candidates[above_threshold]

            # Thresholding is allowed to reveal more tokens than the scheduled
            # decoder, but never fewer than its cumulative quota.  Without this
            # floor, a high threshold can leave the block unchanged for several
            # identical deterministic forwards and then force every remaining
            # mask on the final step, which severely degrades generation quality.
            already_committed = initial_count - int(candidates.numel())
            target_committed = math.ceil(
                initial_count * (step_index + 1) / num_steps
            )
            minimum = min(
                max(target_committed - already_committed, 1),
                int(candidates.numel()),
            )
            if selected.numel() < minimum:
                needed = minimum - int(selected.numel())
                remaining_confidence = candidate_confidence.masked_fill(
                    above_threshold, -torch.inf
                )
                supplemental = candidates[
                    torch.topk(remaining_confidence, k=needed, largest=True).indices
                ]
                selected = torch.cat([selected, supplemental])

            updates = torch.zeros_like(mutable)
            updates[0, selected] = True
            return updates, int(selected.numel()), 0

        already_committed = initial_count - int(candidates.numel())
        target_committed = math.ceil(initial_count * (step_index + 1) / num_steps)
        required = min(
            max(target_committed - already_committed, 1), int(candidates.numel())
        )
        selected = candidates[
            torch.topk(candidate_confidence, k=required, largest=True).indices
        ]
        updates = torch.zeros_like(mutable)
        updates[0, selected] = True
        return updates, int(selected.numel()), 0

    @torch.inference_mode()
    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Generate one unpadded sequence with hybrid block diffusion.

        This deliberately preserves the evaluator used on 2026-10-12:
        prompt prefill and completed-block KV updates are causal, while only
        the currently masked block is decoded with bidirectional attention.
        """
        if prompt_ids.shape[0] != 1:
            raise ValueError("_generate_one expects exactly one prompt.")
        if max_new_tokens <= 0:
            return prompt_ids

        model = self.model
        prompt_length = prompt_ids.shape[1]
        num_blocks = math.ceil(max_new_tokens / self.block_length)
        total_steps = self.diffusion_steps or max_new_tokens
        steps_per_block = self._allocate_steps(total_steps, num_blocks)

        with self._autocast_context():
            self._increment_forward_count("prefill")
            prefill = model(
                input_ids=prompt_ids,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = prefill.past_key_values
        _, next_token = self._sample(prefill.logits[:, -1, :], temperature)
        next_token = next_token[:, None]
        output_ids = prompt_ids.clone()
        generated = 0
        block_steps = []
        block_allocated_steps = []
        block_forced_tokens = []
        block_stalled_steps = []
        generated_output = None

        # Decode throughput excludes prefill and includes all block denoising,
        # sampling, cache-update forwards and the associated host work.
        self._metrics_synchronize()
        decode_start = time.perf_counter()

        for block_index in range(num_blocks):
            current_length = min(self.block_length, max_new_tokens - generated)
            block = torch.full(
                (1, current_length),
                self.mask_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            block[:, 0] = next_token[:, 0]
            mutable = block.eq(self.mask_token_id)
            initial_count = int(mutable.sum().item())
            cache_length = past_key_values.get_seq_length()
            position_ids = torch.arange(
                cache_length,
                cache_length + current_length,
                device=prompt_ids.device,
            ).unsqueeze(0)
            cache_position = position_ids[0]
            actual_steps = 0
            forced_tokens = 0
            stalled_steps = 0

            for step_index in range(steps_per_block[block_index]):
                if not mutable.any():
                    break
                denoise_cache = copy.deepcopy(past_key_values)
                with self._autocast_context():
                    self._increment_forward_count("denoise")
                    denoise = model(
                        input_ids=block,
                        attention_mask=self._bidirectional_mask(),
                        past_key_values=denoise_cache,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        use_cache=True,
                        return_dict=True,
                    )
                confidence, proposed = self._sample(denoise.logits, temperature)
                confidence = confidence.masked_fill(~mutable, -torch.inf)
                updates, accepted_count, step_forced_tokens = self._select_updates(
                    confidence,
                    mutable,
                    initial_count,
                    step_index,
                    steps_per_block[block_index],
                )
                block[updates] = proposed[updates]
                mutable &= ~updates
                actual_steps += 1
                forced_tokens += step_forced_tokens
                stalled_steps += accepted_count == 0

            block_steps.append(actual_steps)
            block_allocated_steps.append(steps_per_block[block_index])
            block_forced_tokens.append(forced_tokens)
            block_stalled_steps.append(stalled_steps)

            if mutable.any():
                raise RuntimeError(
                    f"Diffusion block {block_index} still contains mask tokens after "
                    f"{steps_per_block[block_index]} steps."
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
            if eos_end is not None:
                generated_output = output_ids[:, :eos_end]
                break

            # Preserve the 2026-10-12 hybrid decoder: a completed block is
            # inserted into the persistent cache with ordinary causal attention.
            with self._autocast_context():
                self._increment_forward_count("causal")
                causal = model(
                    input_ids=block,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = causal.past_key_values
            if block_index + 1 < num_blocks:
                _, next_token = self._sample(causal.logits[:, -1, :], temperature)
                next_token = next_token[:, None]

        if generated_output is None:
            generated_output = output_ids
        self._metrics_synchronize()
        decode_seconds = time.perf_counter() - decode_start
        generated_tokens = generated_output.shape[1] - prompt_length
        self._record_generation_metrics(
            generated_tokens=generated_tokens,
            decode_seconds=decode_seconds,
            block_steps=block_steps,
            block_allocated_steps=block_allocated_steps,
            block_forced_tokens=block_forced_tokens,
            block_stalled_steps=block_stalled_steps,
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
        del stop  # String stops are applied by HFLM.generate_until after decoding.
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

    def _score_one_token_groups(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        result: list[tuple[float, bool] | None],
        disable_tqdm: bool,
    ) -> set[int]:
        groups: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
        handled = set()
        for index, (_, context, continuation) in enumerate(requests):
            if len(continuation) == 1:
                truncated_context = context[-(self.max_length - 1) :]
                groups[tuple(truncated_context)].append((index, continuation[0]))
                handled.add(index)

        for context, choices in tqdm(
            groups.items(),
            disable=disable_tqdm or self.rank != 0,
            desc="Scoring one-token diffusion choices",
        ):
            input_ids = torch.tensor(
                [list(context) + [self.mask_token_id]],
                dtype=torch.long,
                device=self.device,
            )
            logits = self._diffusion_logits(input_ids)[0, -1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            greedy_token = int(logits.argmax().item())
            for index, token_id in choices:
                result[index] = (
                    float(log_probs[token_id].item()),
                    self.check_greedy and token_id == greedy_token,
                )
        return handled

    def _mc_loglikelihood(self, context: list[int], continuation: list[int]) -> float:
        if len(continuation) > self.max_length:
            raise ValueError(
                f"Continuation length {len(continuation)} exceeds "
                f"max_length={self.max_length}."
            )
        context = context[-(self.max_length - len(continuation)) :]
        clean = torch.tensor(
            context + continuation, dtype=torch.long, device=self.device
        )
        context_length = len(context)
        target_length = len(continuation)
        target = clean[context_length:]
        accumulated_loss = 0.0

        for offset in range(0, self.mc_num, self.mc_batch_size):
            batch_size = min(self.mc_batch_size, self.mc_num - offset)
            batch = clean.unsqueeze(0).repeat(batch_size, 1)
            masked_counts = torch.randint(
                1, target_length + 1, (batch_size,), device=self.device
            )
            random_order = torch.rand(
                (batch_size, target_length), device=self.device
            ).argsort(dim=-1)
            target_mask = random_order < masked_counts[:, None]
            batch[:, context_length:] = torch.where(
                target_mask,
                torch.full_like(batch[:, context_length:], self.mask_token_id),
                batch[:, context_length:],
            )
            logits = self._diffusion_logits(batch)[:, context_length:, :].float()
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.repeat(batch_size).reshape(-1),
                reduction="none",
            ).reshape(batch_size, target_length)
            mask_probability = masked_counts.float() / target_length
            accumulated_loss += float(
                ((losses * target_mask) / mask_probability[:, None]).sum().item()
            )

        return -accumulated_loss / self.mc_num

    def _confidence_greedy(self, context: list[int], continuation: list[int]) -> bool:
        if not self.check_greedy:
            return False
        context = context[-(self.max_length - len(continuation)) :]
        sequence = torch.tensor(
            [context + [self.mask_token_id] * len(continuation)],
            dtype=torch.long,
            device=self.device,
        )
        mutable = sequence.eq(self.mask_token_id)
        for _ in continuation:
            logits = self._diffusion_logits(sequence)
            probabilities = torch.softmax(logits.float(), dim=-1)
            confidence, proposed = probabilities.max(dim=-1)
            confidence = confidence.masked_fill(~mutable, -torch.inf)
            position = int(confidence[0].argmax().item())
            sequence[0, position] = proposed[0, position]
            mutable[0, position] = False
        predicted = sequence[0, -len(continuation) :].tolist()
        return predicted == continuation

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        del override_bs
        result: list[tuple[float, bool] | None] = [None] * len(requests)
        handled = self._score_one_token_groups(
            requests, result, disable_tqdm=disable_tqdm
        )

        for index, (_, context, continuation) in enumerate(
            tqdm(
                requests,
                disable=disable_tqdm or self.rank != 0,
                desc="Scoring multi-token diffusion continuations",
            )
        ):
            if index in handled:
                continue
            if not continuation:
                raise ValueError("Cannot score an empty continuation.")
            result[index] = (
                self._mc_loglikelihood(context, continuation),
                self._confidence_greedy(context, continuation),
            )

        finalized = []
        for request, answer in zip(requests, result, strict=True):
            assert answer is not None
            request_str = request[0]
            if request_str is not None:
                self.cache_hook.add_partial("loglikelihood", request_str, answer)
            finalized.append(answer)
        return finalized

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        del requests, disable_tqdm
        raise NotImplementedError(
            "Rolling perplexity is not implemented for hf-diffusion. Conditional "
            "loglikelihood and generate_until tasks are supported."
        )
