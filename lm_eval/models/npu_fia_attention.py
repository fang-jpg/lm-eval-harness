"""Phase-aware Ascend attention backend for hybrid diffusion decoding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


NPU_HYBRID_FIA = "npu_hybrid_fia"
_FULL_ATTENTION_WINDOW = 2_147_483_647
_COMPRESSED_CAUSAL_MASK_SIZE = 2048
_EAGER_ATTENTION_FORWARD: Callable[..., Any] | None = None
_SDPA_ATTENTION_FORWARD: Callable[..., Any] | None = None
_COMPRESSED_CAUSAL_MASKS: dict[str, torch.Tensor] = {}
_REGISTERED = False


def _qwen3_eager_attention_forward() -> Callable[..., Any]:
    global _EAGER_ATTENTION_FORWARD
    if _EAGER_ATTENTION_FORWARD is None:
        from transformers.models.qwen3.modeling_qwen3 import (
            eager_attention_forward,
        )

        _EAGER_ATTENTION_FORWARD = eager_attention_forward
    return _EAGER_ATTENTION_FORWARD


def _sdpa_attention_forward() -> Callable[..., Any]:
    global _SDPA_ATTENTION_FORWARD
    if _SDPA_ATTENTION_FORWARD is None:
        from transformers.integrations.sdpa_attention import (
            sdpa_attention_forward,
        )

        _SDPA_ATTENTION_FORWARD = sdpa_attention_forward
    return _SDPA_ATTENTION_FORWARD


def _load_fia_op() -> Callable[..., Any]:
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError(
            "denoise_attention_backend=npu_fia requires torch_npu."
        ) from error
    operation = getattr(torch_npu, "npu_fused_infer_attention_score", None)
    if not callable(operation):
        raise RuntimeError(
            "The installed torch_npu does not provide "
            "npu_fused_infer_attention_score."
        )
    return operation


def _compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Return the shared 2048x2048 mask required by FIA sparse mode 3."""
    device_key = str(device)
    mask = _COMPRESSED_CAUSAL_MASKS.get(device_key)
    if mask is None:
        mask = torch.triu(
            torch.ones(
                (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        ).contiguous()
        _COMPRESSED_CAUSAL_MASKS[device_key] = mask
    return mask


def validate_npu_fia_runtime(
    device: torch.device | None = None,
    *,
    prepare_causal_mask: bool = False,
) -> None:
    """Fail early if FIA is absent and optionally prewarm its causal mask."""
    _load_fia_op()
    if prepare_causal_mask:
        if device is None or device.type != "npu":
            raise RuntimeError("Preparing the FIA causal mask requires an NPU device.")
        _compressed_causal_mask(device)
        torch.npu.synchronize(device)


def _validate_fia_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FIA expects four-dimensional BNSD query/key/value tensors.")
    if key.shape != value.shape:
        raise ValueError(
            f"FIA key/value shapes must match, got {key.shape} and {value.shape}."
        )
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("FIA query and key must share batch and head dimensions.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            "FIA GQA requires query heads to be divisible by key/value heads."
        )


def _npu_fia_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    attention_mask: torch.Tensor | None,
    sparse_mode: int,
    next_tokens: int,
    fia_op: Callable[..., Any] | None = None,
) -> torch.Tensor:
    """Run FIA with BNSD inputs and convert its output to Qwen3 BSND."""
    _validate_fia_inputs(query, key, value)

    batch_size = int(query.shape[0])
    query_length = int(query.shape[-2])
    kv_length = int(key.shape[-2])
    operation = fia_op or _load_fia_op()
    attention_output, _ = operation(
        query=query.contiguous(),
        key=key.contiguous(),
        value=value.contiguous(),
        atten_mask=(
            attention_mask.contiguous() if attention_mask is not None else None
        ),
        actual_seq_lengths=[query_length] * batch_size,
        actual_seq_lengths_kv=[kv_length] * batch_size,
        num_heads=int(query.shape[1]),
        num_key_value_heads=int(key.shape[1]),
        input_layout="BNSD",
        scale=float(scaling),
        pre_tokens=_FULL_ATTENTION_WINDOW,
        next_tokens=next_tokens,
        sparse_mode=sparse_mode,
    )
    if attention_output.shape != query.shape:
        raise RuntimeError(
            "FIA BNSD output shape does not match query shape: "
            f"expected {query.shape}, found {attention_output.shape}."
        )
    return attention_output.transpose(1, 2).contiguous()


def npu_fia_bidirectional_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    fia_op: Callable[..., Any] | None = None,
) -> torch.Tensor:
    """Run full bidirectional GQA with BNSD inputs and return BSND output."""
    return _npu_fia_attention(
        query,
        key,
        value,
        scaling=scaling,
        attention_mask=None,
        sparse_mode=0,
        next_tokens=_FULL_ATTENTION_WINDOW,
        fia_op=fia_op,
    )


def npu_fia_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    compressed_mask: torch.Tensor | None = None,
    fia_op: Callable[..., Any] | None = None,
) -> torch.Tensor:
    """Run right-down causal FIA, or its mask-free Q=1 incremental path."""
    _validate_fia_inputs(query, key, value)
    query_length = int(query.shape[-2])
    if query_length == 1:
        return _npu_fia_attention(
            query,
            key,
            value,
            scaling=scaling,
            attention_mask=None,
            sparse_mode=0,
            next_tokens=_FULL_ATTENTION_WINDOW,
            fia_op=fia_op,
        )

    if int(key.shape[-2]) < query_length:
        raise ValueError("Causal FIA requires KV length to be at least query length.")
    mask = compressed_mask
    if mask is None:
        mask = _compressed_causal_mask(query.device)
    expected_shape = (_COMPRESSED_CAUSAL_MASK_SIZE,) * 2
    if mask.shape != expected_shape or mask.dtype != torch.bool:
        raise ValueError(
            "FIA sparse_mode=3 requires a bool 2048x2048 compressed mask."
        )
    return _npu_fia_attention(
        query,
        key,
        value,
        scaling=scaling,
        attention_mask=mask,
        sparse_mode=3,
        next_tokens=0,
        fia_op=fia_op,
    )


def sdpa_causal_prefill_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    dropout: float = 0.0,
    sdpa_op: Callable[..., Any] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run equal-length causal prefill through SDPA/FlashAttentionScore."""
    _validate_fia_inputs(query, key, value)
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("SDPA causal prefill requires equal query and KV lengths.")
    operation = sdpa_op or _sdpa_attention_forward()
    return operation(
        module,
        query,
        key,
        value,
        None,
        scaling=scaling,
        dropout=dropout,
        is_causal=True,
        **kwargs,
    )


def npu_hybrid_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    diffusion_bidirectional: bool = False,
    diffusion_attention_phase: str | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Route explicit NPU inference phases while preserving eager fallback."""
    phase = diffusion_attention_phase
    if phase is None and diffusion_bidirectional:
        phase = "denoise"
    if phase is None or query.device.type != "npu":
        eager_attention = _qwen3_eager_attention_forward()
        return eager_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    if module.training or dropout != 0.0:
        raise RuntimeError("NPU fused attention requires eval mode and dropout=0.")

    if phase == "prefill":
        return sdpa_causal_prefill_attention(
            module,
            query,
            key,
            value,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    if phase == "denoise":
        if attention_mask is not None:
            raise RuntimeError(
                "NPU FIA denoising expects an unrestricted per-layer mask."
            )
        attention_output = npu_fia_bidirectional_attention(
            query,
            key,
            value,
            scaling=scaling,
        )
        return attention_output, None

    if phase in {"causal_refresh", "ar_decode"}:
        if phase == "ar_decode" and query.shape[-2] != 1:
            raise RuntimeError("The ar_decode attention phase requires Q length 1.")
        attention_output = npu_fia_causal_attention(
            query,
            key,
            value,
            scaling=scaling,
        )
        return attention_output, None

    raise ValueError(
        "Unknown diffusion attention phase; expected prefill, denoise, "
        f"causal_refresh, or ar_decode, got {phase!r}."
    )


def register_npu_hybrid_attention() -> str:
    """Register phase-aware attention and eager-compatible mask creation."""
    global _REGISTERED
    if _REGISTERED:
        return NPU_HYBRID_FIA

    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask

    _qwen3_eager_attention_forward()
    _sdpa_attention_forward()
    AttentionInterface.register(NPU_HYBRID_FIA, npu_hybrid_attention_forward)
    AttentionMaskInterface.register(NPU_HYBRID_FIA, eager_mask)
    _REGISTERED = True
    return NPU_HYBRID_FIA
