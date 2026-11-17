"""Ascend FIA attention backend used only for diffusion denoising."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


NPU_HYBRID_FIA = "npu_hybrid_fia"
_FULL_ATTENTION_WINDOW = 2_147_483_647
_EAGER_ATTENTION_FORWARD: Callable[..., Any] | None = None
_REGISTERED = False


def _qwen3_eager_attention_forward() -> Callable[..., Any]:
    global _EAGER_ATTENTION_FORWARD
    if _EAGER_ATTENTION_FORWARD is None:
        from transformers.models.qwen3.modeling_qwen3 import (
            eager_attention_forward,
        )

        _EAGER_ATTENTION_FORWARD = eager_attention_forward
    return _EAGER_ATTENTION_FORWARD


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


def validate_npu_fia_runtime() -> None:
    """Fail before evaluation if the installed torch_npu lacks FIA v1."""
    _load_fia_op()


def npu_fia_bidirectional_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    fia_op: Callable[..., Any] | None = None,
) -> torch.Tensor:
    """Run full bidirectional GQA with BNSD inputs and return BSND output."""
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

    batch_size = int(query.shape[0])
    query_length = int(query.shape[-2])
    kv_length = int(key.shape[-2])
    operation = fia_op or _load_fia_op()
    attention_output, _ = operation(
        query=query.contiguous(),
        key=key.contiguous(),
        value=value.contiguous(),
        atten_mask=None,
        actual_seq_lengths=[query_length] * batch_size,
        actual_seq_lengths_kv=[kv_length] * batch_size,
        num_heads=int(query.shape[1]),
        num_key_value_heads=int(key.shape[1]),
        input_layout="BNSD",
        scale=float(scaling),
        pre_tokens=_FULL_ATTENTION_WINDOW,
        next_tokens=_FULL_ATTENTION_WINDOW,
        sparse_mode=0,
    )
    if attention_output.shape != query.shape:
        raise RuntimeError(
            "FIA BNSD output shape does not match query shape: "
            f"expected {query.shape}, found {attention_output.shape}."
        )
    return attention_output.transpose(1, 2).contiguous()


def npu_hybrid_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    diffusion_bidirectional: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use FIA for NPU denoise; preserve Qwen3 eager attention elsewhere."""
    if not diffusion_bidirectional or query.device.type != "npu":
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
        raise RuntimeError("NPU FIA denoising requires evaluation mode and dropout=0.")
    if attention_mask is not None:
        raise RuntimeError(
            "NPU FIA denoising expects an unrestricted per-layer attention mask."
        )

    attention_output = npu_fia_bidirectional_attention(
        query,
        key,
        value,
        scaling=scaling,
    )
    return attention_output, None


def register_npu_hybrid_attention() -> str:
    """Register the hybrid attention and its eager-compatible causal mask."""
    global _REGISTERED
    if _REGISTERED:
        return NPU_HYBRID_FIA

    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask

    _qwen3_eager_attention_forward()
    AttentionInterface.register(NPU_HYBRID_FIA, npu_hybrid_attention_forward)
    AttentionMaskInterface.register(NPU_HYBRID_FIA, eager_mask)
    _REGISTERED = True
    return NPU_HYBRID_FIA
