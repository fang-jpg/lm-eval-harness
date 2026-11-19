from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import lm_eval.models.npu_fia_attention as npu_fia


def test_npu_fia_bidirectional_attention_uses_gqa_and_returns_bsnd():
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)
    calls = []

    def fake_fia(**kwargs):
        calls.append(kwargs)
        return kwargs["query"] + 1, None

    output = npu_fia.npu_fia_bidirectional_attention(
        query,
        key,
        value,
        scaling=8**-0.5,
        fia_op=fake_fia,
    )

    assert output.shape == (2, 3, 4, 8)
    assert torch.equal(output, (query + 1).transpose(1, 2))
    assert len(calls) == 1
    call = calls[0]
    assert call["atten_mask"] is None
    assert call["actual_seq_lengths"] == [3, 3]
    assert call["actual_seq_lengths_kv"] == [5, 5]
    assert call["num_heads"] == 4
    assert call["num_key_value_heads"] == 2
    assert call["input_layout"] == "BNSD"
    assert call["sparse_mode"] == 0
    assert call["pre_tokens"] == 2_147_483_647
    assert call["next_tokens"] == 2_147_483_647


def test_npu_fia_causal_attention_uses_right_down_compressed_mask():
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 8)
    compressed_mask = torch.triu(
        torch.ones((2048, 2048), dtype=torch.bool), diagonal=1
    ).contiguous()
    calls = []

    def fake_fia(**kwargs):
        calls.append(kwargs)
        return kwargs["query"], None

    output = npu_fia.npu_fia_causal_attention(
        query,
        key,
        value,
        scaling=8**-0.5,
        compressed_mask=compressed_mask,
        fia_op=fake_fia,
    )

    assert output.shape == (1, 3, 4, 8)
    call = calls[0]
    assert call["sparse_mode"] == 3
    assert call["next_tokens"] == 0
    assert call["atten_mask"].data_ptr() == compressed_mask.data_ptr()
    assert call["actual_seq_lengths"] == [3]
    assert call["actual_seq_lengths_kv"] == [7]


def test_npu_fia_q1_causal_attention_uses_mask_free_incremental_path():
    query = torch.randn(1, 4, 1, 8)
    key = torch.randn(1, 2, 7, 8)
    calls = []

    def fake_fia(**kwargs):
        calls.append(kwargs)
        return kwargs["query"], None

    npu_fia.npu_fia_causal_attention(
        query,
        key,
        key,
        scaling=8**-0.5,
        fia_op=fake_fia,
    )

    call = calls[0]
    assert call["atten_mask"] is None
    assert call["sparse_mode"] == 0
    assert call["next_tokens"] == 2_147_483_647


def test_sdpa_prefill_uses_equal_length_causal_fast_path():
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    calls = []

    def fake_sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(1, 5, 4, 8), None

    output, weights = npu_fia.sdpa_causal_prefill_attention(
        SimpleNamespace(),
        query,
        key,
        key,
        scaling=8**-0.5,
        sdpa_op=fake_sdpa,
    )

    assert output.shape == (1, 5, 4, 8)
    assert weights is None
    args, kwargs = calls[0]
    assert args[4] is None
    assert kwargs["is_causal"] is True
    assert kwargs["scaling"] == 8**-0.5


def test_sdpa_prefill_matches_causal_eager_reference():
    torch.manual_seed(19)
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    scaling = 8**-0.5

    actual, _ = npu_fia.sdpa_causal_prefill_attention(
        SimpleNamespace(num_key_value_groups=2, is_causal=True),
        query,
        key,
        value,
        scaling=scaling,
    )
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(2, 3)) * scaling
    future_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(
        scores.masked_fill(future_mask[None, None], -torch.inf), dim=-1
    )
    expected = torch.matmul(weights, expanded_value).transpose(1, 2).contiguous()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_npu_hybrid_attention_keeps_eager_fallback_on_cpu(monkeypatch):
    expected = torch.randn(1, 3, 4, 8)
    calls = []

    def fake_eager(*args, **kwargs):
        calls.append((args, kwargs))
        return expected, "weights"

    monkeypatch.setattr(npu_fia, "_EAGER_ATTENTION_FORWARD", fake_eager)
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    output, weights = npu_fia.npu_hybrid_attention_forward(
        SimpleNamespace(training=False),
        query,
        key,
        value,
        attention_mask=None,
        scaling=8**-0.5,
        diffusion_bidirectional=True,
    )

    assert output is expected
    assert weights == "weights"
    assert len(calls) == 1


def test_registered_hybrid_backend_matches_eager_on_cpu():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        eager_logits = model(input_ids=input_ids, use_cache=False).logits
        backend = npu_fia.register_npu_hybrid_attention()
        model.set_attn_implementation(backend)
        hybrid_logits = model(
            input_ids=input_ids,
            use_cache=False,
            diffusion_bidirectional=True,
        ).logits

    assert torch.equal(eager_logits, hybrid_logits)


def _npu_fia_available() -> bool:
    try:
        import torch_npu
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    return bool(
        npu is not None
        and npu.is_available()
        and callable(getattr(torch_npu, "npu_fused_infer_attention_score", None))
    )


@pytest.mark.skipif(not _npu_fia_available(), reason="Ascend FIA is unavailable")
def test_npu_fia_matches_full_bidirectional_eager_reference():
    torch.manual_seed(11)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    query = torch.randn(1, 4, 4, 128, device=device).to(torch.bfloat16)
    key = torch.randn(1, 2, 12, 128, device=device).to(torch.bfloat16)
    value = torch.randn(1, 2, 12, 128, device=device).to(torch.bfloat16)
    scaling = 128**-0.5

    actual = npu_fia.npu_fia_bidirectional_attention(
        query,
        key,
        value,
        scaling=scaling,
    )
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    weights = torch.softmax(
        torch.matmul(query.float(), expanded_key.float().transpose(2, 3))
        * scaling,
        dim=-1,
    ).to(query.dtype)
    expected = torch.matmul(weights, expanded_value).transpose(1, 2).contiguous()
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.skipif(not _npu_fia_available(), reason="Ascend FIA is unavailable")
def test_npu_fia_matches_right_down_causal_eager_reference():
    torch.manual_seed(13)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    query = torch.randn(1, 4, 4, 128, device=device).to(torch.bfloat16)
    key = torch.randn(1, 2, 12, 128, device=device).to(torch.bfloat16)
    value = torch.randn(1, 2, 12, 128, device=device).to(torch.bfloat16)
    scaling = 128**-0.5

    actual = npu_fia.npu_fia_causal_attention(
        query,
        key,
        value,
        scaling=scaling,
    )
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = (
        torch.matmul(query.float(), expanded_key.float().transpose(2, 3))
        * scaling
    )
    query_positions = torch.arange(8, 12, device=device)[:, None]
    key_positions = torch.arange(12, device=device)[None, :]
    future_mask = key_positions > query_positions
    weights = torch.softmax(
        scores.masked_fill(future_mask[None, None], -torch.inf), dim=-1
    ).to(query.dtype)
    expected = torch.matmul(weights, expanded_value).transpose(1, 2).contiguous()
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.skipif(not _npu_fia_available(), reason="Ascend FIA is unavailable")
def test_npu_fia_q1_incremental_matches_eager_reference():
    torch.manual_seed(17)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    query = torch.randn(1, 4, 1, 128, device=device).to(torch.bfloat16)
    key = torch.randn(1, 2, 12, 128, device=device).to(torch.bfloat16)
    value = torch.randn(1, 2, 12, 128, device=device).to(torch.bfloat16)
    scaling = 128**-0.5

    actual = npu_fia.npu_fia_causal_attention(
        query,
        key,
        value,
        scaling=scaling,
    )
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    weights = torch.softmax(
        torch.matmul(query.float(), expanded_key.float().transpose(2, 3))
        * scaling,
        dim=-1,
    ).to(query.dtype)
    expected = torch.matmul(weights, expanded_value).transpose(1, 2).contiguous()
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=3e-2,
        atol=3e-2,
    )
