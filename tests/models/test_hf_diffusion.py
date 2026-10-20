import math
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from lm_eval.api.registry import get_model
from lm_eval.models.hf_diffusion import HFDiffusionLM


class _CacheHook:
    def add_partial(self, *args):
        del args


class _FakeMaskedLM(torch.nn.Module):
    def __init__(self, vocab_size=4, preferred_token=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        del kwargs
        self.calls += 1
        logits = torch.zeros(
            (*input_ids.shape, self.vocab_size), device=input_ids.device
        )
        if self.preferred_token is not None:
            logits[..., self.preferred_token] = 2.0
        return SimpleNamespace(logits=logits)


def _make_lm(model, *, mc_num=8, mc_batch_size=4):
    lm = HFDiffusionLM.__new__(HFDiffusionLM)
    lm._model = model
    lm._device = torch.device("cpu")
    lm._config = SimpleNamespace(layer_types=["full_attention"])
    lm._max_length = 32
    lm.mask_token_id = 3
    lm.mc_num = mc_num
    lm.mc_batch_size = mc_batch_size
    lm.check_greedy = True
    lm.denoising_steps_per_block = None
    lm.response_aligned_blocks = False
    lm.mixed_precision_dtype = None
    lm._rank = 0
    lm.cache_hook = _CacheHook()
    return lm


def test_allocate_steps_distributes_remainder_to_early_blocks():
    assert get_model("hf-diffusion") is HFDiffusionLM
    assert HFDiffusionLM._allocate_steps(8, 3) == [3, 3, 2]
    with pytest.raises(ValueError, match="at least the number"):
        HFDiffusionLM._allocate_steps(2, 3)


def test_threshold_accepts_only_confident_tokens_before_final_step():
    lm = HFDiffusionLM.__new__(HFDiffusionLM)
    lm.threshold = 0.8
    confidence = torch.tensor([[0.91, 0.79, 0.80]])
    mutable = torch.tensor([[True, True, True]])

    updates, accepted_count, forced_count = lm._select_updates(
        confidence,
        mutable,
        initial_count=3,
        step_index=0,
        num_steps=2,
    )

    assert updates.tolist() == [[True, False, True]]
    assert accepted_count == 2
    assert forced_count == 0


def test_threshold_supplements_highest_confidence_to_meet_quota():
    lm = HFDiffusionLM.__new__(HFDiffusionLM)
    lm.threshold = 0.8
    confidence = torch.tensor([[0.91, 0.79, 0.70, 0.60]])
    mutable = torch.tensor([[True, True, True, True]])

    updates, accepted_count, forced_count = lm._select_updates(
        confidence,
        mutable,
        initial_count=4,
        step_index=0,
        num_steps=2,
    )

    assert updates.tolist() == [[True, True, False, False]]
    assert accepted_count == 2
    assert forced_count == 0


def test_threshold_makes_progress_when_no_token_reaches_threshold():
    lm = HFDiffusionLM.__new__(HFDiffusionLM)
    lm.threshold = 0.8
    confidence = torch.tensor([[0.70, 0.79, 0.60]])
    mutable = torch.tensor([[True, True, True]])

    updates, accepted_count, forced_count = lm._select_updates(
        confidence,
        mutable,
        initial_count=3,
        step_index=0,
        num_steps=3,
    )

    assert updates.tolist() == [[False, True, False]]
    assert accepted_count == 1
    assert forced_count == 0


def test_threshold_final_step_meets_quota_without_forced_tokens():
    lm = HFDiffusionLM.__new__(HFDiffusionLM)
    lm.threshold = 0.8
    confidence = torch.tensor([[0.91, 0.79, 0.80]])
    mutable = torch.tensor([[True, True, True]])

    updates, accepted_count, forced_count = lm._select_updates(
        confidence,
        mutable,
        initial_count=3,
        step_index=1,
        num_steps=2,
    )

    assert updates.tolist() == [[True, True, True]]
    assert accepted_count == 3
    assert forced_count == 0


def test_single_token_choices_reuse_context_forward():
    model = _FakeMaskedLM(preferred_token=1)
    lm = _make_lm(model)
    requests = [
        (("context", " A"), [0, 2], [1]),
        (("context", " B"), [0, 2], [2]),
    ]

    result = lm._loglikelihood_tokens(requests, disable_tqdm=True)

    assert model.calls == 1
    assert result[0][0] > result[1][0]
    assert result[0][1] is True
    assert result[1][1] is False


def test_uniform_mc_likelihood_equals_target_length_times_log_vocab():
    vocab_size = 4
    model = _FakeMaskedLM(vocab_size=vocab_size)
    lm = _make_lm(model, mc_num=7, mc_batch_size=3)

    score = lm._mc_loglikelihood([0, 1], [1, 2, 0])

    assert score == pytest.approx(-3 * math.log(vocab_size), rel=1e-6)


def test_tiny_qwen3_records_threshold_generation_metrics():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    lm = _make_lm(model)
    lm.tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=0)
    lm.block_length = 2
    lm.diffusion_steps = 4
    lm.diffusion_temperature = 0.0
    lm.threshold = 0.8
    lm.top_k = 0
    lm.top_p = 1.0
    lm._metrics_dir = "enabled-for-test"
    lm._perf_metrics = lm._new_perf_metrics()

    prompt = torch.tensor([[1, 4, 5]])
    output = lm._generate_one(prompt, max_new_tokens=4, temperature=0.0)

    assert output.shape == (1, 7)
    assert not output[:, prompt.shape[1] :].eq(lm.mask_token_id).any()
    assert lm._perf_metrics["forward_counts"] == {
        "prefill": 1,
        "denoise": 2,
        "causal": 2,
        "scoring": 0,
    }
    assert lm._perf_metrics["requests"][0]["block_steps"] == [1, 1]
    assert lm._perf_metrics["requests"][0][
        "block_forced_low_confidence_tokens"
    ] == [0, 0]
    assert lm._perf_metrics["requests"][0]["block_stalled_steps"] == [0, 0]


def test_tiny_qwen3_response_aligned_blocks_match_jetengine_steps():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    lm = _make_lm(model)
    lm.tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=0)
    lm.block_length = 2
    lm.diffusion_steps = None
    lm.denoising_steps_per_block = 2
    lm.response_aligned_blocks = True
    lm.diffusion_temperature = 1.0
    lm.threshold = 0.0
    lm.top_k = 1
    lm.top_p = 1.0
    lm._metrics_dir = "enabled-for-test"
    lm._perf_metrics = lm._new_perf_metrics()

    prompt = torch.tensor([[1, 4, 5]])
    output = lm._generate_one(prompt, max_new_tokens=4, temperature=1.0)

    assert output.shape == (1, 7)
    assert not output[:, prompt.shape[1] :].eq(lm.mask_token_id).any()
    assert lm._perf_metrics["forward_counts"] == {
        "prefill": 1,
        "denoise": 4,
        "causal": 2,
        "scoring": 0,
    }
    assert lm._perf_metrics["requests"][0]["block_steps"] == [2, 2]
    assert lm._perf_metrics["requests"][0]["block_allocated_steps"] == [2, 2]


def test_constructor_forces_eager_attention_on_loaded_checkpoint(tmp_path):
    unknown_piece = "<unk>"
    padding_piece = "<pad>"
    end_piece = "<eos>"
    mask_piece = "<mask>"
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=1,
    )
    config.mask_token_id = 3
    Qwen3ForCausalLM(config).save_pretrained(tmp_path)

    backend = Tokenizer(
        WordLevel(
            {
                padding_piece: 0,
                end_piece: 1,
                unknown_piece: 2,
                mask_piece: 3,
                "hello": 4,
                "world": 5,
                "A": 6,
                "B": 7,
            },
            unk_token=unknown_piece,
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unknown_piece,
        pad_token=padding_piece,
        eos_token=end_piece,
        mask_token=mask_piece,
    )
    tokenizer.save_pretrained(tmp_path)

    lm = HFDiffusionLM(
        pretrained=str(tmp_path),
        tokenizer=str(tmp_path),
        device="cpu",
        dtype="float32",
        batch_size=1,
        local_files_only=True,
        block_length=2,
        diffusion_steps=4,
    )

    assert lm.model.config._attn_implementation == "eager"
    assert lm.mask_token_id == 3
