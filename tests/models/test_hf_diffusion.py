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
    lm.mixed_precision_dtype = None
    lm._rank = 0
    lm.cache_hook = _CacheHook()
    return lm


def test_allocate_steps_distributes_remainder_to_early_blocks():
    assert get_model("hf-diffusion") is HFDiffusionLM
    assert HFDiffusionLM._allocate_steps(8, 3) == [3, 3, 2]
    with pytest.raises(ValueError, match="at least the number"):
        HFDiffusionLM._allocate_steps(2, 3)


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


def test_tiny_qwen3_runs_cached_block_diffusion_generation():
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
    lm.threshold = 0.0
    lm.top_k = 0
    lm.top_p = 1.0

    prompt = torch.tensor([[1, 4, 5]])
    output = lm._generate_one(prompt, max_new_tokens=4, temperature=0.0)

    assert output.shape == (1, 7)
    assert not output[:, prompt.shape[1] :].eq(lm.mask_token_id).any()


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
