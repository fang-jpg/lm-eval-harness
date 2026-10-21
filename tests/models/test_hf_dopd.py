from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from lm_eval.api.registry import get_model
from lm_eval.models.hf_dopd import HFDOPDLM


class _RecordingQwen3(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.recorded_attention_masks = []

    def forward(self, *args, **kwargs):
        self.recorded_attention_masks.append(kwargs.get("attention_mask"))
        return super().forward(*args, **kwargs)


def _tiny_dopd_lm():
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
    model = _RecordingQwen3(config).eval()
    lm = HFDOPDLM.__new__(HFDOPDLM)
    lm._model = model
    lm._device = torch.device("cpu")
    lm._config = SimpleNamespace(layer_types=["full_attention"])
    lm._max_length = 32
    lm.mask_token_id = 3
    lm.block_length = 2
    lm.denoising_steps_per_block = 2
    lm.diffusion_temperature = 1.0
    lm.top_k = 1
    lm.top_p = 1.0
    lm.seed = 42
    lm._generation_calls = 0
    lm.mixed_precision_dtype = None
    lm._rank = 0
    lm.tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=0)
    lm._metrics_dir = "enabled-for-test"
    lm._perf_metrics = lm._new_perf_metrics()
    return lm, model


def test_hf_dopd_has_an_independent_registry_entry():
    assert get_model("hf-dopd") is HFDOPDLM
    assert get_model("hf-diffusion") is not HFDOPDLM


def test_dopd_transfer_schedule_covers_every_token():
    assert HFDOPDLM._transfer_schedule(4, 4) == [1, 1, 1, 1]
    assert HFDOPDLM._transfer_schedule(8, 4) == [2, 2, 2, 2]


def test_dopd_uses_all_mask_blocks_and_its_own_forward_schedule():
    lm, model = _tiny_dopd_lm()
    prompt = torch.tensor([[1, 4, 5]])

    output = lm._generate_one(prompt, max_new_tokens=4, temperature=1.0)

    assert output.shape == (1, 7)
    assert not output[:, prompt.shape[1] :].eq(lm.mask_token_id).any()
    assert lm._perf_metrics["forward_counts"] == {
        "prefill": 1,
        "denoise": 4,
        "saving": 2,
    }
    assert lm._perf_metrics["requests"][0]["block_steps"] == [2, 2]
    assert lm._perf_metrics["requests"][0]["block_allocated_steps"] == [2, 2]
    assert len(model.recorded_attention_masks) == 7
    assert all(isinstance(mask, dict) for mask in model.recorded_attention_masks)
