import torch

from lm_eval.api.registry import get_model
from lm_eval.models.hf_ar_metrics import HFARMetricsLM
from lm_eval.models.huggingface import HFLM


class _FakeARModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        del kwargs
        return input_ids


def test_ar_metrics_backend_counts_prefill_and_decode_without_changing_output(
    monkeypatch,
):
    def fake_hf_generate(
        self,
        context,
        max_length,
        stop,
        attention_mask=None,
        **generation_kwargs,
    ):
        del max_length, stop, attention_mask, generation_kwargs
        generated = context
        for token_id in (6, 7, 8):
            model_input = (
                generated
                if generated.shape[1] == context.shape[1]
                else generated[:, -1:]
            )
            self.model(input_ids=model_input)
            next_token = torch.full(
                (generated.shape[0], 1),
                token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    monkeypatch.setattr(HFLM, "_model_generate", fake_hf_generate)
    lm = HFARMetricsLM.__new__(HFARMetricsLM)
    lm._model = _FakeARModel()
    lm._device = torch.device("cpu")
    lm._rank = 0
    lm._world_size = 1
    lm.batch_size_per_gpu = 1
    lm._ar_metrics_dir = "enabled-for-test"
    lm._ar_perf_metrics = lm._new_ar_perf_metrics()

    context = torch.tensor([[1, 2, 3]])
    output = lm._model_generate(
        context=context,
        max_length=6,
        stop=[],
    )

    assert get_model("hf-ar-metrics") is HFARMetricsLM
    assert output.tolist() == [[1, 2, 3, 6, 7, 8]]
    assert lm._ar_perf_metrics["forward_counts"] == {
        "prefill": 1,
        "decode": 2,
        "scoring": 0,
    }
    assert lm._ar_perf_metrics["generation_requests"] == 1
    assert lm._ar_perf_metrics["raw_generated_tokens"] == 3
    request = lm._ar_perf_metrics["requests"][0]
    assert request["actual_forward_count"] == 3
    assert request["prefill_forward_count"] == 1
    assert request["decode_forward_count"] == 2
