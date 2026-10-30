# AR vs block-diffusion Ascend operator profiling

This repository can collect a bounded CANN Level1 trace for one representative
generation region. Profiling is opt-in and does not change normal evaluation.

## Recommended comparison

Run on an Atlas A2 server from the `lm-evaluation-harness` checkout:

```bash
AR_MODEL_PATH=/path/to/ar-checkpoint \
DIFFUSION_MODEL_PATH=/path/to/joint-checkpoint \
PROFILE_DEVICE=0 \
BLOCK_LENGTH=4 \
THRESHOLD=0.8 \
THRESHOLD_MODE=strict \
MAX_DENOISING_STEPS_PER_BLOCK=3 \
bash profile_ar_diffusion_operators.sh
```

The wrapper uses two GSM8K samples on one NPU. The first request warms up the
model; the second is generated to 256 tokens. It first captures two complete
request traces, which provide the per-operator total-time comparison:

- `ar_request`
- `diffusion_request`

It then captures five bounded stage traces near generated token 128, which
explain the per-call cost:

- `ar_decode_forward`
- `diffusion_denoise_forward`
- `diffusion_causal_refresh_forward`
- `diffusion_cache_copy`
- `diffusion_sample_and_select`

Every target is run in a fresh process and with a fresh response-cache path.
This is required because an lm-eval cache hit performs no model forward.
Set `RUN_STAGE_PROFILES=0` to collect only the two request totals, or
`RUN_FULL_REQUEST_PROFILES=0` to collect only the smaller stage traces.
The wrapper preserves normal diffusion EOS checks by default. If a checkpoint
actually stops before 256 tokens, set `PROFILE_IGNORE_EOS=1`; record that as a
separate experiment because removing the per-block EOS check also removes its
host/device synchronization cost.

The final report is:

```text
<PROFILE_ROOT>/comparison/operator_comparison.md
```

Per-profile CSV summaries and `profile_totals.json` are written beside it. Raw
CANN exports remain under each target's `trace/` directory for MindStudio.
`Region wall ms` is synchronized around the selected request/stage, so it also
shows time spent in Python or runtime work that produces no NPU kernel row.

## Direct profiling controls

Both `hf_xyf2.sh` and `hf_diffusion_xyf2.sh` accept these environment variables:

| Variable | Meaning | Default in profiling mode |
|---|---|---|
| `NPU_PROFILE=1` | Enable profiling | disabled |
| `HF_NPU_PROFILE_DIR` | Raw trace root | below evaluation output |
| `HF_NPU_PROFILE_SCOPE` | `forward` or `request` | `forward` |
| `HF_NPU_PROFILE_REQUEST_START` | Zero-based request to profile | `1` |
| `HF_NPU_PROFILE_TARGET` | Named forward/region | backend-specific |
| `HF_NPU_PROFILE_TARGET_OCCURRENCE` | Zero-based occurrence within the selected request | backend-specific |
| `HF_NPU_PROFILE_RANKS` | Comma-separated global ranks or `all` | `0` |
| `HF_NPU_PROFILE_LEVEL` | `level0`, `level1`, or `level2` | `level1` |
| `HF_NPU_PROFILE_WITH_CPU` | Include CPU events/markers | `1` |
| `HF_NPU_PROFILE_RECORD_SHAPES` | Record tensor shapes | `1` |

Request scope captures cumulative behavior but a 256-token trace can be very
large:

```bash
NPU_PROFILE=1 \
HF_NPU_PROFILE_SCOPE=request \
HF_NPU_PROFILE_REQUEST_START=1 \
LIMIT=2 \
bash hf_xyf2.sh
```

## How to read the comparison

Check these hypotheses in order:

1. **Attention backend mismatch.** `hf-diffusion` forces eager attention so it
   can pass a bidirectional mask. If AR contains `IncreFlashAttention` while
   diffusion contains separate `MatMul`, mask and `Softmax` kernels, fewer
   forwards may still take longer.
2. **Query length.** AR decode has query length 1. Denoise and causal refresh
   have query length `B`, so projection, MLP, logits and attention operators do
   more work in every diffusion forward.
3. **KV-cache copy.** `diffusion_cache_copy` identifies device copy kernels and
   CPU gaps caused by `copy.deepcopy(past_key_values)` before every denoise.
4. **Full-vocabulary selection.** `diffusion_sample_and_select` exposes the
   `Softmax`, `ArgMax`, `TopK`, masking and indexing cost outside the transformer.
5. **Host gaps.** Compare `Device busy ms` with `Trace span ms`. A low busy/span
   ratio indicates Python/cache/dispatch overhead rather than slow NPU kernels.

`Sum kernel ms` is the sum of individual kernel durations and can exceed wall
time when streams overlap. `Device busy ms` merges overlapping intervals and is
the safer device-active comparison.

Do not compare profiled tok/s with normal tok/s: profiling adds synchronization
and trace overhead. Use the trace only to explain normal-run timings.
