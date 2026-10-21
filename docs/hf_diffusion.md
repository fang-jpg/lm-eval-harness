# Hugging Face block-diffusion evaluation

The `hf-diffusion` model backend evaluates an exported hybrid AR + masked
diffusion checkpoint with `lm-evaluation-harness`. It is designed for
decoder-only checkpoints that can already be loaded by the ordinary `hf`
backend, including the Qwen3 diffusion checkpoint exported by MindSpeed-LLM.

The d-OPD evaluator is deliberately separate. `hf_dopd_xyf2.sh` uses the
independent `hf-dopd` backend in `lm_eval/models/hf_dopd.py`; it does not route
response-aligned d-OPD generation through this hybrid decoder.

The backend has two evaluation paths:

- `generate_until` uses causal prompt/KV-cache processing between blocks and
  bidirectional masked-token denoising inside every generated block.
- `loglikelihood` uses LLaDA-style masked conditional-likelihood estimation.
  This path is not used by the generation suite because the MindSpeed Qwen3
  checkpoint is trained with the `block_diff` `[noisy, original]` layout and
  its special block attention mask.

In particular, the stock `mmlu`, `boolq`, `arc_challenge`, `hellaswag`, `piqa`,
and `winogrande` tasks use option likelihood. The suite selects generative
variants instead. The comparison suite runs the baseline checkpoint in AR
mode, then runs the trained CPT-Diffusion-AR checkpoint once in AR mode and
once in Diffusion mode. All three runs use identical prompts, few-shot
settings, answer extraction, and metrics. The short-answer tasks generate at
most 32 tokens and ignore a hallucinated next `Question:` or `Context:` when
extracting an answer.

The model is loaded with standard `AutoModelForCausalLM`. The exported
checkpoint must therefore retain the standard backbone and `lm_head` parameter
names, contain the trained mask-token embedding, and store `mask_token_id` in
`config.json` (or receive it through `--model_args`). Transformers 5.2 or newer
is required for the per-layer attention-mask mapping used during denoising.

## Run the three-way comparison

The extended generation suite uses the separate `hf_generation_suite.sh`
runner. The `hf_diffusion_xyf2.sh` launcher also supports confidence-threshold
decoding and writes a combined accuracy/performance summary for each task.
From the `lm-evaluation-harness` checkout, run only the AR baseline with:

```bash
EVAL_MODE=ar \
MODEL_PATH=/path/to/ar/global_step_500 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash hf_generation_suite.sh
```

Run the trained checkpoint in AR mode with:

```bash
EVAL_MODE=ar \
MODEL_PATH=/path/to/trained/global_step_102500 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash hf_generation_suite.sh
```

Run that same trained checkpoint in Diffusion mode with:

```bash
EVAL_MODE=diffusion \
MODEL_PATH=/path/to/trained/global_step_102500 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash hf_generation_suite.sh
```

Or run all three evaluations sequentially:

```bash
BASELINE_MODEL_PATH=/path/to/ar/global_step_500 \
TRAINED_MODEL_PATH=/path/to/trained/global_step_102500 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash hf_compare_generation_xyf2.sh
```

Before a full run, verify model loading, the diffusion mask ID and dataset
cache with a small subset:

```bash
BASELINE_MODEL_PATH=/path/to/ar/global_step_500 \
TRAINED_MODEL_PATH=/path/to/trained/global_step_102500 \
ASCEND_RT_VISIBLE_DEVICES=0 \
TASKS=arc_challenge,hellaswag,piqa,winogrande \
LIMIT=8 \
bash hf_compare_generation_xyf2.sh
```

The shared runner supports these environment overrides:

- `TASKS`: comma-separated subset of `mmlu,gsm8k,boolq,bbh,humaneval,`
  `arc_challenge,hellaswag,piqa,winogrande`. The safe default excludes
  `humaneval`.
- `BLOCK_LENGTH`: generated tokens per diffusion block; default `8`.
- `DIFFUSION_STEPS`: total denoising steps per request. If unset, it equals
  that request's generation length. It must be at least the number of blocks.
- `MASK_TOKEN_ID`: override `config.json` only when necessary.
- `THRESHOLD`: confidence threshold for committing masked tokens. `0` preserves
  scheduled quota decoding. Values in `(0, 1]` commit every token at or above
  the threshold, then supplement the selection with the highest-confidence
  remaining tokens when needed to satisfy the current scheduled quota. This
  guarantees progress on every denoising step and prevents a stalled block from
  forcing all remaining masks at the end. `hf_diffusion_xyf2.sh` defaults this
  to `0`, which preserves the 2026-10-12 scheduled hybrid decoder.
- `THRESHOLD_MODE`: `scheduled` (the default) preserves the fixed denoising
  quota above. `strict` commits every token whose raw softmax probability is
  at least `THRESHOLD`. If no position reaches the threshold, it commits only
  the single highest-confidence proposal so the next forward sees updated
  block context. The first token of each block remains the unconditional AR
  seed required by the restored 2026-10-12 hybrid decoder; the threshold
  governs the remaining masked positions.
- `MAX_DENOISING_STEPS_PER_BLOCK`: required in `strict` mode and used as a
  finite safety limit. It must be at least `BLOCK_LENGTH - 1`, because every
  strict step commits at least one of the non-seed positions. There is no
  final-step force-all. Leave `DIFFUSION_STEPS` empty in this mode.
- `TRACE_REQUESTS_PER_RANK` and `TRACE_BLOCKS_PER_REQUEST`: bound the optional
  token-probability trace written to the summary log. The trace shows every
  still-masked position's proposed token, raw probability, threshold decision,
  one-token progress fallback, and the completed block. Both default to `0`
  so full evaluations do not create very large logs.
- `DIFFUSION_TEMPERATURE`, `TOP_K`, `TOP_P`: sampling controls. A task's
  `do_sample: false` generation setting takes precedence and keeps the suite
  deterministic.
- `MC_NUM` and `MC_BATCH_SIZE`: only affect custom likelihood tasks; they are
  not used by the generation suite.
- `LIMIT`: harness sample limit for smoke tests.
- `EVAL_MODE`: `ar` selects the ordinary `hf` backend; `diffusion` selects
  `hf-diffusion`.
- `OUTPUT_ROOT`: result root for a direct runner invocation.
- `DECODER_CACHE_VERSION`: response-cache namespace for decoder semantics. The
  runner uses a hybrid-causal namespace so d-OPD responses and responses from
  the former mixed decoder cannot be reused.
- `COMPARISON_OUTPUT_ROOT`: root for the three-way comparison. Results are
  separated into `ar_baseline`, `cpt_diffusion_ar/ar`, and
  `cpt_diffusion_ar/diffusion`.
- `MODEL_DEVICE`: explicit single-process device. The script automatically uses
  `npu:0` when only one Ascend device is visible; override this for a CPU/CUDA
  development run.
- `ENABLE_UNSAFE_CODE=1`: explicit opt-in required when `TASKS` contains
  `humaneval`.

For every successfully completed task, `hf_diffusion_xyf2.sh` creates a
`*_summary.log` beside the ordinary task log. It combines the lm-eval accuracy
metrics with actual prefill, denoise, causal and scoring forward counts;
decode-only tokens/second; maximum per-rank allocated/reserved device memory;
aggregate block-step statistics; one compact per-request list containing the
actual step count for every generated block; and any requested bounded token
probability traces.
Rank-local machine-readable JSON is retained below `METRICS_ROOT`, which
defaults to `<run output>/metrics`.

For example, strict threshold decoding with a short trace can be launched as:

```bash
TASKS=gsm8k \
BLOCK_LENGTH=8 \
DIFFUSION_STEPS= \
THRESHOLD_MODE=strict \
THRESHOLD=0.8 \
MAX_DENOISING_STEPS_PER_BLOCK=8 \
TRACE_REQUESTS_PER_RANK=1 \
TRACE_BLOCKS_PER_REQUEST=2 \
bash hf_diffusion_xyf2.sh
```

### CANN environment discovery

The shared runner does not require a server-specific CANN path. It resolves the
environment in this order:

1. Reuse the current environment when `libhccl.so` is already loadable.
2. Source the explicit `CANN_ENV` script, when provided.
3. Search below `CANN_ROOT`, `ASCEND_HOME_PATH`, and `ASCEND_TOOLKIT_HOME`.
4. Search common `/usr/local/Ascend`, `/opt/ascend`, `/home/*`, and `/share/*`
   CANN layouts.

For a one-off run on a server with a nonstandard installation, use a path that
is visible inside the container:

```bash
CANN_ROOT=/home/f50058111/Ascend910B_CANN_9.1.0 \
bash hf_compare_generation_xyf2.sh
```

or point directly at the official environment script:

```bash
CANN_ENV=/home/f50058111/Ascend910B_CANN_9.1.0/cann-9.1.0/set_env.sh \
bash hf_compare_generation_xyf2.sh
```

If the host directory is not visible after `docker exec`, the container must be
recreated with a bind mount. Prefer a stable container path such as
`/opt/ascend/cann`, then set `CANN_ROOT=/opt/ascend/cann` independently of the
host installation path.

`--batch_size 1` is intentional for generation: prompts in a batch have
different lengths, while block-diffusion KV-cache generation is performed one
unpadded prompt at a time. Multi-NPU execution still provides data parallelism
through `accelerate launch`.

The task aliases map to `mmlu_generative`, `boolq-seq2seq`,
`arc_challenge_generative`, `hellaswag_generative`, `piqa_generative`, and
`winogrande_generative`. Stock likelihood-task scores are not directly
comparable with these generation metrics (`exact_match,get_response`).

## Dataset cache requirements

Offline evaluation requires the corresponding Hugging Face datasets to be
present under `HF_HOME`. The four added task variants use:

- `allenai/ai2_arc`
- `Rowan/hellaswag`
- `baber/piqa`
- `allenai/winogrande`

Use `LIMIT=8` for each task after populating the cache and before starting a
full multi-NPU run.
