# Hugging Face block-diffusion evaluation

The `hf-diffusion` model backend evaluates an exported hybrid AR + masked
diffusion checkpoint with `lm-evaluation-harness`. It is designed for
decoder-only checkpoints that can already be loaded by the ordinary `hf`
backend, including the Qwen3 diffusion checkpoint exported by MindSpeed-LLM.

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

The existing `hf_xyf2.sh` and `hf_diffusion_xyf2.sh` launchers are unchanged.
The extended generation suite uses the separate `hf_generation_suite.sh`
runner. From the `lm-evaluation-harness` checkout, run only the AR baseline
with:

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
- `DIFFUSION_TEMPERATURE`, `THRESHOLD`, `TOP_K`, `TOP_P`: sampling controls.
  A task's `do_sample: false` generation setting takes precedence and keeps the
  suite deterministic.
- `MC_NUM` and `MC_BATCH_SIZE`: only affect custom likelihood tasks; they are
  not used by the generation suite.
- `LIMIT`: harness sample limit for smoke tests.
- `EVAL_MODE`: `ar` selects the ordinary `hf` backend; `diffusion` selects
  `hf-diffusion`.
- `OUTPUT_ROOT`: result root for a direct runner invocation.
- `COMPARISON_OUTPUT_ROOT`: root for the three-way comparison. Results are
  separated into `ar_baseline`, `cpt_diffusion_ar/ar`, and
  `cpt_diffusion_ar/diffusion`.
- `MODEL_DEVICE`: explicit single-process device. The script automatically uses
  `npu:0` when only one Ascend device is visible; override this for a CPU/CUDA
  development run.
- `ENABLE_UNSAFE_CODE=1`: explicit opt-in required when `TASKS` contains
  `humaneval`.

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
