# Hugging Face block-diffusion evaluation

The `hf-diffusion` model backend evaluates an exported hybrid AR + masked
diffusion checkpoint with `lm-evaluation-harness`. It is designed for
decoder-only checkpoints that can already be loaded by the ordinary `hf`
backend, including the Qwen3 diffusion checkpoint exported by MindSpeed-LLM.

The backend has two evaluation paths:

- `generate_until` uses causal prompt/KV-cache processing between blocks and
  bidirectional masked-token denoising inside every generated block.
- `loglikelihood` uses masked conditional-likelihood estimation. Single-token
  answer choices (MMLU and BoolQ) use one exact masked forward per question.
  Longer continuations use the Monte Carlo estimator from the official LLaDA
  evaluation implementation.

The model is loaded with standard `AutoModelForCausalLM`. The exported
checkpoint must therefore retain the standard backbone and `lm_head` parameter
names, contain the trained mask-token embedding, and store `mask_token_id` in
`config.json` (or receive it through `--model_args`). Transformers 5.2 or newer
is required for the per-layer attention-mask mapping used during denoising.

## Run the five-task suite

From the `lm-evaluation-harness` checkout:

```bash
MODEL_PATH=/path/to/global_step_102500 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash hf_diffusion_xyf2.sh
```

Before a full run, verify loading, the mask ID and dataset cache with a small
subset:

```bash
MODEL_PATH=/path/to/global_step_102500 \
ASCEND_RT_VISIBLE_DEVICES=0 \
TASKS=mmlu,gsm8k \
LIMIT=8 \
bash hf_diffusion_xyf2.sh
```

The script supports these environment overrides:

- `TASKS`: comma-separated subset of `mmlu,gsm8k,humaneval,boolq,bbh`.
- `BLOCK_LENGTH`: generated tokens per diffusion block; default `32`.
- `DIFFUSION_STEPS`: total denoising steps per request. If unset, it equals
  that request's generation length. It must be at least the number of blocks.
- `MASK_TOKEN_ID`: override `config.json` only when necessary.
- `DIFFUSION_TEMPERATURE`, `THRESHOLD`, `TOP_K`, `TOP_P`: sampling controls.
  A task's `do_sample: false` generation setting takes precedence and keeps the
  five-task suite deterministic.
- `MC_NUM`: Monte Carlo samples for multi-token conditional likelihood;
  default `128`. It does not increase the cost of single-token choices.
- `MC_BATCH_SIZE`: likelihood-estimation microbatch size; lower this after an
  out-of-memory error.
- `LIMIT`: harness sample limit for smoke tests.
- `OUTPUT_ROOT`: result root; default `./eval_results_diffusion`.
- `MODEL_DEVICE`: explicit single-process device. The script automatically uses
  `npu:0` when only one Ascend device is visible; override this for a CPU/CUDA
  development run.

`--batch_size 1` is intentional for generation: prompts in a batch have
different lengths, while block-diffusion KV-cache generation is performed one
unpadded prompt at a time. Multi-NPU execution still provides data parallelism
through `accelerate launch`.
