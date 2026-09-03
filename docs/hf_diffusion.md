# Hugging Face block-diffusion evaluation

The `hf-diffusion` model backend evaluates an exported hybrid AR + masked
diffusion checkpoint with `lm-evaluation-harness`. It is designed for
decoder-only checkpoints that can already be loaded by the ordinary `hf`
backend, including the Qwen3 diffusion checkpoint exported by MindSpeed-LLM.

The backend has two evaluation paths:

- `generate_until` uses causal prompt/KV-cache processing between blocks and
  bidirectional masked-token denoising inside every generated block.
- `loglikelihood` uses LLaDA-style masked conditional-likelihood estimation.
  This path is not used by the five-task script because the MindSpeed Qwen3
  checkpoint is trained with the `block_diff` `[noisy, original]` layout and
  its special block attention mask.

In particular, the stock `mmlu` and `boolq` tasks are multiple-choice
likelihood tasks. The script selects `mmlu_generative` and `boolq-seq2seq`
instead, so all five benchmarks exercise `generate_until`. MMLU uses the same
prompt prefix and A-D answer extraction rules as MindSpeed-LLM's
`evaluate_qwen3_diffusion.py`; BoolQ similarly extracts a generated `yes` or
`no`. Both tasks generate at most 32 tokens (one default diffusion block).

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
- `MC_NUM` and `MC_BATCH_SIZE`: only affect custom likelihood tasks; they are
  not used by the five-task generation suite.
- `LIMIT`: harness sample limit for smoke tests.
- `OUTPUT_ROOT`: result root; default `./eval_results_diffusion`.
- `MODEL_DEVICE`: explicit single-process device. The script automatically uses
  `npu:0` when only one Ascend device is visible; override this for a CPU/CUDA
  development run.

`--batch_size 1` is intentional for generation: prompts in a batch have
different lengths, while block-diffusion KV-cache generation is performed one
unpadded prompt at a time. Multi-NPU execution still provides data parallelism
through `accelerate launch`.

For an AR-versus-diffusion comparison, rerun the AR model with
`mmlu_generative` and `boolq-seq2seq` as well. The stock AR results from
`mmlu`/`boolq` use option likelihood and are not directly comparable with the
generation metrics (`exact_match,get_response`).
