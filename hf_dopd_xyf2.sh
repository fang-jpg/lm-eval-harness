#!/usr/bin/env bash
set -euo pipefail

# Evaluate the final d-OPD FSDP2/DCP checkpoint with the existing
# hf-diffusion lm-eval backend. No training checkpoint is modified: the
# Hugging Face export is written to ${DCP_MODEL_PATH}/hf_ckpt by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_ROOT="${MINDSPEED_ROOT:-${SCRIPT_DIR}/../MindSpeed-LLM}"

DCP_MODEL_PATH="${DCP_MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_dopd_1010/checkpoints/global_step_483_final}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${DCP_MODEL_PATH%/}/hf_ckpt}"
DOPD_CONFIG="${DOPD_CONFIG:-$(dirname "$(dirname "${DCP_MODEL_PATH%/}")")/dopd_config.json}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"

if [[ ! -d "${DCP_MODEL_PATH}" ]]; then
  echo "d-OPD checkpoint directory does not exist: ${DCP_MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DCP_MODEL_PATH}/.metadata" ]]; then
  echo "Not a valid DCP checkpoint (missing .metadata): ${DCP_MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MINDSPEED_ROOT}/mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py" ]]; then
  echo "MindSpeed-LLM not found at: ${MINDSPEED_ROOT}" >&2
  echo "Set MINDSPEED_ROOT to the MindSpeed-LLM repository root." >&2
  exit 2
fi

# Prefer an explicit BASE_MODEL_PATH. Otherwise recover the exact Base model
# path used by training from the resolved d-OPD configuration.
if [[ -z "${BASE_MODEL_PATH}" && -f "${DOPD_CONFIG}" ]]; then
  BASE_MODEL_PATH="$(python -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("model", {}).get("model_name_or_path", ""))' "${DOPD_CONFIG}")"
fi
if [[ -z "${BASE_MODEL_PATH}" || ! -f "${BASE_MODEL_PATH}/config.json" ]]; then
  echo "Unable to find the Qwen3-1.7B-Base model configuration." >&2
  echo "Resolved path: ${BASE_MODEL_PATH:-<empty>}" >&2
  echo "Set BASE_MODEL_PATH=/path/to/Qwen3-1.7B-Base and rerun." >&2
  exit 2
fi

# lm-eval loads Hugging Face weights, while d-OPD training saves a distributed
# checkpoint. Convert once and reuse the completed export on later evaluations.
if [[ ! -f "${HF_MODEL_PATH}/model.safetensors" && \
      ! -f "${HF_MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "===== Converting d-OPD DCP checkpoint to Hugging Face ====="
  echo "DCP source : ${DCP_MODEL_PATH}"
  echo "HF output  : ${HF_MODEL_PATH}"
  echo "Base config: ${BASE_MODEL_PATH}"
  python "${MINDSPEED_ROOT}/mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py" \
    --load-dir "${DCP_MODEL_PATH}" \
    --save-dir "${HF_MODEL_PATH}" \
    --model-configs "${BASE_MODEL_PATH}"
else
  echo "===== Reusing existing Hugging Face export: ${HF_MODEL_PATH} ====="
fi

if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
  echo "Converted model is missing config.json: ${HF_MODEL_PATH}" >&2
  exit 2
fi

# merge_dcp_to_hf saves the model config and processor when available. Ensure
# a text-only Qwen tokenizer is also present because hf_diffusion_xyf2.sh uses
# MODEL_PATH for both the model and tokenizer.
if [[ ! -f "${HF_MODEL_PATH}/tokenizer_config.json" ]]; then
  python -c 'from transformers import AutoTokenizer; import sys; AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=True).save_pretrained(sys.argv[2])' \
    "${BASE_MODEL_PATH}" "${HF_MODEL_PATH}"
fi

echo "===== Evaluating d-OPD checkpoint with hf-diffusion ====="
echo "DCP checkpoint: ${DCP_MODEL_PATH}"
echo "HF model      : ${HF_MODEL_PATH}"

# d-OPD was trained with block_size=4 and denoise_steps=4. Leaving
# DIFFUSION_STEPS empty makes hf-diffusion use max_new_tokens total steps,
# which allocates approximately four denoising steps to every four-token block.
MODEL_PATH="${HF_MODEL_PATH}" \
TASKS="${TASKS:-mmlu,boolq,bbh}" \
BLOCK_LENGTH="${BLOCK_LENGTH:-4}" \
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}" \
MASK_TOKEN_ID="${MASK_TOKEN_ID:-151669}" \
DIFFUSION_TEMPERATURE="${DIFFUSION_TEMPERATURE:-0.0}" \
THRESHOLD="${THRESHOLD:-0.0}" \
TOP_K="${TOP_K:-0}" \
TOP_P="${TOP_P:-1.0}" \
MC_NUM="${MC_NUM:-128}" \
MC_BATCH_SIZE="${MC_BATCH_SIZE:-16}" \
SEED="${SEED:-42}" \
LIMIT="${LIMIT:-}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_dopd}" \
bash "${SCRIPT_DIR}/hf_diffusion_xyf2.sh" "$@"
