#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-/data2/model_train/ckpt/qwen3_1.7b_baseline/models/global_step_10500}"
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-/data2/model_train/ckpt/phase4_cpt_ar_diffusion_0916/models/global_step_10500}"
COMPARISON_OUTPUT_ROOT="${COMPARISON_OUTPUT_ROOT:-./eval_results_generation_comparison}"
TASKS="${TASKS:-mmlu,gsm8k,boolq,bbh,arc_challenge,hellaswag,piqa,winogrande}"

echo "===== AR baseline generation suite ====="
EVAL_MODE=ar \
MODEL_PATH="${BASELINE_MODEL_PATH}" \
OUTPUT_ROOT="${COMPARISON_OUTPUT_ROOT}/ar_baseline" \
TASKS="${TASKS}" \
bash "${SCRIPT_DIR}/hf_generation_suite.sh" "$@"

echo "===== Trained CPT-Diffusion-AR checkpoint: AR mode ====="
EVAL_MODE=ar \
MODEL_PATH="${TRAINED_MODEL_PATH}" \
OUTPUT_ROOT="${COMPARISON_OUTPUT_ROOT}/cpt_diffusion_ar/ar" \
TASKS="${TASKS}" \
bash "${SCRIPT_DIR}/hf_generation_suite.sh" "$@"

echo "===== Trained CPT-Diffusion-AR checkpoint: Diffusion mode ====="
EVAL_MODE=diffusion \
MODEL_PATH="${TRAINED_MODEL_PATH}" \
OUTPUT_ROOT="${COMPARISON_OUTPUT_ROOT}/cpt_diffusion_ar/diffusion" \
TASKS="${TASKS}" \
bash "${SCRIPT_DIR}/hf_generation_suite.sh" "$@"
