#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE_ROOT="${PROFILE_ROOT:-${SCRIPT_DIR}/operator_profiles_$(date +%Y%m%d_%H%M%S)}"
AR_MODEL_PATH="${AR_MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_ar_cpt_baseline_0920/models/global_step_102500/}"
DIFFUSION_MODEL_PATH="${DIFFUSION_MODEL_PATH:-/share/f50058111/model_train/ckpt/phase4_cpt_ar_duffision_0916/models/global_step_102500/}"
PROFILE_DEVICE="${PROFILE_DEVICE:-0}"
PROFILE_LIMIT="${PROFILE_LIMIT:-2}"
PROFILE_REQUEST_START="${PROFILE_REQUEST_START:-1}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-256}"
RUN_FULL_REQUEST_PROFILES="${RUN_FULL_REQUEST_PROFILES:-1}"
RUN_STAGE_PROFILES="${RUN_STAGE_PROFILES:-1}"
PROFILE_IGNORE_EOS="${PROFILE_IGNORE_EOS:-0}"

BLOCK_LENGTH="${BLOCK_LENGTH:-4}"
THRESHOLD="${THRESHOLD:-0.8}"
THRESHOLD_MODE="${THRESHOLD_MODE:-strict}"
MAX_DENOISING_STEPS_PER_BLOCK="${MAX_DENOISING_STEPS_PER_BLOCK:-$((BLOCK_LENGTH - 1))}"
if [[ "${THRESHOLD_MODE}" != "strict" ]]; then
  MAX_DENOISING_STEPS_PER_BLOCK=""
fi

# Occurrences are zero-based within the second (profiled) request. These
# defaults sample approximately the same generated position (~token 128) for
# AR and B=4 diffusion. Override them for another block/threshold setting.
AR_DECODE_OCCURRENCE="${AR_DECODE_OCCURRENCE:-127}"
DIFFUSION_DENOISE_OCCURRENCE="${DIFFUSION_DENOISE_OCCURRENCE:-56}"
DIFFUSION_REFRESH_OCCURRENCE="${DIFFUSION_REFRESH_OCCURRENCE:-31}"
DIFFUSION_CACHE_COPY_OCCURRENCE="${DIFFUSION_CACHE_COPY_OCCURRENCE:-56}"
DIFFUSION_SAMPLE_OCCURRENCE="${DIFFUSION_SAMPLE_OCCURRENCE:-56}"

GEN_KWARGS_JSON="${GEN_KWARGS_JSON:-{\"do_sample\":false,\"max_gen_toks\":${MAX_GEN_TOKENS},\"min_new_tokens\":${MAX_GEN_TOKENS},\"until\":[]}}"

mkdir -p "${PROFILE_ROOT}"

run_ar_profile() {
  local target="$1"
  local occurrence="$2"
  local name="$3"
  local root="${PROFILE_ROOT}/${name}"
  echo "[PROFILE] ${name}: target=${target} occurrence=${occurrence}"
  ASCEND_RT_VISIBLE_DEVICES="${PROFILE_DEVICE}" \
  MODEL_PATH="${AR_MODEL_PATH}" \
  OUTPUT_ROOT="${root}/run" \
  LIMIT="${PROFILE_LIMIT}" \
  GEN_KWARGS_JSON="${GEN_KWARGS_JSON}" \
  NPU_PROFILE=1 \
  HF_NPU_PROFILE_DIR="${root}/trace" \
  HF_NPU_PROFILE_SCOPE=forward \
  HF_NPU_PROFILE_REQUEST_START="${PROFILE_REQUEST_START}" \
  HF_NPU_PROFILE_REQUEST_COUNT=1 \
  HF_NPU_PROFILE_TARGET="${target}" \
  HF_NPU_PROFILE_TARGET_OCCURRENCE="${occurrence}" \
  HF_NPU_PROFILE_RANKS=0 \
  bash "${SCRIPT_DIR}/hf_xyf2.sh"
}

run_ar_request_profile() {
  local root="${PROFILE_ROOT}/ar_request"
  echo "[PROFILE] ar_request: complete 256-token generation request"
  ASCEND_RT_VISIBLE_DEVICES="${PROFILE_DEVICE}" \
  MODEL_PATH="${AR_MODEL_PATH}" \
  OUTPUT_ROOT="${root}/run" \
  LIMIT="${PROFILE_LIMIT}" \
  GEN_KWARGS_JSON="${GEN_KWARGS_JSON}" \
  NPU_PROFILE=1 \
  HF_NPU_PROFILE_DIR="${root}/trace" \
  HF_NPU_PROFILE_SCOPE=request \
  HF_NPU_PROFILE_REQUEST_START="${PROFILE_REQUEST_START}" \
  HF_NPU_PROFILE_REQUEST_COUNT=1 \
  HF_NPU_PROFILE_RANKS=0 \
  bash "${SCRIPT_DIR}/hf_xyf2.sh"
}

run_diffusion_profile() {
  local target="$1"
  local occurrence="$2"
  local name="$3"
  local root="${PROFILE_ROOT}/${name}"
  echo "[PROFILE] ${name}: target=${target} occurrence=${occurrence}"
  ASCEND_RT_VISIBLE_DEVICES="${PROFILE_DEVICE}" \
  MODEL_PATH="${DIFFUSION_MODEL_PATH}" \
  TASKS=gsm8k \
  OUTPUT_ROOT="${root}/run" \
  RESPONSE_CACHE_ROOT="${root}/response_cache" \
  MAX_TASK_ATTEMPTS=1 \
  LIMIT="${PROFILE_LIMIT}" \
  GEN_KWARGS_JSON="${GEN_KWARGS_JSON}" \
  BLOCK_LENGTH="${BLOCK_LENGTH}" \
  THRESHOLD="${THRESHOLD}" \
  THRESHOLD_MODE="${THRESHOLD_MODE}" \
  MAX_DENOISING_STEPS_PER_BLOCK="${MAX_DENOISING_STEPS_PER_BLOCK}" \
  IGNORE_EOS="${PROFILE_IGNORE_EOS}" \
  NPU_PROFILE=1 \
  HF_NPU_PROFILE_DIR="${root}/trace" \
  HF_NPU_PROFILE_SCOPE=forward \
  HF_NPU_PROFILE_REQUEST_START="${PROFILE_REQUEST_START}" \
  HF_NPU_PROFILE_REQUEST_COUNT=1 \
  HF_NPU_PROFILE_TARGET="${target}" \
  HF_NPU_PROFILE_TARGET_OCCURRENCE="${occurrence}" \
  HF_NPU_PROFILE_RANKS=0 \
  bash "${SCRIPT_DIR}/hf_diffusion_xyf2.sh"
}

run_diffusion_request_profile() {
  local root="${PROFILE_ROOT}/diffusion_request"
  echo "[PROFILE] diffusion_request: complete 256-token generation request"
  ASCEND_RT_VISIBLE_DEVICES="${PROFILE_DEVICE}" \
  MODEL_PATH="${DIFFUSION_MODEL_PATH}" \
  TASKS=gsm8k \
  OUTPUT_ROOT="${root}/run" \
  RESPONSE_CACHE_ROOT="${root}/response_cache" \
  MAX_TASK_ATTEMPTS=1 \
  LIMIT="${PROFILE_LIMIT}" \
  GEN_KWARGS_JSON="${GEN_KWARGS_JSON}" \
  BLOCK_LENGTH="${BLOCK_LENGTH}" \
  THRESHOLD="${THRESHOLD}" \
  THRESHOLD_MODE="${THRESHOLD_MODE}" \
  MAX_DENOISING_STEPS_PER_BLOCK="${MAX_DENOISING_STEPS_PER_BLOCK}" \
  IGNORE_EOS="${PROFILE_IGNORE_EOS}" \
  NPU_PROFILE=1 \
  HF_NPU_PROFILE_DIR="${root}/trace" \
  HF_NPU_PROFILE_SCOPE=request \
  HF_NPU_PROFILE_REQUEST_START="${PROFILE_REQUEST_START}" \
  HF_NPU_PROFILE_REQUEST_COUNT=1 \
  HF_NPU_PROFILE_RANKS=0 \
  bash "${SCRIPT_DIR}/hf_diffusion_xyf2.sh"
}

SUMMARY_ARGS=()
if [[ "${RUN_FULL_REQUEST_PROFILES}" == "1" ]]; then
  run_ar_request_profile
  run_diffusion_request_profile
  SUMMARY_ARGS+=(
    --profile "ar_request=${PROFILE_ROOT}/ar_request/trace"
    --profile "diffusion_request=${PROFILE_ROOT}/diffusion_request/trace"
  )
fi

if [[ "${RUN_STAGE_PROFILES}" == "1" ]]; then
  run_ar_profile ar_decode_forward "${AR_DECODE_OCCURRENCE}" ar_decode
  run_diffusion_profile diffusion_denoise_forward "${DIFFUSION_DENOISE_OCCURRENCE}" diffusion_denoise
  run_diffusion_profile diffusion_causal_refresh_forward "${DIFFUSION_REFRESH_OCCURRENCE}" diffusion_refresh
  run_diffusion_profile diffusion_cache_copy "${DIFFUSION_CACHE_COPY_OCCURRENCE}" diffusion_cache_copy
  run_diffusion_profile diffusion_sample_and_select "${DIFFUSION_SAMPLE_OCCURRENCE}" diffusion_sample
  SUMMARY_ARGS+=(
    --profile "ar_decode=${PROFILE_ROOT}/ar_decode/trace"
    --profile "diffusion_denoise=${PROFILE_ROOT}/diffusion_denoise/trace"
    --profile "diffusion_refresh=${PROFILE_ROOT}/diffusion_refresh/trace"
    --profile "diffusion_cache_copy=${PROFILE_ROOT}/diffusion_cache_copy/trace"
    --profile "diffusion_sample=${PROFILE_ROOT}/diffusion_sample/trace"
  )
fi

if (( ${#SUMMARY_ARGS[@]} == 0 )); then
  echo "At least one of RUN_FULL_REQUEST_PROFILES or RUN_STAGE_PROFILES must be 1." >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/summarize_npu_profile.py" \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${PROFILE_ROOT}/comparison"

echo "Operator profiling complete: ${PROFILE_ROOT}"
echo "Comparison report: ${PROFILE_ROOT}/comparison/operator_comparison.md"
