#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EVAL_MODE="${EVAL_MODE:-diffusion}"
case "${EVAL_MODE}" in
  ar|diffusion) ;;
  *)
    echo "EVAL_MODE must be 'ar' or 'diffusion'; got ${EVAL_MODE@Q}." >&2
    exit 2
    ;;
esac

DRY_RUN="${DRY_RUN:-0}"
SKIP_CANN="${SKIP_CANN:-0}"
CANN_ENV="${CANN_ENV:-}"
CANN_ROOT="${CANN_ROOT:-}"

can_load_ascend_backend() {
  python - <<'PY' >/dev/null 2>&1
import ctypes

ctypes.CDLL("libhccl.so")
import torch_npu  # noqa: F401
PY
}

source_cann_env() {
  local env_script="$1"
  local source_status

  echo "[CANN] Sourcing ${env_script}"
  set +e
  set +u
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
  # shellcheck disable=SC1090
  source "${env_script}"
  source_status=$?
  set -e
  set -u
  return "${source_status}"
}

add_cann_root_candidates() {
  local root="$1"
  local candidate

  [[ -z "${root}" ]] && return
  CANN_ENV_CANDIDATES+=(
    "${root}/set_env.sh"
    "${root}/ascend-toolkit/set_env.sh"
    "${root}/latest/set_env.sh"
  )
  shopt -s nullglob
  for candidate in \
    "${root}"/cann-*/set_env.sh \
    "${root}"/cann-*/ascend-toolkit/set_env.sh \
    "${root}"/ascend-toolkit/*/set_env.sh; do
    CANN_ENV_CANDIDATES+=("${candidate}")
  done
  shopt -u nullglob
}

configure_cann() {
  local candidate
  local -A seen_candidates=()
  CANN_ENV_CANDIDATES=()

  if can_load_ascend_backend; then
    echo "[CANN] Reusing the environment already available in this container."
    return
  fi

  if [[ -n "${CANN_ENV}" ]]; then
    if [[ ! -f "${CANN_ENV}" ]]; then
      echo "[CANN] CANN_ENV does not exist inside the container: ${CANN_ENV}" >&2
      return 1
    fi
    if ! source_cann_env "${CANN_ENV}" || ! can_load_ascend_backend; then
      echo "[CANN] ${CANN_ENV} did not make torch_npu and libhccl.so loadable." >&2
      return 1
    fi
    return
  fi

  add_cann_root_candidates "${CANN_ROOT}"
  add_cann_root_candidates "${ASCEND_HOME_PATH:-}"
  add_cann_root_candidates "${ASCEND_TOOLKIT_HOME:-}"
  add_cann_root_candidates "/usr/local/Ascend/cann"
  add_cann_root_candidates "/usr/local/Ascend/ascend-toolkit"
  add_cann_root_candidates "/opt/ascend/cann"

  shopt -s nullglob
  CANN_ENV_CANDIDATES+=(
    /home/*/Ascend*CANN*/set_env.sh
    /home/*/Ascend*CANN*/cann-*/set_env.sh
    /home/*/Ascend*CANN*/cann-*/ascend-toolkit/set_env.sh
    /share/*/Ascend*CANN*/set_env.sh
    /share/*/Ascend*CANN*/cann-*/set_env.sh
    /share/*/Ascend*CANN*/cann-*/ascend-toolkit/set_env.sh
    /home/*/activate_cann*.sh
    /share/*/activate_cann*.sh
  )
  shopt -u nullglob

  for candidate in "${CANN_ENV_CANDIDATES[@]}"; do
    [[ -f "${candidate}" ]] || continue
    [[ -n "${seen_candidates[${candidate}]:-}" ]] && continue
    seen_candidates["${candidate}"]=1
    if source_cann_env "${candidate}" && can_load_ascend_backend; then
      echo "[CANN] torch_npu and libhccl.so are available after sourcing ${candidate}."
      return
    fi
  done

  cat >&2 <<'EOF'
[CANN] Unable to load libhccl.so inside the container.
[CANN] Either mount the CANN installation into the container, or set one of:
[CANN]   CANN_ENV=/path/to/set_env.sh
[CANN]   CANN_ROOT=/path/to/cann-installation
[CANN] Paths must be paths visible inside the container, not host-only paths.
EOF
  return 1
}
if [[ "${DRY_RUN}" != "1" && "${SKIP_CANN}" != "1" ]]; then
  configure_cann
  echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
  echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
fi

export HF_HOME="${HF_HOME:-/home/f50058111/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-3600}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60300-60399}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61300-61399}"
export MASTER_PORT="${MASTER_PORT:-29503}"

if [[ "${EVAL_MODE}" == "ar" ]]; then
  DEFAULT_MODEL_PATH="/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_ar_cpt_baseline_0916/models/global_step_500"
  MODEL_BACKEND="hf"
  OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_ar_baseline}"
else
  DEFAULT_MODEL_PATH="/share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_0910/models/global_step_102500"
  MODEL_BACKEND="hf-diffusion"
  OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_cpt_diffusion_ar}"
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-True}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VERBOSITY="${VERBOSITY:-INFO}"
LIMIT="${LIMIT:-}"
MODEL_DEVICE="${MODEL_DEVICE:-}"
ENABLE_UNSAFE_CODE="${ENABLE_UNSAFE_CODE:-0}"

# The default suite is safe and fully generative for both backends. HumanEval
# remains available through TASKS=humaneval ENABLE_UNSAFE_CODE=1.
TASKS="${TASKS:-mmlu,gsm8k,boolq,bbh,arc_challenge,hellaswag,piqa,winogrande}"

IFS=',' read -r -a DEVICE_LIST <<<"${ASCEND_RT_VISIBLE_DEVICES}"
NUM_DEVICES=0
for device in "${DEVICE_LIST[@]}"; do
  [[ -n "${device//[[:space:]]/}" ]] && ((NUM_DEVICES += 1))
done
if (( NUM_DEVICES == 0 )); then
  echo "ASCEND_RT_VISIBLE_DEVICES does not contain any device IDs." >&2
  exit 2
fi

RUN_NAME="$(basename "${MODEL_PATH%/}")"
RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_OUTPUT}"
fi

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},local_files_only=${LOCAL_FILES_ONLY},trust_remote_code=${TRUST_REMOTE_CODE},dtype=${DTYPE}"

if [[ "${EVAL_MODE}" == "diffusion" ]]; then
  BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
  MASK_TOKEN_ID="${MASK_TOKEN_ID:-}"
  DIFFUSION_TEMPERATURE="${DIFFUSION_TEMPERATURE:-0.0}"
  THRESHOLD="${THRESHOLD:-0.0}"
  TOP_K="${TOP_K:-0}"
  TOP_P="${TOP_P:-1.0}"
  MC_NUM="${MC_NUM:-128}"
  MC_BATCH_SIZE="${MC_BATCH_SIZE:-16}"
  MODEL_ARGS+=",block_length=${BLOCK_LENGTH},diffusion_temperature=${DIFFUSION_TEMPERATURE},threshold=${THRESHOLD},top_k=${TOP_K},top_p=${TOP_P},mc_num=${MC_NUM},mc_batch_size=${MC_BATCH_SIZE},seed=${SEED}"
  [[ -n "${DIFFUSION_STEPS}" ]] && MODEL_ARGS+=",diffusion_steps=${DIFFUSION_STEPS}"
  [[ -n "${MASK_TOKEN_ID}" ]] && MODEL_ARGS+=",mask_token_id=${MASK_TOKEN_ID}"
fi

if [[ -z "${MODEL_DEVICE}" && "${NUM_DEVICES}" == "1" ]]; then
  MODEL_DEVICE="npu:0"
fi
[[ -n "${MODEL_DEVICE}" ]] && MODEL_ARGS+=",device=${MODEL_DEVICE}"

ACCELERATE_ARGS=(--num_processes "${NUM_DEVICES}" --main_process_port "${MASTER_PORT}")
if (( NUM_DEVICES > 1 )); then
  ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi

LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS=(--limit "${LIMIT}")

run_task() {
  local requested_task="$1"
  local actual_task
  local output_name
  local fewshot
  local unsafe=0

  case "${requested_task}" in
    mmlu|mmlu_generative)
      actual_task=mmlu_generative; output_name=mmlu; fewshot=5 ;;
    gsm8k)
      actual_task=gsm8k; output_name=gsm8k; fewshot=5 ;;
    boolq|boolq-seq2seq)
      actual_task=boolq-seq2seq; output_name=boolq; fewshot=5 ;;
    bbh)
      actual_task=bbh; output_name=bbh; fewshot=3 ;;
    humaneval)
      actual_task=humaneval; output_name=humaneval; fewshot=0; unsafe=1 ;;
    arc_challenge|arc_challenge_generative)
      actual_task=arc_challenge_generative; output_name=arc_challenge; fewshot=0 ;;
    hellaswag|hellaswag_generative)
      actual_task=hellaswag_generative; output_name=hellaswag; fewshot=0 ;;
    piqa|piqa_generative)
      actual_task=piqa_generative; output_name=piqa; fewshot=0 ;;
    winogrande|winogrande_generative)
      actual_task=winogrande_generative; output_name=winogrande; fewshot=0 ;;
    generative_choice)
      actual_task=generative_choice; output_name=generative_choice; fewshot=0 ;;
    *)
      echo "Unsupported suite task: ${requested_task}" >&2
      echo "Supported aliases: mmlu,gsm8k,boolq,bbh,humaneval,arc_challenge,hellaswag,piqa,winogrande,generative_choice" >&2
      exit 2
      ;;
  esac

  EXTRA_ARGS=()
  if (( unsafe )); then
    if [[ "${ENABLE_UNSAFE_CODE}" != "1" ]]; then
      echo "HumanEval executes generated code. Set ENABLE_UNSAFE_CODE=1 to opt in." >&2
      exit 2
    fi
    export HF_ALLOW_CODE_EVAL=1
    EXTRA_ARGS=(--confirm_run_unsafe_code)
  fi

  command=(
    accelerate launch "${ACCELERATE_ARGS[@]}"
    -m lm_eval
    --model "${MODEL_BACKEND}"
    --model_args "${MODEL_ARGS}"
    --tasks "${actual_task}"
    --num_fewshot "${fewshot}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --verbosity "${VERBOSITY}"
    --log_samples
    --output_path "${RUN_OUTPUT}/${output_name}"
    "${LIMIT_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
  )

  echo "==== mode=${EVAL_MODE} task=${actual_task} fewshot=${fewshot} model=${MODEL_PATH} ===="
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

echo "Evaluation mode : ${EVAL_MODE}"
echo "Model backend   : ${MODEL_BACKEND}"
echo "Model path      : ${MODEL_PATH}"
echo "Tasks           : ${TASKS}"
echo "Devices         : ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Output          : ${RUN_OUTPUT}"

normalized_tasks="${TASKS//,/ }"
for requested_task in ${normalized_tasks}; do
  [[ -n "${requested_task}" ]] && run_task "${requested_task}"
done

echo "${EVAL_MODE} generation evaluation finished. Results: ${RUN_OUTPUT}"
