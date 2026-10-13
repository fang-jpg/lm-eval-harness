#!/usr/bin/env bash

# ==============================
# 1. Load CANN
# ==============================

set +e
set +u

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

source /share/f50058111/activate_cann_910b_9.1.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "===== CANN environment loaded ====="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"

python - <<'PY'
import torch
import torch_npu
print("torch =", torch.__version__)
print("NPU available =", torch.npu.is_available())
PY

# ==============================
# 2. HuggingFace offline cache
# ==============================

export HF_HOME=/share/f50058111/huggingface_cache
export HF_HUB_CACHE=/share/f50058111/huggingface_cache/hub
export HF_DATASETS_CACHE=/share/f50058111/huggingface_cache/datasets

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
# BBH diffusion requests have highly variable generation times across ranks.
# Fast ranks may reach the final collective more than one hour before slow
# ranks.  CANN 9.1 allows HCCL_CONNECT_TIMEOUT up to 7200 seconds, so use that
# ceiling while retaining environment-variable overrides.
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-7200}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60300-60399}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61300-61399}"
export MASTER_PORT="${MASTER_PORT:-29503}"

MODEL_PATH="${MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_0910/models/global_step_102500/}"
TASKS="${TASKS:-mmlu,gsm8k,humaneval,boolq,bbh}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
MASK_TOKEN_ID="${MASK_TOKEN_ID:-}"
DIFFUSION_TEMPERATURE="${DIFFUSION_TEMPERATURE:-0.0}"
# THRESHOLD=0 keeps scheduled quota decoding. A value in (0, 1] accepts every
# token above the threshold, then supplements with the highest-confidence
# remaining tokens as needed to satisfy the current scheduled quota.
THRESHOLD="${THRESHOLD:-0.8}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-1.0}"
MC_NUM="${MC_NUM:-128}"
MC_BATCH_SIZE="${MC_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_diffusion}"
MODEL_DEVICE="${MODEL_DEVICE:-}"
# Every completed model response is committed to a rank-local SQLite file.
# If the long-running inference succeeds but the final HCCL gather fails, the
# next attempt reads these files and only recomputes genuinely missing samples.
RESPONSE_CACHE_ROOT="${RESPONSE_CACHE_ROOT:-${OUTPUT_ROOT}/response_cache}"
# Bump this whenever decoding semantics change so an otherwise identical run
# cannot reuse responses produced by an older decoder implementation.
DECODER_CACHE_VERSION="${DECODER_CACHE_VERSION:-threshold-scheduled-quota-v2}"
# Total attempts, including the first one.  A retry only computes responses
# still missing from each rank-local cache, so an extra attempt is inexpensive
# and covers a slow rank that needs more than one timeout window to catch up.
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"

NUM_DEVICES="$(awk -F',' '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
RUN_NAME="$(basename "${MODEL_PATH%/}")"
RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
METRICS_ROOT="${METRICS_ROOT:-${RUN_OUTPUT}/metrics}"
mkdir -p "${RUN_OUTPUT}"

if ! [[ "${MAX_TASK_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TASK_ATTEMPTS must be a positive integer, got: ${MAX_TASK_ATTEMPTS}" >&2
  exit 2
fi

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True,dtype=bfloat16,block_length=${BLOCK_LENGTH},diffusion_temperature=${DIFFUSION_TEMPERATURE},threshold=${THRESHOLD},top_k=${TOP_K},top_p=${TOP_P},mc_num=${MC_NUM},mc_batch_size=${MC_BATCH_SIZE},seed=${SEED}"
if [[ -n "${DIFFUSION_STEPS}" ]]; then
  MODEL_ARGS+=",diffusion_steps=${DIFFUSION_STEPS}"
fi
if [[ -n "${MASK_TOKEN_ID}" ]]; then
  MODEL_ARGS+=",mask_token_id=${MASK_TOKEN_ID}"
fi
# HFLM's historical default is "cuda". Accelerate overrides it for a
# multi-process NPU launch, but a one-process NPU smoke test needs an explicit
# local device.
if [[ -z "${MODEL_DEVICE}" && "${NUM_DEVICES}" == "1" ]]; then
  MODEL_DEVICE="npu:0"
fi
if [[ -n "${MODEL_DEVICE}" ]]; then
  MODEL_ARGS+=",device=${MODEL_DEVICE}"
fi

# CachingLM's request key contains the prompt and generation kwargs, but not
# the model identity or hf-diffusion-specific decoding settings.  Namespace
# the cache with those arguments so changing a checkpoint/setting cannot reuse
# stale responses accidentally.
CACHE_CONFIG_ID="$(python -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16])' "${DECODER_CACHE_VERSION}|${MODEL_ARGS}")"

ACCELERATE_ARGS=(--num_processes "${NUM_DEVICES}" --main_process_port "${MASTER_PORT}")
if (( NUM_DEVICES > 1 )); then
  ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

task_selected() {
  [[ ",${TASKS}," == *",$1,"* ]]
}

run_task() {
  local task="$1"
  local fewshot="$2"
  shift 2

  local log_dir="${RUN_OUTPUT}/logs"
  mkdir -p "${log_dir}"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"

  local log_file="${log_dir}/${task}_${timestamp}.log"
  local summary_log="${log_dir}/${task}_${timestamp}_summary.log"
  local metrics_run_dir="${METRICS_ROOT}/${task}_${timestamp}"
  local cache_prefix="${RESPONSE_CACHE_ROOT}/${RUN_NAME}/${CACHE_CONFIG_ID}/${task}/responses"
  mkdir -p "$(dirname "${cache_prefix}")"
  mkdir -p "${metrics_run_dir}"

  echo "===================================================="
  echo "Task       : ${task}"
  echo "Model      : ${MODEL_PATH}"
  echo "Fewshot    : ${fewshot}"
  echo "Block size : ${BLOCK_LENGTH}"
  echo "Diff steps : ${DIFFUSION_STEPS:-auto}"
  echo "Threshold  : ${THRESHOLD}"
  echo "Log file   : ${log_file}"
  echo "Summary log: ${summary_log}"
  echo "Metrics dir: ${metrics_run_dir}"
  echo "Cache ID   : ${CACHE_CONFIG_ID}"
  echo "Decoder ver: ${DECODER_CACHE_VERSION}"
  echo "Cache files: ${cache_prefix}_rank<N>.db"
  echo "HCCL wait  : connect=${HCCL_CONNECT_TIMEOUT}s, exec=${HCCL_EXEC_TIMEOUT}s"
  echo "Start time : $(date '+%F %T')"
  echo "===================================================="

  # Save stdout/stderr while preserving accelerate's exit status.  The
  # response cache is rank-dependent, so retries must keep the same device
  # count and ordering.  lm-eval adds the `_rank<N>.db` suffix itself.
  local attempt
  local exit_code=1
  for ((attempt = 1; attempt <= MAX_TASK_ATTEMPTS; attempt++)); do
    local attempt_metrics_dir="${metrics_run_dir}/attempt_${attempt}"
    mkdir -p "${attempt_metrics_dir}"
    echo "[ATTEMPT ${attempt}/${MAX_TASK_ATTEMPTS}] ${task}" \
      | tee -a "${log_file}"

    if HF_DIFFUSION_METRICS_DIR="${attempt_metrics_dir}" \
      accelerate launch "${ACCELERATE_ARGS[@]}" \
      -m lm_eval \
      --model hf-diffusion \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --num_fewshot "${fewshot}" \
      --batch_size 1 \
      --verbosity INFO \
      --log_samples \
      --use_cache "${cache_prefix}" \
      --cache_requests \
      --output_path "${RUN_OUTPUT}/${task}" \
      "${LIMIT_ARGS[@]}" \
      "$@" \
      2>&1 | tee -a "${log_file}"; then

      echo "[SUCCESS] ${task} completed at $(date '+%F %T')" \
        | tee -a "${log_file}"
      if python "${SCRIPT_DIR}/scripts/summarize_diffusion_metrics.py" \
        --metrics-dir "${metrics_run_dir}" \
        --results-dir "${RUN_OUTPUT}/${task}" \
        --output "${summary_log}" \
        --task "${task}" \
        --model "${MODEL_PATH}" \
        --source-log "${log_file}"; then
        echo "[SUMMARY] Accuracy and diffusion performance: ${summary_log}" \
          | tee -a "${log_file}"
      else
        echo "[ERROR] Failed to create diffusion summary log." \
          | tee -a "${log_file}"
        return 1
      fi
      return 0
    else
      exit_code="${PIPESTATUS[0]}"
      echo "[ERROR] ${task} attempt ${attempt} failed, exit code=${exit_code}" \
        | tee -a "${log_file}"
      echo "[RECOVERY] Completed responses remain in ${cache_prefix}_rank<N>.db" \
        | tee -a "${log_file}"
    fi

    if (( attempt < MAX_TASK_ATTEMPTS )); then
      echo "[RETRY] Restarting all ranks; cached responses will skip inference." \
        | tee -a "${log_file}"
    fi
  done

  return "${exit_code}"
}

# MMLU and BoolQ must exercise the same block_diff generation path as the
# free-form tasks.  The stock `mmlu` and `boolq` aliases are multiple-choice
# likelihood tasks and are not valid for this checkpoint's diffusion mode.
# task_selected mmlu && run_task mmlu_generative 5

# All five tasks now call block-diffusion generate_until.
# task_selected gsm8k && run_task gsm8k 5
# task_selected humaneval && run_task humaneval 0 --confirm_run_unsafe_code
# task_selected boolq && run_task boolq-seq2seq 5
task_selected bbh && run_task bbh 3

task_selected mmlu && run_task mmlu_generative 5
task_selected boolq && run_task boolq-seq2seq 5

echo "Diffusion evaluation finished. Results: ${RUN_OUTPUT}"
