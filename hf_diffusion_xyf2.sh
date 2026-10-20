#!/usr/bin/env bash

# ==============================
# 1. Load CANN
# ==============================

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# The CANN operator package on this machine was copied from another server, so
# never guess its old absolute path.  CANN_ENV is optional; CANN_OPP_PATH must
# be the real `opp` directory visible on the current server/container.
CANN_ENV="${CANN_ENV:-}"
CANN_OPP_PATH="${CANN_OPP_PATH:-}"
CANN_ROOT="${CANN_ROOT:-}"
if [[ -z "${CANN_ROOT}" && -n "${CANN_OPP_PATH}" ]]; then
  CANN_ROOT="$(dirname -- "${CANN_OPP_PATH%/}")"
fi

if [[ -n "${CANN_ENV}" ]]; then
  if [[ ! -f "${CANN_ENV}" ]]; then
    echo "[CANN] CANN_ENV does not exist: ${CANN_ENV}" >&2
    exit 2
  fi
  echo "[CANN] Loading ${CANN_ENV}"
  # Vendor activation scripts can reference unset variables or return nonzero
  # from optional probes.  The real NPU Cast check below is authoritative.
  set +e
  set +u
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
  # shellcheck disable=SC1090
  source "${CANN_ENV}"
else
  echo "[CANN] Reusing the runtime environment used by hf_xyf2.sh."
fi
SOURCED_CANN_ROOT="${ASCEND_HOME_PATH:-}"
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# A copied set_env.sh can export absolute paths from its source server. Rebase
# all standard path variables and ASCEND_* variables onto the actual local
# CANN root before importing torch_npu. The copied files are not modified.
if [[ -n "${CANN_ROOT}" ]]; then
  if [[ ! -d "${CANN_ROOT}" ]]; then
    echo "[CANN] CANN_ROOT does not exist on this server: ${CANN_ROOT}" >&2
    exit 2
  fi
  if [[ -n "${SOURCED_CANN_ROOT}" && "${SOURCED_CANN_ROOT}" != "${CANN_ROOT}" ]]; then
    rebase_cann_variable() {
      local variable_name="$1"
      local old_value="${!variable_name:-}"
      if [[ "${old_value}" == *"${SOURCED_CANN_ROOT}"* ]]; then
        printf -v "${variable_name}" '%s' \
          "${old_value//${SOURCED_CANN_ROOT}/${CANN_ROOT}}"
        export "${variable_name}"
      fi
    }
    while IFS= read -r variable_name; do
      rebase_cann_variable "${variable_name}"
    done < <(compgen -e)
    echo "[CANN] Relocated environment: ${SOURCED_CANN_ROOT} -> ${CANN_ROOT}"
  fi
  export ASCEND_HOME_PATH="${CANN_ROOT}"
  export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
fi

if [[ -n "${CANN_OPP_PATH}" ]]; then
  export ASCEND_OPP_PATH="${CANN_OPP_PATH%/}"
fi

if [[ -z "${ASCEND_OPP_PATH:-}" ]]; then
  cat >&2 <<'EOF'
[CANN] ASCEND_OPP_PATH is empty.
[CANN] Locate the copied 910B operator package, then rerun with:
[CANN]   CANN_OPP_PATH=/actual/path/to/opp bash hf_diffusion_xyf2.sh
EOF
  exit 2
fi

if [[ ! -d "${ASCEND_OPP_PATH}" ]]; then
  echo "[CANN] ASCEND_OPP_PATH does not exist on this server: ${ASCEND_OPP_PATH}" >&2
  echo "[CANN] Set CANN_OPP_PATH to the copied package's real opp directory." >&2
  exit 2
fi

# Older kernel packages have one JSON per op; newer packages use a full kernel
# index.  The checker verifies that a full index actually mentions Cast; an
# unrelated ops_oam index is not sufficient.
if ! KERNEL_INDEX="$("${PYTHON_BIN}" \
  "${SCRIPT_DIR}/scripts/find_working_cann.py" \
  --check-opp "${ASCEND_OPP_PATH}")"; then
  echo "[CANN] This is not the correct opp root, or its 910B kernels package is incomplete." >&2
  echo "[CANN] Run: ${PYTHON_BIN} scripts/find_working_cann.py" >&2
  exit 2
fi

echo "===== CANN environment ====="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
echo "910B kernel index=${KERNEL_INDEX}"

# ==============================
# 2. HuggingFace offline cache
# ==============================

export HF_HOME="${HF_HOME:-/mnt/pipeline-data/f50058111/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

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

MODEL_PATH="${MODEL_PATH:-/mnt/pipeline-data/f50058111/model_train/ckpt/models/global_step_102500/}"
TASKS="${TASKS:-mmlu,gsm8k,humaneval,boolq,bbh}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
MASK_TOKEN_ID="${MASK_TOKEN_ID:-}"
DIFFUSION_TEMPERATURE="${DIFFUSION_TEMPERATURE:-0.0}"
# THRESHOLD=0 keeps scheduled quota decoding. A value in (0, 1] accepts every
# token above the threshold, then supplements with the highest-confidence
# remaining tokens as needed to satisfy the current scheduled quota.
THRESHOLD="${THRESHOLD:-0.0}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-1.0}"
MC_NUM="${MC_NUM:-128}"
MC_BATCH_SIZE="${MC_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_diffusion_1020}"
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
if ! [[ "${NUM_DEVICES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ASCEND_RT_VISIBLE_DEVICES must contain at least one device." >&2
  exit 2
fi

# Importing torch_npu is not enough to prove that the operator package is
# configured.  This is the exact dtype conversion that previously failed in
# hf_diffusion.py at attention_mask.bool(); run it before loading eight models.
"${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu  # noqa: F401

if not torch.npu.is_available():
    raise SystemExit("[CANN] NPU is unavailable; check CANN_ENV and torch_npu.")

try:
    torch.npu.set_device(0)
    attention_mask = torch.tensor([[1, 0]], dtype=torch.int64, device="npu:0")
    attention_mask.bool()
    torch.npu.synchronize()
except Exception as exc:
    raise SystemExit(
        "[CANN] NPU Cast self-check failed although a 910B kernel index exists. "
        "Check CANN_OPP_PATH, file permissions, and the CANN/torch_npu version match.\n"
        f"Original error: {exc}"
    ) from exc

print(f"[CANN] NPU Cast self-check passed (torch={torch.__version__}).")
PY

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
CACHE_CONFIG_ID="$("${PYTHON_BIN}" -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16])' "${DECODER_CACHE_VERSION}|${MODEL_ARGS}")"

ACCELERATE_ARGS=(--num_processes "${NUM_DEVICES}" --num_machines 1 --mixed_precision no --dynamo_backend no --main_process_port "${MASTER_PORT}")
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

    # A killed multi-rank run can leave an individual SQLite response cache
    # malformed.  Keep healthy rank caches, but move broken databases and
    # their WAL/SHM files aside before every attempt so retries can recover.
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_sqlite_cache.py" \
      --cache-prefix "${cache_prefix}" \
      --quarantine-root "${cache_prefix}_corrupt" \
      2>&1 | tee -a "${log_file}"; then
      echo "[ERROR] Unable to validate/quarantine response caches." \
        | tee -a "${log_file}"
      return 1
    fi

    if HF_DIFFUSION_METRICS_DIR="${attempt_metrics_dir}" \
      "${PYTHON_BIN}" -m accelerate.commands.launch "${ACCELERATE_ARGS[@]}" \
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
      if "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/summarize_diffusion_metrics.py" \
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
task_selected gsm8k && run_task gsm8k 5
# task_selected humaneval && run_task humaneval 0 --confirm_run_unsafe_code
# task_selected boolq && run_task boolq-seq2seq 5
# task_selected bbh && run_task bbh 3

# task_selected mmlu && run_task mmlu_generative 5
# task_selected boolq && run_task boolq-seq2seq 5

echo "Diffusion evaluation finished. Results: ${RUN_OUTPUT}"
