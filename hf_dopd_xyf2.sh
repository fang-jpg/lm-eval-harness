#!/usr/bin/env bash

# d-OPD paper Appendix C.4 evaluation for one 8-card Ascend node.
# Decoder semantics are intentionally fixed; only paths, device/runtime knobs,
# selected tasks, and LIMIT (for a non-comparable smoke test) are configurable.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DOPD_EVAL_SCRIPT_VERSION="standalone-hf-dopd-gsm8k-5shot-greedy-v2"
echo "[d-OPD] Evaluation script version: ${DOPD_EVAL_SCRIPT_VERSION}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export CANN_ROOT="/share/f50058111/Ascend910B_CANN_9.1.0/cann-9.1.0"
export CANN_ENV="${CANN_ROOT}/set_env.sh"
export CANN_OPP_PATH="${CANN_ROOT}/opp"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "[CANN] CANN_ENV does not exist: ${CANN_ENV}" >&2
  exit 2
fi
if [[ ! -d "${CANN_ROOT}" ]]; then
  echo "[CANN] CANN_ROOT does not exist: ${CANN_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CANN_OPP_PATH}" ]]; then
  echo "[CANN] CANN_OPP_PATH does not exist: ${CANN_OPP_PATH}" >&2
  exit 2
fi

echo "[CANN] Loading ${CANN_ENV}"
set +e
set +u
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
# shellcheck disable=SC1090
source "${CANN_ENV}"
SOURCED_CANN_ROOT="${ASCEND_HOME_PATH:-}"
set -euo pipefail

# The package was copied from another server.  If set_env.sh still contains
# paths from that machine, rewrite the exported values in memory only.
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
export ASCEND_OPP_PATH="${CANN_OPP_PATH%/}"

if ! KERNEL_INDEX="$("${PYTHON_BIN}" \
  "${SCRIPT_DIR}/scripts/find_working_cann.py" \
  --check-opp "${ASCEND_OPP_PATH}")"; then
  echo "[CANN] The configured 910B operator package is incomplete or incompatible." >&2
  exit 2
fi

echo "===== CANN environment ====="
echo "CANN_ENV=${CANN_ENV}"
echo "CANN_ROOT=${CANN_ROOT}"
echo "CANN_OPP_PATH=${CANN_OPP_PATH}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
echo "910B kernel index=${KERNEL_INDEX}"

# Verify the same Cast operation used by the d-OPD attention-mask path
# before checkpoint conversion and eight-rank model loading begin.
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

MINDSPEED_ROOT="${MINDSPEED_ROOT:-${SCRIPT_DIR}/../MindSpeed-LLM}"

DCP_MODEL_PATH="${DCP_MODEL_PATH:-/share/f50058111/model_train/dopd/qwen3_1.7b_dopd_paper_1node_8card_1019/checkpoints/global_step_150}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${DCP_MODEL_PATH%/}/hf_ckpt}"
DOPD_CONFIG="${DOPD_CONFIG:-$(dirname "$(dirname "${DCP_MODEL_PATH%/}")")/dopd_config.json}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/f50058111/model_weights/Qwen3-1.7B/Qwen3-1.7B}"

export HF_HOME="${HF_HOME:-/share/f50058111/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-7200}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60300-60399}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61300-61399}"
export MASTER_PORT="${MASTER_PORT:-29503}"

# User-facing task aliases. These six are exactly the paper benchmark set.
TASKS="${TASKS:-mmlu,mmlu_pro,gpqa_diamond,gsm8k,math500,aime25}"
LIMIT="${LIMIT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_dopd_1022_150}"
MODEL_DEVICE="${MODEL_DEVICE:-}"
SEED="${SEED:-42}"
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"

# Appendix C.4 constants. Do not turn these into ordinary overrides: a run
# with different values is an ablation and must not be labelled paper-aligned.
BLOCK_LENGTH=4
DENOISING_STEPS_PER_BLOCK=4
MASK_TOKEN_ID=151669
DIFFUSION_TEMPERATURE=0.0
TOP_K=1
TOP_P=1.0
MAX_NEW_TOKENS=4096

if [[ ! -d "${DCP_MODEL_PATH}" || ! -f "${DCP_MODEL_PATH}/.metadata" ]]; then
  echo "无效的 d-OPD DCP checkpoint: ${DCP_MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MINDSPEED_ROOT}/mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py" ]]; then
  echo "找不到 MindSpeed-LLM: ${MINDSPEED_ROOT}" >&2
  exit 2
fi

if [[ -z "${BASE_MODEL_PATH}" && -f "${DOPD_CONFIG}" ]]; then
  BASE_MODEL_PATH="$("${PYTHON_BIN}" -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("model", {}).get("model_name_or_path", ""))' "${DOPD_CONFIG}")"
fi
if [[ -z "${BASE_MODEL_PATH}" || ! -f "${BASE_MODEL_PATH}/config.json" ]]; then
  echo "找不到训练所用的 Qwen3-1.7B-Base；请设置 BASE_MODEL_PATH。" >&2
  echo "当前解析结果: ${BASE_MODEL_PATH:-<empty>}" >&2
  exit 2
fi

if [[ ! -f "${HF_MODEL_PATH}/model.safetensors" && ! -f "${HF_MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "===== 将 d-OPD DCP 转换为 Hugging Face checkpoint ====="
  "${PYTHON_BIN}" "${MINDSPEED_ROOT}/mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py" \
    --load-dir "${DCP_MODEL_PATH}" \
    --save-dir "${HF_MODEL_PATH}"
else
  echo "===== 复用已有 Hugging Face checkpoint: ${HF_MODEL_PATH} ====="
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_dopd_hf_checkpoint.py" \
  --source-model-dir "${BASE_MODEL_PATH}" \
  --output-dir "${HF_MODEL_PATH}" \
  --mask-token-id "${MASK_TOKEN_ID}"

NUM_DEVICES="$(awk -F',' '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
if ! [[ "${NUM_DEVICES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ASCEND_RT_VISIBLE_DEVICES 必须至少包含一张 NPU。" >&2
  exit 2
fi
RUN_NAME="$(basename "${DCP_MODEL_PATH%/}")"
RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
METRICS_ROOT="${METRICS_ROOT:-${RUN_OUTPUT}/metrics}"
RESPONSE_CACHE_ROOT="${RESPONSE_CACHE_ROOT:-${OUTPUT_ROOT}/response_cache}"
DECODER_CACHE_VERSION="standalone-hf-dopd-gsm8k-5shot-greedy-v2"
mkdir -p "${RUN_OUTPUT}"

if ! [[ "${MAX_TASK_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TASK_ATTEMPTS 必须是正整数，当前为: ${MAX_TASK_ATTEMPTS}" >&2
  exit 2
fi
if [[ -n "${LIMIT}" ]]; then
  echo "[警告] LIMIT=${LIMIT} 仅用于冒烟测试，不可与论文结果比较。"
fi

MODEL_ARGS="pretrained=${HF_MODEL_PATH},tokenizer=${HF_MODEL_PATH},local_files_only=True,trust_remote_code=True,dtype=bfloat16,block_length=${BLOCK_LENGTH},denoising_steps_per_block=${DENOISING_STEPS_PER_BLOCK},diffusion_temperature=${DIFFUSION_TEMPERATURE},top_k=${TOP_K},top_p=${TOP_P},mask_token_id=${MASK_TOKEN_ID},seed=${SEED},enable_thinking=false"
if [[ -z "${MODEL_DEVICE}" && "${NUM_DEVICES}" == "1" ]]; then
  MODEL_DEVICE="npu:0"
fi
if [[ -n "${MODEL_DEVICE}" ]]; then
  MODEL_ARGS+=",device=${MODEL_DEVICE}"
fi

CACHE_CONFIG_ID="$("${PYTHON_BIN}" -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16])' "${DECODER_CACHE_VERSION}|${MODEL_ARGS}|max_new_tokens=${MAX_NEW_TOKENS}|chat_template=true")"
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

if ! task_selected gsm8k; then
  echo "当前脚本只启用了 GSM8K；请让 TASKS 包含 gsm8k。" >&2
  exit 2
fi

# Fail once before spawning eight ranks if the repaired HF assets, installed
# packages, or the currently enabled task registration are invalid.
"${PYTHON_BIN}" - "${HF_MODEL_PATH}" "${MASK_TOKEN_ID}" \
  gsm8k_cot <<'PY'
import sys

import accelerate
import transformers
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer

from lm_eval.tasks import TaskManager


model_path = sys.argv[1]
mask_token_id = int(sys.argv[2])
tasks = sys.argv[3:]

if Version(transformers.__version__) < Version("5.2.0"):
    raise SystemExit(
        "[PREFLIGHT] hf-dopd requires transformers>=5.2.0; "
        f"found {transformers.__version__}."
    )

config = AutoConfig.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)
if config.model_type != "qwen3":
    raise SystemExit(
        f"[PREFLIGHT] Expected exported model_type='qwen3', got {config.model_type!r}."
    )
if not 0 <= mask_token_id < int(config.vocab_size):
    raise SystemExit(
        f"[PREFLIGHT] mask_token_id={mask_token_id} is outside "
        f"vocab_size={config.vocab_size}."
    )
AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)

task_manager = TaskManager()
missing = [task for task in tasks if task not in task_manager.all_tasks]
if missing:
    raise SystemExit(f"[PREFLIGHT] Unregistered lm-eval tasks: {missing}")

print(
    "[PREFLIGHT] HF config/tokenizer and lm-eval task passed "
    f"(transformers={transformers.__version__}, accelerate={accelerate.__version__})."
)
PY

run_task() {
  if (( $# < 3 )); then
    echo "run_task 至少需要三个参数: <alias> <task> <expected_examples> [fewshot]" >&2
    return 2
  fi
  local task_alias="$1"
  local task_name="$2"
  local expected_examples="$3"
  local fewshot="${4:-0}"
  if (( $# >= 4 )); then
    shift 4
  else
    shift 3
  fi
  if ! [[ "${fewshot}" =~ ^[0-9]+$ ]]; then
    echo "fewshot 必须是非负整数，当前为: ${fewshot}" >&2
    return 2
  fi
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_dir="${RUN_OUTPUT}/logs"
  local log_file="${log_dir}/${task_alias}_${timestamp}.log"
  local summary_log="${log_dir}/${task_alias}_${timestamp}_summary.log"
  local metrics_run_dir="${METRICS_ROOT}/${task_alias}_${timestamp}"
  local cache_prefix="${RESPONSE_CACHE_ROOT}/${RUN_NAME}/${CACHE_CONFIG_ID}/${task_alias}/responses"
  mkdir -p "${log_dir}" "${metrics_run_dir}" "$(dirname "${cache_prefix}")"

  echo "===================================================="
  echo "论文任务      : ${task_alias} -> ${task_name}"
  echo "完整样本数    : ${expected_examples}"
  echo "Few-shot      : ${fewshot}"
  echo "Checkpoint    : ${DCP_MODEL_PATH}"
  echo "Block/steps   : ${BLOCK_LENGTH}/${DENOISING_STEPS_PER_BLOCK}"
  echo "起始状态      : response-aligned 全 MASK block"
  echo "重掩码        : low_confidence_static"
  echo "采样          : temperature=${DIFFUSION_TEMPERATURE}, top-p=${TOP_P}, top-k=${TOP_K},do_sample=false"
  echo "Prompt        : Qwen3 chat template, thinking=false"
  echo "生成          : ${fewshot}-shot, 1 completion, max ${MAX_NEW_TOKENS} tokens"
  echo "日志          : ${log_file}"
  echo "===================================================="

  local attempt
  local exit_code=1
  for ((attempt = 1; attempt <= MAX_TASK_ATTEMPTS; attempt++)); do
    local attempt_metrics_dir="${metrics_run_dir}/attempt_${attempt}"
    mkdir -p "${attempt_metrics_dir}"
    echo "[ATTEMPT ${attempt}/${MAX_TASK_ATTEMPTS}] ${task_alias}" | tee -a "${log_file}"
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_sqlite_cache.py" \
      --cache-prefix "${cache_prefix}" \
      --quarantine-root "${cache_prefix}_corrupt" \
      2>&1 | tee -a "${log_file}"; then
      echo "[ERROR] 无法检查或隔离损坏的响应缓存。" | tee -a "${log_file}"
      return 1
    fi
    if HF_DOPD_METRICS_DIR="${attempt_metrics_dir}" \
      "${PYTHON_BIN}" -m accelerate.commands.launch "${ACCELERATE_ARGS[@]}" \
      -m lm_eval \
      --model hf-dopd \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task_name}" \
      --num_fewshot "${fewshot}" \
      --batch_size 1 \
      --apply_chat_template \
      --gen_kwargs "max_gen_toks=${MAX_NEW_TOKENS},temperature=${DIFFUSION_TEMPERATURE},top_p=${TOP_P},do_sample=False" \
      --verbosity INFO \
      --log_samples \
      --use_cache "${cache_prefix}" \
      --cache_requests \
      --output_path "${RUN_OUTPUT}/${task_alias}" \
      "${LIMIT_ARGS[@]}" \
      "$@" 2>&1 | tee -a "${log_file}"; then
      "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/summarize_dopd_metrics.py" \
        --metrics-dir "${metrics_run_dir}" \
        --results-dir "${RUN_OUTPUT}/${task_alias}" \
        --output "${summary_log}" \
        --task "${task_name}" \
        --model "${HF_MODEL_PATH}" \
        --source-log "${log_file}"
      return 0
    else
      exit_code="${PIPESTATUS[0]}"
      echo "[ERROR] ${task_alias} 第 ${attempt} 次运行失败，exit=${exit_code}" | tee -a "${log_file}"
    fi
  done
  return "${exit_code}"
}

echo "===== d-OPD 论文附录 C.4 对齐评测 ====="
# task_selected mmlu && run_task mmlu mmlu_generative 14042
# task_selected mmlu_pro && run_task mmlu_pro leaderboard_mmlu_pro 12032
# task_selected gpqa_diamond && run_task gpqa_diamond gpqa_diamond_cot_zeroshot 198
# task_selected gsm8k && run_task gsm8k gsm8k_cot_zeroshot 1319
# task_selected math500 && run_task math500 hendrycks_math500 500
# task_selected aime25 && run_task aime25 aime25 30
# task_selected gsm8k && run_task gsm8k gsm8k_cot_zeroshot 1319 5
task_selected gsm8k && run_task gsm8k gsm8k_cot 1319 5

echo "评测完成: ${RUN_OUTPUT}"
echo "注意：论文还在 step 50,100,...,450,483 中按六项平均分选同一个最佳 checkpoint。"
