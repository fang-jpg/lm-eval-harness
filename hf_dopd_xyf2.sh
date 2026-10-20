#!/usr/bin/env bash
set +e
set +u

# d-OPD paper Appendix C.4 evaluation for one 8-card Ascend node.
# Decoder semantics are intentionally fixed; only paths, device/runtime knobs,
# selected tasks, and LIMIT (for a non-comparable smoke test) are configurable.

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /share/f50058111/activate_cann_910b_9.1.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_ROOT="${MINDSPEED_ROOT:-${SCRIPT_DIR}/../MindSpeed-LLM}"

DCP_MODEL_PATH="${DCP_MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_dopd_1010/checkpoints/global_step_483_final}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${DCP_MODEL_PATH%/}/hf_ckpt}"
DOPD_CONFIG="${DOPD_CONFIG:-$(dirname "$(dirname "${DCP_MODEL_PATH%/}")")/dopd_config.json}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_dopd_paper_c4}"
MODEL_DEVICE="${MODEL_DEVICE:-}"
MC_NUM="${MC_NUM:-128}"
MC_BATCH_SIZE="${MC_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"

# Appendix C.4 constants. Do not turn these into ordinary overrides: a run
# with different values is an ablation and must not be labelled paper-aligned.
BLOCK_LENGTH=4
DENOISING_STEPS_PER_BLOCK=4
MASK_TOKEN_ID=151669
DIFFUSION_TEMPERATURE=1.0
THRESHOLD=0.0
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
  BASE_MODEL_PATH="$(python -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("model", {}).get("model_name_or_path", ""))' "${DOPD_CONFIG}")"
fi
if [[ -z "${BASE_MODEL_PATH}" || ! -f "${BASE_MODEL_PATH}/config.json" ]]; then
  echo "找不到训练所用的 Qwen3-1.7B-Base；请设置 BASE_MODEL_PATH。" >&2
  echo "当前解析结果: ${BASE_MODEL_PATH:-<empty>}" >&2
  exit 2
fi

if [[ ! -f "${HF_MODEL_PATH}/model.safetensors" && ! -f "${HF_MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "===== 将 d-OPD DCP 转换为 Hugging Face checkpoint ====="
  python "${MINDSPEED_ROOT}/mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py" \
    --load-dir "${DCP_MODEL_PATH}" \
    --save-dir "${HF_MODEL_PATH}" \
    --model-configs "${BASE_MODEL_PATH}"
else
  echo "===== 复用已有 Hugging Face checkpoint: ${HF_MODEL_PATH} ====="
fi

if [[ ! -f "${HF_MODEL_PATH}/tokenizer_config.json" ]]; then
  python -c 'from transformers import AutoTokenizer; import sys; AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=True).save_pretrained(sys.argv[2])' \
    "${BASE_MODEL_PATH}" "${HF_MODEL_PATH}"
fi

NUM_DEVICES="$(awk -F',' '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
RUN_NAME="$(basename "${DCP_MODEL_PATH%/}")"
RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
METRICS_ROOT="${METRICS_ROOT:-${RUN_OUTPUT}/metrics}"
RESPONSE_CACHE_ROOT="${RESPONSE_CACHE_ROOT:-${OUTPUT_ROOT}/response_cache}"
DECODER_CACHE_VERSION="dopd-paper-c4-block-cache-v2"
mkdir -p "${RUN_OUTPUT}"

if ! [[ "${MAX_TASK_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TASK_ATTEMPTS 必须是正整数，当前为: ${MAX_TASK_ATTEMPTS}" >&2
  exit 2
fi
if [[ -n "${LIMIT}" ]]; then
  echo "[警告] LIMIT=${LIMIT} 仅用于冒烟测试，不可与论文结果比较。"
fi

MODEL_ARGS="pretrained=${HF_MODEL_PATH},tokenizer=${HF_MODEL_PATH},local_files_only=True,trust_remote_code=True,dtype=bfloat16,block_length=${BLOCK_LENGTH},denoising_steps_per_block=${DENOISING_STEPS_PER_BLOCK},response_aligned_blocks=true,diffusion_temperature=${DIFFUSION_TEMPERATURE},threshold=${THRESHOLD},top_k=${TOP_K},top_p=${TOP_P},mask_token_id=${MASK_TOKEN_ID},mc_num=${MC_NUM},mc_batch_size=${MC_BATCH_SIZE},seed=${SEED},enable_thinking=false"
if [[ -z "${MODEL_DEVICE}" && "${NUM_DEVICES}" == "1" ]]; then
  MODEL_DEVICE="npu:0"
fi
if [[ -n "${MODEL_DEVICE}" ]]; then
  MODEL_ARGS+=",device=${MODEL_DEVICE}"
fi

CACHE_CONFIG_ID="$(python -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16])' "${DECODER_CACHE_VERSION}|${MODEL_ARGS}|max_new_tokens=${MAX_NEW_TOKENS}|chat_template=true")"
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
  local alias="$1"
  local task="$2"
  local expected_examples="$3"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_dir="${RUN_OUTPUT}/logs"
  local log_file="${log_dir}/${alias}_${timestamp}.log"
  local summary_log="${log_dir}/${alias}_${timestamp}_summary.log"
  local metrics_run_dir="${METRICS_ROOT}/${alias}_${timestamp}"
  local cache_prefix="${RESPONSE_CACHE_ROOT}/${RUN_NAME}/${CACHE_CONFIG_ID}/${alias}/responses"
  mkdir -p "${log_dir}" "${metrics_run_dir}" "$(dirname "${cache_prefix}")"

  echo "===================================================="
  echo "论文任务      : ${alias} -> ${task}"
  echo "完整样本数    : ${expected_examples}"
  echo "Checkpoint    : ${DCP_MODEL_PATH}"
  echo "Block/steps   : ${BLOCK_LENGTH}/${DENOISING_STEPS_PER_BLOCK}"
  echo "起始状态      : response-aligned 全 MASK block"
  echo "重掩码        : low_confidence_static"
  echo "采样          : temperature=1, top-p=1, top-k=1"
  echo "Prompt        : Qwen3 chat template, thinking=false"
  echo "生成          : zero-shot, 1 completion, max ${MAX_NEW_TOKENS} tokens"
  echo "日志          : ${log_file}"
  echo "===================================================="

  local attempt
  local exit_code=1
  for ((attempt = 1; attempt <= MAX_TASK_ATTEMPTS; attempt++)); do
    local attempt_metrics_dir="${metrics_run_dir}/attempt_${attempt}"
    mkdir -p "${attempt_metrics_dir}"
    echo "[ATTEMPT ${attempt}/${MAX_TASK_ATTEMPTS}] ${alias}" | tee -a "${log_file}"
    if HF_DIFFUSION_METRICS_DIR="${attempt_metrics_dir}" \
      accelerate launch "${ACCELERATE_ARGS[@]}" \
      -m lm_eval \
      --model hf-diffusion \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --num_fewshot 0 \
      --batch_size 1 \
      --apply_chat_template \
      --gen_kwargs "max_gen_toks=${MAX_NEW_TOKENS},temperature=1.0,top_p=1.0,do_sample=True" \
      --verbosity INFO \
      --log_samples \
      --use_cache "${cache_prefix}" \
      --cache_requests \
      --output_path "${RUN_OUTPUT}/${alias}" \
      "${LIMIT_ARGS[@]}" \
      "$@" 2>&1 | tee -a "${log_file}"; then
      python "${SCRIPT_DIR}/scripts/summarize_diffusion_metrics.py" \
        --metrics-dir "${metrics_run_dir}" \
        --results-dir "${RUN_OUTPUT}/${alias}" \
        --output "${summary_log}" \
        --task "${task}" \
        --model "${HF_MODEL_PATH}" \
        --source-log "${log_file}"
      return 0
    else
      exit_code="${PIPESTATUS[0]}"
      echo "[ERROR] ${alias} 第 ${attempt} 次运行失败，exit=${exit_code}" | tee -a "${log_file}"
    fi
  done
  return "${exit_code}"
}

echo "===== d-OPD 论文附录 C.4 对齐评测 ====="
task_selected mmlu && run_task mmlu mmlu_generative 14042
task_selected mmlu_pro && run_task mmlu_pro mmlu_pro 12032
task_selected gpqa_diamond && run_task gpqa_diamond gpqa_diamond_cot_zeroshot 198
task_selected gsm8k && run_task gsm8k gsm8k_cot_zeroshot 1319
task_selected math500 && run_task math500 hendrycks_math500 500
task_selected aime25 && run_task aime25 aime25 30

echo "评测完成: ${RUN_OUTPUT}"
echo "注意：论文还在 step 50,100,...,450,483 中按六项平均分选同一个最佳 checkpoint。"
