#!/usr/bin/env bash

# ==============================
# 1. Load CANN
# ==============================

set +e
set +u

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

source /share/f50058111/activate_cann_910b_9.1.sh

set -euo pipefail

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
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-3600}"

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
THRESHOLD="${THRESHOLD:-0.0}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-1.0}"
MC_NUM="${MC_NUM:-128}"
MC_BATCH_SIZE="${MC_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_diffusion}"
MODEL_DEVICE="${MODEL_DEVICE:-}"

NUM_DEVICES="$(awk -F',' '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
RUN_NAME="$(basename "${MODEL_PATH%/}")"
RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_OUTPUT}"

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

  echo "==== hf-diffusion task=${task}, fewshot=${fewshot}, model=${MODEL_PATH} ===="
  accelerate launch "${ACCELERATE_ARGS[@]}" \
    -m lm_eval \
    --model hf-diffusion \
    --model_args "${MODEL_ARGS}" \
    --tasks "${task}" \
    --num_fewshot "${fewshot}" \
    --batch_size 1 \
    --verbosity INFO \
    --log_samples \
    --output_path "${RUN_OUTPUT}/${task}" \
    "${LIMIT_ARGS[@]}" \
    "$@"
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

echo "Diffusion evaluation finished. Results: ${RUN_OUTPUT}"
