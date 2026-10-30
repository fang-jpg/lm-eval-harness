#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


# ============================================================
# 1. Explicitly initialize CANN 9.1 for Ascend 910B
# ============================================================

export CANN_ROOT="/share/f50058111/Ascend910B_CANN_9.1.0/cann-9.1.0"
export CANN_ENV="${CANN_ROOT}/set_env.sh"
export CANN_OPP_PATH="${CANN_ROOT}/opp"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CANN_ENV}" ]]; then
    echo "[ERROR] CANN set_env.sh not found: ${CANN_ENV}" >&2
    exit 2
fi

if [[ ! -d "${CANN_OPP_PATH}" ]]; then
    echo "[ERROR] CANN OPP directory not found: ${CANN_OPP_PATH}" >&2
    exit 2
fi

# Avoid mixing old CANN toolkit libraries with CANN 9.1.
# Retain non-CANN library paths and Ascend driver libraries.
clean_cann_path() {
    local value="${1:-}"
    local entry
    local result=""
    local -a entries=()

    IFS=':' read -r -a entries <<< "${value}"

    for entry in "${entries[@]}"; do
        [[ -z "${entry}" ]] && continue

        case "${entry}" in
            *Ascend910B_CANN_*|*Ascend910C_CANN_*|\
            */ascend-toolkit/*|*/ascend-toolkit|\
            */ascend/ascend-toolkit/*|*/ascend/ascend-toolkit)
                continue
                ;;
        esac

        if [[ -z "${result}" ]]; then
            result="${entry}"
        else
            result="${result}:${entry}"
        fi
    done

    printf '%s' "${result}"
}

export LD_LIBRARY_PATH="$(clean_cann_path "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(clean_cann_path "${PYTHONPATH:-}")"

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

echo "[CANN] Loading ${CANN_ENV}"

# The vendor script may use optional environment probes.
# Restore strict error handling immediately afterwards.
set +e
set +u
source "${CANN_ENV}"
CANN_SOURCE_STATUS=$?
set -eo pipefail

if [[ ${CANN_SOURCE_STATUS} -ne 0 ]]; then
    echo "[WARN] CANN initialization returned ${CANN_SOURCE_STATUS}" >&2
fi

# Override potentially stale values from a copied set_env.sh.
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_OPP_PATH}"

echo "===== CANN environment ====="
echo "CANN_ROOT=${CANN_ROOT}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Optional existing project-specific OPP kernel verification.
if [[ -f "${SCRIPT_DIR}/scripts/find_working_cann.py" ]]; then
    "${PYTHON_BIN}" \
        "${SCRIPT_DIR}/scripts/find_working_cann.py" \
        --check-opp "${ASCEND_OPP_PATH}" \
        || exit 2
fi

# ============================================================
# 2. Verify PyTorch / torch_npu / basic Cast operation
# ============================================================

"${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch_npu

print("===== NPU environment verification =====")
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("ASCEND_HOME_PATH:", os.environ.get("ASCEND_HOME_PATH"))
print("ASCEND_OPP_PATH:", os.environ.get("ASCEND_OPP_PATH"))

assert torch.npu.is_available(), "NPU unavailable"

x = torch.arange(16, dtype=torch.float32, device="npu:0")
y = x.to(torch.float16)
torch.npu.synchronize()

assert y.dtype == torch.float16
print("[CANN] NPU Cast self-check passed")
PY

# ============================================================
# 3. Enable diagnostic output for segmentation faults
# ============================================================

export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Enable only during debugging: changes execution timing.
if [[ "${NPU_DEBUG_SYNC:-0}" == "1" ]]; then
    export ASCEND_LAUNCH_BLOCKING=1
fi


# 统一代理与基础环境
export GIT_SSL_NO_VERIFY=true
export https_proxy=http://80.253.89.234:8080
export http_proxy=http://80.253.89.234:8080
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local
git config --global http.proxy http://80.253.89.234:8080
git config --global https.proxy http://80.253.89.234:8080

export HF_TOKEN="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"

# 全离线模式（仅使用 ~/.cache/huggingface 中已有的数据）
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_TIMEOUT=3600

# ---------------------------------------------------------------------------
# NPU / 分布式配置（可通过环境变量覆盖，便于并行或串行复用）
#
# 串行（独占机器，上一组已跑完）:
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash hf_zss.sh
#
# 并行（例如 NPU 0-3 已有任务，本脚本用 4-7）:
#   改 ASCEND_RT_VISIBLE_DEVICES，并换一组 HCCL/MASTER 端口，例如:
#   HCCL_HOST_SOCKET_PORT_RANGE=60200-60299 \
#   HCCL_NPU_SOCKET_PORT_RANGE=61200-61299 \
#   MASTER_PORT=29502 \
#   MODEL_PATH=/path/to/weights bash hf_zss.sh
# 
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# ---------------------------------------------------------------------------
#export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-12,13,14,15}"
#export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-14,15}"
#export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | awk -F',' '{print NF}')
PYTHON_BIN="${PYTHON_BIN:-python}"

# HCCL 默认 60000-60015；多任务同机并行必须换端口段（MASTER_PORT 管不了 HCCL）
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60100-60199}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61100-61199}"
# export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60200-60299}"
# export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61200-61299}"
export MASTER_PORT="${MASTER_PORT:-29501}"
# export MASTER_PORT="${MASTER_PORT:-29502}"
ACCELERATE_PORT_ARGS=(--main_process_port "${MASTER_PORT}")

# 待评测 HF 权重路径
# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/OPD_200_GSM_DAPO}"
# /home/l00899529/OPD/weights/Dolci_sft_hybrid_hf_OPD_stage1gsm8k

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/Dolci_sft_hybrid_hf_OPD_stage1gsm8k}"

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/code/OPD/weights/Qwen2.5-7B-Instruct}"

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/Dolci_sft_hybrid_hf_OPD_stage1gsm8k}"


# MODEL_PATH="${MODEL_PATH:-/home/l00899529/code/OPD/weights/Qwen2.5-1.5B}"


# MODEL_PATH="${MODEL_PATH:-/home/l00899529/code/OPD/weights/qwen_hybrid_untie}"

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/0704_longSFT_sci_if_math_code_3500}"

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/0704_longSFT_sci_if_math_code_1600}" 

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/0702_longSFT_sci_if_math_code_1450}" 

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/OPD/weights/0706_k1_SFT_sci_if_math_code_1550_OPD_OpenMath}" 

# MODEL_PATH="${MODEL_PATH:-/home/l00899529/code/OPD/weights/0702_SFT_sci_if_math_code_HF}"

# MODEL_PATH="${MODEL_PATH:-/home/z50062551/weights/Qwen2.5-Math-7B-Instruct}"

#MODEL_PATH="${MODEL_PATH:-/home/z50062551/weights/0707_top64_teacher7bmath_SFT_sci_if_math_code_50_OPD_500_OpenMath}"
#MODEL_PATH="${MODEL_PATH:-/home/l00899529/code/OPD/weights/HF/0709_top64_SFT_sci_if_math_code_1550_OPD_OpenMath_100_mmlu_200_HF}"
#MODEL_PATH="${MODEL_PATH:-/home/l00899529/code/OPD/weights/HF/global_step_400}"

MODEL_PATH="${MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_ar_cpt_baseline_0920/models/global_step_102500/}"
# MODEL_PATH="${MODEL_PATH:-/home/x00840191/model_train/Qwen3-1.7B-Base/}"

# 当前脚本只执行一项普通 AR 生成评测；可通过环境变量覆盖。
TASK="${TASK:-gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
LIMIT="${LIMIT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/eval_results_CPT_baseline_AR_256}"
RUN_OUTPUT="${OUTPUT_ROOT}/${TASK}"
LOG_DIR="${OUTPUT_ROOT}/logs"
METRICS_ROOT="${OUTPUT_ROOT}/metrics"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
METRICS_DIR="${METRICS_ROOT}/${TASK}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/${TASK}_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/${TASK}_${TIMESTAMP}_summary.log"
mkdir -p "${RUN_OUTPUT}" "${LOG_DIR}" "${METRICS_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True"
LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

# ---------------------------------------------------------------------------
# 本地 cache 状态（docker: lm_eval_zss, ~/.cache/huggingface/）
#   ✅ openai/gsm8k              -> hub/datasets--openai--gsm8k
#   ✅ cais/mmlu                 -> hub/datasets--cais--mmlu
#   ✅ openai/openai_humaneval   -> hub/datasets--openai--openai_humaneval
#   ✅ aps/super_glue (boolq)    -> hub/datasets--aps--super_glue
#   ✅ SaylorTwift/bbh (bbh)     -> hub/datasets--SaylorTwift--bbh
#                                 + datasets/SaylorTwift___bbh
#   BBH 若缺失，可从 s00525112 复制:
#     cp -a /home/s00525112/huggingface/hub/datasets--SaylorTwift--bbh ~/.cache/huggingface/hub/
#     cp -a /home/s00525112/huggingface/datasets/SaylorTwift___bbh ~/.cache/huggingface/datasets/
# ---------------------------------------------------------------------------

# ==============================================
# 第一组：GSM8K 评测 (0-shot，参考 hf_nlc.sh)
# ==============================================

# ==============================================
# 第二组：MMLU 评测 (5-shot)
# ==============================================
# accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" "${ACCELERATE_PORT_ARGS[@]}" \
#    -m lm_eval --model hf \
#    --model_args pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True \
#    --tasks mmlu \
#    --num_fewshot 5 \
#    --batch_size 1 \
#    --verbosity DEBUG \
#    --log_samples \
#    --output_path ./eval_results/mmlu

# ==============================================
# 第▒~@▒~D▒~ZGSM8K ▒~D▒~K (0-shot▒~L▒~O~B▒~@~C hf_nlc.sh)
# ==============================================
echo "===================================================="
echo "AR evaluation with performance metrics"
echo "Task       : ${TASK}"
echo "Model      : ${MODEL_PATH}"
echo "Fewshot    : ${NUM_FEWSHOT}"
echo "Devices    : ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Results    : ${RUN_OUTPUT}"
echo "Metrics    : ${METRICS_DIR}"
echo "Log        : ${LOG_FILE}"
echo "Summary    : ${SUMMARY_LOG}"
echo "Start time : $(date '+%F %T')"
echo "===================================================="

if HF_AR_METRICS_DIR="${METRICS_DIR}" \
  accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" "${ACCELERATE_PORT_ARGS[@]}" \
    -m lm_eval --model hf-ar-metrics \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --batch_size 1 \
    # --gen_kwargs '{"do_sample":false,"max_gen_toks":256,"min_new_tokens":256,"until":[]}' \    #忽略EOS终止符
    --verbosity DEBUG \
    --log_samples \
    --output_path "${RUN_OUTPUT}" \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/summarize_ar_metrics.py" \
    --metrics-dir "${METRICS_DIR}" \
    --results-dir "${RUN_OUTPUT}" \
    --output "${SUMMARY_LOG}" \
    --task "${TASK}" \
    --model "${MODEL_PATH}" \
    --source-log "${LOG_FILE}"
  echo "[SUMMARY] Accuracy and AR performance: ${SUMMARY_LOG}" | tee -a "${LOG_FILE}"
else
  EXIT_CODE="${PIPESTATUS[0]}"
  echo "[ERROR] AR evaluation failed, exit code=${EXIT_CODE}" | tee -a "${LOG_FILE}"
  exit "${EXIT_CODE}"
fi


# ==============================================
# 第三组：HumanEval 评测 (0-shot，需执行生成代码)
# ==============================================
# accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" "${ACCELERATE_PORT_ARGS[@]}" \
#    -m lm_eval --model hf \
#    --model_args pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True \
#    --tasks humaneval \
#    --batch_size 1 \
#    --verbosity DEBUG \
#    --log_samples \
#    --output_path ./eval_results/humaneval \
#    --confirm_run_unsafe_code

# ==============================================
# 第四组：BoolQ 评测 (5-shot)
# ==============================================
# accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" "${ACCELERATE_PORT_ARGS[@]}" \
#    -m lm_eval --model hf \
#    --model_args pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True \
#    --tasks boolq \
#    --num_fewshot 5 \
#    --batch_size 1 \
#    --verbosity DEBUG \
#    --log_samples \
#    --output_path ./eval_results/boolq

# # ==============================================
# # 第五组：BBH 评测 (3-shot)
# # ==============================================
# accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" "${ACCELERATE_PORT_ARGS[@]}" \
#     -m lm_eval --model hf \
#     --model_args pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True \
#     --tasks bbh \
#     --num_fewshot 3 \
#     --batch_size 1 \
#     --verbosity DEBUG \
#     --log_samples \
#     --output_path ./eval_results/bbh

