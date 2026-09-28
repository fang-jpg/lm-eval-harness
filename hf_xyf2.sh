#!/bin/bash
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

MODEL_PATH="${MODEL_PATH:-/data2/model_train/ckpt/phase4_cpt_ar_diffusion_0916/models/global_step_10500}"
# MODEL_PATH="${MODEL_PATH:-/home/x00840191/model_train/Qwen3-1.7B-Base/}"

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
accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" "${ACCELERATE_PORT_ARGS[@]}" \
   -m lm_eval --model hf \
   --model_args pretrained=${MODEL_PATH},tokenizer=${MODEL_PATH},local_files_only=True,trust_remote_code=True \
   --tasks gsm8k \
   --num_fewshot 5 \
   --batch_size 1 \
   --verbosity DEBUG \
   --log_samples \
   --output_path ./eval_results_baseline/gsm8k


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

