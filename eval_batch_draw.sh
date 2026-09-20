#!/usr/bin/env bash

# ==============================================================================
# GSM8K Baseline AR Evaluation (0920)
#
# 默认：
#   模型：
#   /share/dataset/x00840191/model_train/ckpt/
#       qwen3_1.7b_ar_cpt_baseline_0920/models/global_step_xxx
#
#   checkpoints:
#       自动发现 models/global_step_*/config.json 已完整导出的 checkpoint
#
#   devices:
#       NPU 0~7
#
# 使用：
#   bash eval_batch_draw.sh
#
# 单卡调试：
#   ASCEND_RT_VISIBLE_DEVICES=6 bash eval_batch_draw.sh
#
# 指定输出：
#   OUTPUT_ROOT=./eval_results_test bash eval_batch_draw.sh
#
# 只评测指定 checkpoint（逗号或空格分隔）：
#   STEPS=5000 bash eval_batch_draw.sh
#
# 预览命令：
#   STEPS=5000 DRY_RUN=1 bash eval_batch_draw.sh
# ==============================================================================

set -eo pipefail

trap 'printf "[ERROR] line %s, exit %s\n" "$LINENO" "$?" >&2' ERR

echo "============================================================"
echo "[startup] GSM8K Baseline AR evaluation"
echo "============================================================"

cd "$(dirname "${BASH_SOURCE[0]}")"


# ==============================================================================
# 1. 加载 Ascend / CANN 环境
# ==============================================================================

if [[ "${DRY_RUN:-0}" != "1" ]]; then

    CANN_ENV="${CANN_ENV:-/share/f50058111/activate_cann_910b_9.1.sh}"

    echo "[startup] Loading CANN:"
    echo "          ${CANN_ENV}"

    # 部分 CANN activation 脚本可能引用未定义变量，
    # 因此 source 时临时关闭严格模式。
    set +e
    set +u

    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

    source "${CANN_ENV}"

fi

set -euo pipefail


# ==============================================================================
# 2. HuggingFace 离线环境
# ==============================================================================

export HF_HOME="${HF_HOME:-/share/f50058111/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export TOKENIZERS_PARALLELISM=false


# ==============================================================================
# 3. HCCL / NPU
# ==============================================================================

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-3600}"

# 默认使用 8 张卡。
#
# 如果机器上另一个评测任务正在占 0~7，
# 不要再直接启动这个任务。
#
# 可以例如：
#
# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
# MASTER_PORT=29513 \
# HCCL_HOST_SOCKET_PORT_RANGE=62300-62399 \
# HCCL_NPU_SOCKET_PORT_RANGE=63300-63399 \
# bash eval_batch_draw.sh
#
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"


# ==============================================================================
# 4. 分布式端口
# ==============================================================================


export MASTER_PORT="${MASTER_PORT:-29503}"

export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60300-60399}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61300-61399}"


# ==============================================================================
# 5. 输出一些环境信息
# ==============================================================================

echo "[startup] Python:"
echo "          $(command -v python)"

echo "[startup] NPU devices:"
echo "          ${ASCEND_RT_VISIBLE_DEVICES}"

echo "[startup] MASTER_PORT:"
echo "          ${MASTER_PORT}"

echo "[startup] HCCL host ports:"
echo "          ${HCCL_HOST_SOCKET_PORT_RANGE}"

echo "[startup] HCCL NPU ports:"
echo "          ${HCCL_NPU_SOCKET_PORT_RANGE}"


# ==============================================================================
# 6. 主评测逻辑
# ==============================================================================

python -u - <<'PY'

import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


# ==============================================================================
# 基础配置
# ==============================================================================

env = os.environ

dry_run = env.get("DRY_RUN", "0") == "1"


# ------------------------------------------------------------------------------
# Baseline 模型目录
# ------------------------------------------------------------------------------

run_name = env.get("RUN_NAME", "0920")

root = Path(
    env.get(
        "MODEL_ROOT",
        "/share/dataset/x00840191/model_train/ckpt/"
        "qwen3_1.7b_ar_cpt_baseline_0920/models"
    )
)


# ------------------------------------------------------------------------------
# checkpoint 范围
# ------------------------------------------------------------------------------

steps_arg = env.get("STEPS", "").strip()

if steps_arg:
    try:
        steps = sorted(
            {
                int(value)
                for value in steps_arg.replace(",", " ").split()
            }
        )
    except ValueError as e:
        raise SystemExit(
            "STEPS must contain integers separated by commas or spaces; "
            f"got {steps_arg!r}"
        ) from e

    if not steps or any(step <= 0 for step in steps):
        raise SystemExit(
            f"STEPS must contain positive integers; got {steps_arg!r}"
        )

    steps_source = "STEPS override"

else:
    steps = []

    if root.is_dir():
        for model_dir in root.glob("global_step_*"):
            suffix = model_dir.name.removeprefix("global_step_")

            # save_model_weights() writes config.json after all weight shards,
            # so its presence is used as the completed-export marker.
            if (
                model_dir.is_dir()
                and suffix.isdigit()
                and (model_dir / "config.json").is_file()
            ):
                steps.append(int(suffix))

    steps = sorted(set(steps))
    steps_source = "auto-discovered"

if not steps:
    raise SystemExit(
        "No completed HF checkpoints found.\n"
        f"Model root: {root}\n"
        "Wait for the next save step, set MODEL_ROOT, or provide STEPS."
    )


# ------------------------------------------------------------------------------
# NPU
# ------------------------------------------------------------------------------

devices = [
    x.strip()
    for x in env["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if x.strip()
]

if not devices:
    raise SystemExit(
        "ASCEND_RT_VISIBLE_DEVICES is empty"
    )


num_processes = len(devices)


# ------------------------------------------------------------------------------
# 评测参数
# ------------------------------------------------------------------------------

seed = env.get("SEED", "42")

metric_strict = "exact_match,strict-match"

metric_flexible = "exact_match,flexible-extract"


# ------------------------------------------------------------------------------
# 输出目录
# ------------------------------------------------------------------------------

output = Path(
    env.get(
        "OUTPUT_ROOT",
        "./eval_results_gsm8k_baseline_0920"
    )
).resolve()


print()
print("============================================================")
print("[config]")
print(f"model root     : {root}")
print(f"steps          : {steps[0]} ~ {steps[-1]}")
print(f"num checkpoints: {len(steps)}")
print(f"steps source   : {steps_source}")
print(f"NPU devices    : {devices}")
print(f"num processes  : {num_processes}")
print(f"output         : {output}")
print("============================================================")
print()


# ==============================================================================
# 构造评测命令
# ==============================================================================

def build_command(step, destination):

    model_dir = root / f"global_step_{step}"

    # --------------------------------------------------------------------------
    # 这里不要再放 device=npu:x
    #
    # 多卡模式：
    # Accelerate 会负责：
    #
    # rank0 -> 第一张可见 NPU
    # rank1 -> 第二张可见 NPU
    # ...
    #
    # 不需要手动给每个模型指定 device。
    # --------------------------------------------------------------------------

    model_args = (
        f"pretrained={model_dir},"
        f"tokenizer={model_dir},"
        "local_files_only=True,"
        "trust_remote_code=True,"
        "dtype=bfloat16"
    )


    # --------------------------------------------------------------------------
    # accelerate launcher
    # --------------------------------------------------------------------------

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]


    # --------------------------------------------------------------------------
    # 多卡
    # --------------------------------------------------------------------------

    if num_processes > 1:

        cmd += [
            "--multi_gpu",
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            env["MASTER_PORT"],
        ]


    # --------------------------------------------------------------------------
    # 单卡
    #
    # ASCEND_RT_VISIBLE_DEVICES=6 时，
    # 物理 NPU6 会映射为当前进程中的 npu:0。
    # --------------------------------------------------------------------------

    else:

        cmd += [
            "--num_processes",
            "1",
            "--main_process_port",
            env["MASTER_PORT"],
        ]


    # --------------------------------------------------------------------------
    # lm-eval
    # --------------------------------------------------------------------------

    cmd += [
        "-m",
        "lm_eval",

        "--model",
        "hf",

        "--model_args",
        model_args,
    ]


    # --------------------------------------------------------------------------
    # 单卡时，把 device 作为 lm-eval 顶层参数传递。
    #
    # 注意：
    #
    # 正确：
    #   --device npu:0
    #
    # 不再使用：
    #   --model_args ...,device=npu:0
    #
    # 防止 HFLM(device=..., device=...) 重复传参。
    # --------------------------------------------------------------------------

    if num_processes == 1:
        cmd += [
            "--device",
            "npu:0",
        ]


    cmd += [
        "--tasks",
        "gsm8k",

        "--num_fewshot",
        "5",

        "--batch_size",
        "1",

        "--seed",
        f"{seed},{seed},{seed},{seed}",

        "--verbosity",
        "WARNING",

        "--output_path",
        str(destination),
    ]

    return cmd


# ==============================================================================
# DRY RUN
# ==============================================================================

if dry_run:

    print("[dry-run] Commands:")
    print()

    for step in steps:

        cmd = build_command(
            step,
            "/tmp/gsm8k-baseline-preview",
        )

        print(
            shlex.join(cmd)
        )

    raise SystemExit(0)


# ==============================================================================
# 7. 检查 checkpoints
# ==============================================================================

print(
    f"[startup] Checking all {len(steps)} checkpoints...",
    flush=True,
)

missing = []

for step in steps:

    config = (
        root
        / f"global_step_{step}"
        / "config.json"
    )

    if not config.is_file():
        missing.append(str(config))


if missing:

    raise SystemExit(
        "Missing HF checkpoints:\n"
        + "\n".join(missing)
    )


print(
    "[startup] All checkpoints found.",
    flush=True,
)


# ==============================================================================
# 8. 检查 NPU
# ==============================================================================

print(
    "[startup] Importing matplotlib, torch and torch_npu...",
    flush=True,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch_npu


if not torch.npu.is_available():

    raise SystemExit(
        "NPU is unavailable. "
        "Check CANN / torch_npu environment."
    )


print(
    "[startup] NPU available; starting evaluation.",
    flush=True,
)


# ==============================================================================
# 9. 输出文件
# ==============================================================================

output.mkdir(
    parents=True,
    exist_ok=True,
)

csv_path = output / "results.csv"

json_path = output / "results.json"

plot_path = output / "gsm8k_baseline_curve.png"


# ==============================================================================
# 10. 支持断点续跑
# ==============================================================================

rows = []

summaries = []

finished_steps = set()


if csv_path.exists():

    print(
        f"[resume] Existing results found: {csv_path}",
        flush=True,
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            row["step"] = int(row["step"])

            row["strict_match"] = float(
                row["strict_match"]
            )

            row["flexible_extract"] = float(
                row["flexible_extract"]
            )

            rows.append(row)

            finished_steps.add(
                row["step"]
            )


if json_path.exists():

    try:

        summaries = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:

        print(
            f"[warning] Cannot read existing results.json: {e}",
            flush=True,
        )


if finished_steps:

    print(
        "[resume] Completed steps:",
        sorted(finished_steps),
        flush=True,
    )


# ==============================================================================
# 11. 保存结果
# ==============================================================================

def save_results():

    fields = [
        "run",
        "step",
        "mode",
        "strict_match",
        "flexible_extract",
        "strict_match_stderr",
        "flexible_extract_stderr",
        "checkpoint",
    ]


    # --------------------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------------------

    temp_csv = output / "results.csv.tmp"

    with temp_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)

    temp_csv.replace(csv_path)


    # --------------------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------------------

    temp_json = output / "results.json.tmp"

    temp_json.write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temp_json.replace(json_path)


# ==============================================================================
# 12. 正式评测
# ==============================================================================

try:

    total = len(steps)

    for index, step in enumerate(
        steps,
        start=1,
    ):

        # ----------------------------------------------------------------------
        # 断点续跑
        # ----------------------------------------------------------------------

        if step in finished_steps:

            print(
                f"[skip] step={step} already finished.",
                flush=True,
            )

            continue


        print()
        print(
            "============================================================"
        )

        print(
            f"[{index}/{total}] "
            f"Evaluating baseline step={step} "
            f"(AR, GSM8K 5-shot)"
        )

        print(
            "============================================================",
            flush=True,
        )


        model_dir = (
            root
            / f"global_step_{step}"
        )


        # ----------------------------------------------------------------------
        # lm-eval 临时结果目录
        # ----------------------------------------------------------------------

        with tempfile.TemporaryDirectory(
            prefix="gsm8k-baseline-"
        ) as tmp:

            destination = (
                Path(tmp)
                / "metrics"
            )


            cmd = build_command(
                step,
                destination,
            )


            print(
                "[command]",
                shlex.join(cmd),
                flush=True,
            )


            # ------------------------------------------------------------------
            # 启动 accelerate / lm-eval
            # ------------------------------------------------------------------

            result = subprocess.run(
                cmd,
                env=dict(
                    env,
                    PYTHONUNBUFFERED="1",
                ),
            )


            if result.returncode != 0:

                raise RuntimeError(
                    f"{run_name} baseline "
                    f"step={step} "
                    f"failed with exit code "
                    f"{result.returncode}"
                )


            # ------------------------------------------------------------------
            # 找 lm-eval 输出结果
            # ------------------------------------------------------------------

            result_files = list(
                destination.rglob(
                    "results*.json"
                )
            )


            if not result_files:

                raise RuntimeError(
                    "No results*.json generated "
                    f"in {destination}"
                )


            # 最新生成的结果文件
            result_files.sort(
                key=lambda f: f.stat().st_mtime
            )

            result_file = result_files[-1]


            print(
                f"[result] Reading {result_file}",
                flush=True,
            )


            data = json.loads(
                result_file.read_text(
                    encoding="utf-8"
                )
            )


            # ------------------------------------------------------------------
            # GSM8K scores
            # ------------------------------------------------------------------

            scores = (
                data["results"]["gsm8k"]
            )


            # ------------------------------------------------------------------
            # 校验
            # ------------------------------------------------------------------

            for key in (
                metric_strict,
                metric_flexible,
            ):

                value = scores.get(key)

                if (
                    not isinstance(
                        value,
                        (int, float),
                    )
                    or not math.isfinite(value)
                    or not (0 <= value <= 1)
                ):

                    raise ValueError(
                        f"Invalid metric "
                        f"{key}: {value}"
                    )


            # ------------------------------------------------------------------
            # 保存一行
            # ------------------------------------------------------------------

            row = dict(

                run=run_name,

                step=step,

                mode="AR",

                strict_match=(
                    scores[
                        metric_strict
                    ]
                ),

                flexible_extract=(
                    scores[
                        metric_flexible
                    ]
                ),

                strict_match_stderr=(
                    scores.get(
                        "exact_match_stderr,"
                        "strict-match"
                    )
                ),

                flexible_extract_stderr=(
                    scores.get(
                        "exact_match_stderr,"
                        "flexible-extract"
                    )
                ),

                checkpoint=str(
                    model_dir
                ),
            )


            rows.append(row)


            summaries.append(

                dict(

                    **row,

                    metrics=scores,

                    n_samples=(
                        data.get(
                            "n-samples"
                        )
                    ),

                    command=(
                        cmd[:-1]
                        + [
                            "<temporary-results>"
                        ]
                    ),
                )
            )


            # ------------------------------------------------------------------
            # 每跑完一个 checkpoint 就保存
            # ------------------------------------------------------------------

            save_results()


            print(
                f"[done] step={step}"
            )

            print(
                f"       strict   = "
                f"{scores[metric_strict]:.4%}"
            )

            print(
                f"       flexible = "
                f"{scores[metric_flexible]:.4%}",
                flush=True,
            )


except KeyboardInterrupt:

    save_results()

    raise SystemExit(
        "\nInterrupted by user.\n"
        f"Completed results saved in:\n"
        f"{output}"
    )


# ==============================================================================
# 13. 绘制准确率曲线
# ==============================================================================

if rows:

    rows.sort(
        key=lambda x: int(x["step"])
    )


    x_steps = [
        int(r["step"])
        for r in rows
    ]


    y_strict = [
        float(r["strict_match"]) * 100
        for r in rows
    ]


    y_flexible = [
        float(r["flexible_extract"]) * 100
        for r in rows
    ]


    fig, ax = plt.subplots(
        figsize=(14, 6)
    )


    ax.plot(
        x_steps,
        y_strict,
        linestyle="-",
        marker="o",
        label="AR Baseline (strict-match)",
    )


    ax.plot(
        x_steps,
        y_flexible,
        linestyle="--",
        marker="^",
        label="AR Baseline (flexible-extract)",
    )


    ax.set_title(
        "GSM8K 5-shot Accuracy Curve "
        f"(Qwen3-1.7B AR CPT Baseline {run_name})",
        fontsize=14,
    )


    ax.set_xlabel(
        "Training Step",
        fontsize=12,
    )


    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=12,
    )


    # checkpoint 较多时不把所有刻度全部塞进去
    available_tick_steps = sorted(set(x_steps))

    tick_steps = [

        s for s in available_tick_steps

        if (
            s % 2500 == 0
            or s == available_tick_steps[0]
            or s == available_tick_steps[-1]
        )
    ]


    ax.set_xticks(
        tick_steps
    )


    ax.grid(
        True,
        linestyle=":",
        alpha=0.6,
    )


    ax.legend(
        fontsize=11
    )


    fig.tight_layout()


    fig.savefig(
        plot_path,
        dpi=200,
    )


    plt.close(fig)


# ==============================================================================
# 14. 完成
# ==============================================================================

print()
print("============================================================")
print("All baseline evaluations finished successfully!")
print("============================================================")
print(f"CSV  : {csv_path}")
print(f"JSON : {json_path}")
print(f"Plot : {plot_path}")
print()

PY
