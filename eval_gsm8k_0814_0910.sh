#!/usr/bin/env bash
# Run from the NPU server: bash eval_gsm8k_0814_0910.sh
# Preview all 48 commands without CANN/models: DRY_RUN=1 bash eval_gsm8k_0814_0910.sh
# Optional overrides: ROOT_0814, ROOT_0910, OUTPUT_ROOT, ASCEND_RT_VISIBLE_DEVICES,
# CANN_ENV, HF_HOME, BLOCK_LENGTH, DIFFUSION_STEPS, MASK_TOKEN_ID, MODEL_DEVICE.
# Requires the reference script's Python environment plus matplotlib.
set -eo pipefail
trap 'printf "[ERROR] line %s, exit %s\n" "$LINENO" "$?" >&2' ERR
echo "[startup] GSM8K evaluation starting"
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  echo "[startup] Loading CANN: ${CANN_ENV:-/share/f50058111/activate_cann_910b_9.1.sh}"
  # Match the reference setup script: vendor activation may use unset variables
  # or return nonzero from optional probes. Validate the NPU below instead.
  set +e
  set +u
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
  source "${CANN_ENV:-/share/f50058111/activate_cann_910b_9.1.sh}"
fi
set -euo pipefail
export HF_HOME="${HF_HOME:-/share/f50058111/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-3600}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60300-60399}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61300-61399}"
export MASTER_PORT="${MASTER_PORT:-29503}"

echo "[startup] Python: $(command -v python)"
echo "[startup] NPU devices: ${ASCEND_RT_VISIBLE_DEVICES}"
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

env = os.environ
dry_run = env.get('DRY_RUN', '0') == '1'
roots = {
    run: Path(env.get(f'ROOT_{run}',
        f'/share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_{run}/models'))
    for run in ('0814', '0910')
}
steps = list(range(500, 6001, 500))
devices = [x.strip() for x in env['ASCEND_RT_VISIBLE_DEVICES'].split(',') if x.strip()]
if not devices:
    raise SystemExit('ASCEND_RT_VISIBLE_DEVICES is empty')
device = env.get('MODEL_DEVICE', 'npu:0' if len(devices) == 1 else '')
seed = env.get('SEED', '42')
metric = 'exact_match,strict-match'
flexible = 'exact_match,flexible-extract'
output = Path(env.get('OUTPUT_ROOT', './eval_results_gsm8k_0814_0910')).resolve()

def command(run, step, mode, destination):
    model = roots[run] / f'global_step_{step}'
    args = (f'pretrained={model},tokenizer={model},local_files_only=True,'
            'trust_remote_code=True,dtype=bfloat16')
    if device:
        args += f',device={device}'
    if mode == 'Diffusion':
        args += (f",block_length={env.get('BLOCK_LENGTH', '8')},"
                 f"diffusion_temperature=0.0,threshold={env.get('THRESHOLD', '0.0')},"
                 f'top_k=0,top_p=1.0,mc_num=128,mc_batch_size=16,seed={seed}')
        for key in ('DIFFUSION_STEPS', 'MASK_TOKEN_ID'):
            if env.get(key):
                args += f',{key.lower()}={env[key]}'
    cmd = [sys.executable, '-m', 'accelerate.commands.launch']
    if len(devices) > 1:
        cmd += ['--multi_gpu']
    cmd += ['--num_processes', str(len(devices)), '--main_process_port', env['MASTER_PORT'],
            '-m', 'lm_eval', '--model', 'hf' if mode == 'AR' else 'hf-diffusion',
            '--model_args', args, '--tasks', 'gsm8k', '--num_fewshot', '5',
            '--batch_size', '1', '--seed', f'{seed},{seed},{seed},{seed}',
            '--verbosity', 'WARNING', '--output_path', str(destination)]
    return cmd

if dry_run:
    for run in roots:
        for step in steps:
            for mode in ('AR', 'Diffusion'):
                print(shlex.join(command(run, step, mode, '/tmp/gsm8k-eval-preview')))
    raise SystemExit(0)

# Fail before starting expensive evaluation if any required model is missing.
print('[startup] Checking all 24 checkpoints...', flush=True)
missing = [str(root / f'global_step_{step}' / 'config.json')
           for root in roots.values() for step in steps
           if not (root / f'global_step_{step}' / 'config.json').is_file()]
if missing:
    raise SystemExit('Missing HF checkpoints (expected global_step_<step>):\n' + '\n'.join(missing))
print('[startup] Importing matplotlib, torch and torch_npu...', flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch_npu
if not torch.npu.is_available():
    raise SystemExit('NPU is unavailable; check the CANN/Python environment.')
print('[startup] NPU available; starting evaluation.', flush=True)

output.mkdir(parents=True, exist_ok=True)
# Refuse to mix a new evaluation with old final results.
if any((output / name).exists() for name in ('results.csv', 'results.json', 'gsm8k_comparison.png')):
    raise SystemExit(f'Final results already exist in {output}; choose another OUTPUT_ROOT.')
rows = []
summaries = []

def save_final_metrics():
    # Save completed scores after each run so an interruption does not lose them.
    # These are final aggregate metrics only, never per-example generations.
    fields = ['run', 'step', 'mode', 'strict_match', 'flexible_extract',
              'strict_match_stderr', 'flexible_extract_stderr', 'checkpoint']
    temp = output / 'results.csv.tmp'
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output / 'results.csv')
    temp = output / 'results.json.tmp'
    temp.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(output / 'results.json')

try:
    for run in roots:
        for step in steps:
            for mode in ('AR', 'Diffusion'):
                print(f'[{len(rows)+1}/48] {run} step={step} {mode}', flush=True)
                # Temporary aggregate JSON is removed even on failure.
                # Progress goes directly to the terminal, with no saved console log.
                # No --log_samples, response cache, or persistent evaluation log.
                with tempfile.TemporaryDirectory(prefix='gsm8k-eval-') as tmp:
                    dest = Path(tmp) / 'metrics'
                    cmd = command(run, step, mode, dest)
                    result = subprocess.run(cmd, env=dict(env, PYTHONUNBUFFERED='1'))
                    if result.returncode:
                        raise RuntimeError(f'{run} step={step} {mode} failed: exit {result.returncode}')
                    files = list(dest.rglob('results*.json'))
                    if len(files) != 1:
                        raise RuntimeError(f'Expected one aggregate result, found {len(files)}')
                    data = json.loads(files[0].read_text(encoding='utf-8'))
                    scores = data['results']['gsm8k']
                    for key in (metric, flexible):
                        value = scores[key]
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError(f'Invalid score {key}: {value}')
                    rows.append(dict(run=run, step=step, mode=mode,
                        strict_match=scores[metric], flexible_extract=scores[flexible],
                        strict_match_stderr=scores.get('exact_match_stderr,strict-match'),
                        flexible_extract_stderr=scores.get('exact_match_stderr,flexible-extract'),
                        checkpoint=str(roots[run] / f'global_step_{step}')))
                    summaries.append(dict(**rows[-1], metrics=scores,
                        n_samples=data.get('n-samples'), command=cmd[:-1] + ['<temporary-results>']))
                    save_final_metrics()
                    print(f"  strict={scores[metric]:.4%}, flexible={scores[flexible]:.4%}", flush=True)
except KeyboardInterrupt:
    raise SystemExit(f'Interrupted. Completed final metrics are in {output}')

fig, ax = plt.subplots(figsize=(11, 6))
for run, color in (('0814', '#2563eb'), ('0910', '#ea580c')):
    for mode, style, marker in (('AR', '-', 'o'), ('Diffusion', '--', 's')):
        values = [r for r in rows if r['run'] == run and r['mode'] == mode]
        ax.plot([r['step'] for r in values], [r['strict_match'] * 100 for r in values],
                color=color, linestyle=style, marker=marker, label=f'{run} {mode}')
ax.set(title='GSM8K: AR vs Diffusion (5-shot, full test set)',
       xlabel='Training step', ylabel='Exact match / strict-match (%)', xticks=steps)
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(output / 'gsm8k_comparison.png', dpi=200)
plt.close(fig)
print(f'Finished all 48 evaluations. Final CSV, JSON and chart: {output}')
PY
