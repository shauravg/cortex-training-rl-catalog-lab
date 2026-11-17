#!/usr/bin/env bash
# Launch listing-review GRPO on Cortex with the recipe's tested defaults
# (Qwen3-1.7B, 4 training + 4 sampling GPUs, 8 samples per prompt).
# Run from Arctic-Platform/recipes/rl/trl/gsm8k_cortex after copying
# listing_task.py, run_listing_grpo_cortex.py and deploy_tuned.json into that
# directory. At the end of the run the driver saves a weights-only checkpoint
# and fills its job and checkpoint IDs into deploy_tuned.json.
# Stop with kill -INT so the client cancels its Cortex job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for var in ARCTIC_CORTEX_HOST ARCTIC_CORTEX_DATABASE ARCTIC_CORTEX_SCHEMA ARCTIC_CORTEX_PAT; do
  [[ -n "${!var:-}" ]] || { echo "Set $var (do not print the PAT)" >&2; exit 2; }
done

# The recipe file name has a dot in it, so give Python an importable alias.
ln -sf run_qwen3_1.7b_gsm8k_grpo_cortex.py run_qwen3_1_7b_gsm8k_grpo_cortex.py

export TRL_EXPERIMENTAL_SILENCE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export ARCTIC_ATTN_IMPL="${ARCTIC_ATTN_IMPL:-sdpa}" ARCTIC_ZERO_STAGE="${ARCTIC_ZERO_STAGE:-1}"

exec python -u run_listing_grpo_cortex.py \
  --model "${MODEL:-Qwen/Qwen3-1.7B}" \
  --max-steps "${MAX_STEPS:-200}" \
  --num-train-epochs 1 \
  --num-prompts -1 \
  --training-gpus "${TRAINING_GPUS:-4}" \
  --sampling-gpus "${SAMPLING_GPUS:-4}" \
  --num-generations "${NUM_GEN:-8}" \
  --per-device-bsz "${PER_DEVICE_BSZ:-32}" \
  --max-completion-length "${MAX_COMPLETION_LEN:-256}" \
  --max-seq-len "${MAX_SEQ_LEN:-1024}" \
  --seed "${SEED:-42}" \
  --max-staleness 0 \
  --old-logprobs-source "${OLD_LOGPROBS_SOURCE:-trainer}" \
  --metrics-out "${METRICS_OUT:-metrics_listing_grpo.json}" \
  "$@"
