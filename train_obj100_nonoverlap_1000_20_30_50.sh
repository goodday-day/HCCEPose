#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/yaohua/miniconda3/envs/hccepose/bin/python"
split_dir="${script_dir}/../Pointcept/data_splits"
train_manifest="${split_dir}/raw_depth_nonoverlap_1000_20_30_50_train_seed2026.csv"
val_manifest="${split_dir}/raw_depth_nonoverlap_1000_20_30_50_val_seed2026.csv"
gpu_id="${1:-1}"

[[ -x "${python_bin}" ]] || { echo "Missing HccePose Python: ${python_bin}" >&2; exit 1; }
[[ -f "${train_manifest}" ]] || { echo "Missing train manifest: ${train_manifest}" >&2; exit 1; }
[[ -f "${val_manifest}" ]] || { echo "Missing validation manifest: ${val_manifest}" >&2; exit 1; }

cd "${script_dir}"
env LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64" CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${python_bin}" -u s4_p2_train_bf_pbr.py \
  --train-manifest "${train_manifest}" \
  --val-manifest "${val_manifest}" \
  2>&1 | tee train_obj100_nonoverlap_1000_20_30_50.log
