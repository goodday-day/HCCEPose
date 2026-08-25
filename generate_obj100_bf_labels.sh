#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/yaohua/miniconda3/envs/hccepose/bin/python"
gpu_id="${1:-1}"

[[ -x "${python_bin}" ]] || { echo "Missing HccePose Python: ${python_bin}" >&2; exit 1; }

cd "${script_dir}"
env LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64" CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${python_bin}" -u s4_p1_gen_bf_labels.py \
  2>&1 | tee generate_obj100_bf_labels.log
