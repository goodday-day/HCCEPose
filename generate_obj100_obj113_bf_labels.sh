#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/yaohua/miniconda3/envs/hccepose/bin/python"
data_root="/home/yaohua/PTV3/Pointcept/data/rtless/bop"
gpu_id="${1:-1}"

cd "${script_dir}"
for obj_id in $(seq 100 113); do
  dataset_path="${data_root}/obj${obj_id}"
  [[ -d "${dataset_path}" ]] || { echo "Skipping obj${obj_id}: dataset missing"; continue; }
  echo "Generating BF labels for obj${obj_id}"
  env LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64" CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${python_bin}" -u s4_p1_gen_bf_labels.py \
    --dataset-path "${dataset_path}" --obj-id "${obj_id}"
done 2>&1 | tee generate_obj100_obj113_bf_labels.log
