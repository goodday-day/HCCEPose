#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/yaohua/miniconda3/envs/hccepose/bin/python"
data_root="/home/yaohua/PTV3/Pointcept/data/rtless/bop"
split_dir="${script_dir}/../Pointcept/data_splits"
train_manifest="${split_dir}/raw_depth_nonoverlap_1000_20_30_50_train_seed2026.csv"
val_manifest="${split_dir}/raw_depth_nonoverlap_1000_20_30_50_val_seed2026.csv"
gpu_id="${1:-1}"
exp_root="${script_dir}/exp/obj100_nonoverlap_1000_20_30_50"
seen_object_ids=(100 101 102 104 105 106 107 108 109 110 112 113)

cd "${script_dir}"
mkdir -p "${exp_root}"
for obj_id in "${seen_object_ids[@]}"; do
  train_count=$(awk -F, -v id="${obj_id}" 'NR > 1 && $3 == id { count++ } END { print count + 0 }' "${train_manifest}")
  val_count=$(awk -F, -v id="${obj_id}" 'NR > 1 && $3 == id { count++ } END { print count + 0 }' "${val_manifest}")
  if (( train_count == 0 || val_count == 0 )); then
    echo "Skipping obj${obj_id}: CSV has train=${train_count}, val=${val_count} samples"
    continue
  fi
  dataset_path="${data_root}/obj${obj_id}"
  [[ -d "${dataset_path}/train_pbr_xyz_GT_front" && -d "${dataset_path}/train_pbr_xyz_GT_back" ]] || { echo "Missing BF labels for obj${obj_id}" >&2; exit 1; }
  experiment_dir="${exp_root}/obj${obj_id}"
  mkdir -p "${experiment_dir}"
  echo "Training obj${obj_id}: train=${train_count}, val=${val_count}"
  env LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64" CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${python_bin}" -u s4_p2_train_bf_pbr.py \
    --dataset-path "${dataset_path}" --obj-id "${obj_id}" \
    --train-manifest "${train_manifest}" --val-manifest "${val_manifest}" \
    --batch-size 24 --num-workers 16 \
    --save-dir "${experiment_dir}" \
    2>&1 | tee "${experiment_dir}/train.log"
done
