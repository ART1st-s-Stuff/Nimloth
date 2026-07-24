#!/usr/bin/env bash

# Populate an associative array with the GPU count allocated to each node in a
# `scontrol show job -dd` response. Slurm may compress nodes with equal GRES,
# for example `Nodes=dgx-[40,48] ... GRES=gpu:4`.
nimloth_load_slurm_gpu_counts() {
  local job_details=${1:?missing Slurm job details}
  local output_name=${2:?missing output array name}
  local -n output=${output_name}
  local node_expression gpu_count node

  while read -r node_expression gpu_count; do
    [[ -n "${node_expression}" && -n "${gpu_count}" ]] || continue
    while read -r node; do
      [[ -n "${node}" ]] && output["${node}"]=${gpu_count}
    done < <(scontrol show hostnames "${node_expression}")
  done < <(
    sed -n \
      's/^[[:space:]]*Nodes=\([^[:space:]]*\).*GRES=gpu:\([0-9][0-9]*\).*/\1 \2/p' \
      <<< "${job_details}"
  )
}
