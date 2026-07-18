#!/usr/bin/env bash
#SBATCH --job-name=verl-online-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH hetjob
#SBATCH --job-name=verl-online-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# One human-approved heterogeneous hold: one dynamically selected normal node
# with eight peer-visible trainer GPUs, plus one dynamic normal env GPU node.
set -euo pipefail
trap 'exit 0' TERM INT
while true; do
  sleep 300 &
  wait $!
done
