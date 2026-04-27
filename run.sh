#!/bin/bash
#SBATCH --job-name=llmffn
#SBATCH --partition=h100x
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/HOME/pxyai/pxyaih_0028/Performance01/wlh/llmffn/logs/%x-%j.out
#SBATCH --error=/HOME/pxyai/pxyaih_0028/Performance01/wlh/llmffn/logs/%x-%j.err

set -euo pipefail

cd ~/Performance01/wlh/llmffn
mkdir -p logs

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Current dir: $(pwd)"
echo "=============================="

# source ~/.bashrc
module load CUDA/12.4

source /HOME/pxyai/pxyaih_0028/anaconda3/etc/profile.d/conda.sh
conda activate /HOME/pxyai/pxyaih_0028/Performance01/conda/envs/llmffn

export CUTRACER_SO="$HOME/Performance01/wlh/third_party/CUTracer/lib/cutracer.so"

echo "Python:"
which python3
python3 --version

echo "CUTracer:"
which cutracer

echo "CUDA tools:"
which nvcc
which ptxas
which nvdisasm

echo "GPU:"
nvidia-smi

echo "Start running program..."
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 24 \
  --device-map auto \
  --cutracer-so "$CUTRACER_SO" \
  --no-data-timeout-s 120 \
  --no-dump-cubin \
  --processed-preview-lines 50 \
  --delete-raw-trace-after-postprocess

echo "Program finished."
echo "End time: $(date)"
