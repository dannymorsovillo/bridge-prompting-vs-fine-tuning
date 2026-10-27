#!/bin/bash -l
#SBATCH --job-name=bridge-qlora
#SBATCH --partition=ghx4
#SBATCH --account=bihk-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qlora-%j.out
#SBATCH --error=logs/qlora-%j.err

# Submit from ~/IT299; create logs/ before sbatch.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the IT299 directory}"
test -f bridge_qlora_deltaai.ipynb
# Keep model caches and training outputs on work storage.
# Confirm this project directory exists before submitting.
test -d /work/nvme/bihk || { echo "Missing /work/nvme/bihk; verify project storage." >&2; exit 1; }
export BRIDGE_STORAGE_ROOT="${BRIDGE_STORAGE_ROOT:-/work/nvme/bihk/$USER/IT299}"
export HF_HOME="/work/nvme/bihk/$USER/hf_cache"
mkdir -p "$BRIDGE_STORAGE_ROOT" "$HF_HOME"
test -w "$BRIDGE_STORAGE_ROOT"
module load python/miniforge3_pytorch/2.11.0
conda activate base
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Do not override Slurm's assigned CUDA_VISIBLE_DEVICES.
srun python -u scripts/run_qlora_batch.py "$@"
