#!/bin/bash
#SBATCH --job-name=Gson_AJSQ
#SBATCH --output=/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/logs/%x_%j.out
#SBATCH --error=/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/logs/%x_%j.err
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# sbatch run_AJSQ_gpu.sh
# sbatch run_AJSQ_gpu.sh train consistency
# sbatch run_AJSQ_gpu.sh test

set -eo pipefail

PROJECT_DIR="/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2"
ENV_NAME="pytorch_env"
MODE="${1:-train}"
CONSISTENCY="${2:-normal}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

echo "==== Clean module environment ===="
module purge

echo "==== Load Anaconda module ===="
module load anaconda/2024.02

echo "==== Initialize conda ===="
eval "$(conda shell.bash hook)"

echo "==== Activate conda environment: ${ENV_NAME} ===="
conda activate "${ENV_NAME}"

set -u

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "==== Loaded modules ===="
module list

echo "==== GPU check ===="
nvidia-smi || true

# python main.py "${MODE}"
python main.py "${MODE}" "${CONSISTENCY}"

echo "Job finished at $(date)"