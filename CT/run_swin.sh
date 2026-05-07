#!/bin/bash
#SBATCH --job-name=swin_ct
#SBATCH --output=/work/users/g/s/gsonw/BIOS740/final_project/CT/logs/%x_%j.out
#SBATCH --error=/work/users/g/s/gsonw/BIOS740/final_project/CT/logs/%x_%j.err
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00


set -eo pipefail

PROJECT_DIR="/work/users/g/s/gsonw/BIOS740/final_project/CT"
ENV_NAME="pytorch_env"
DEFAULT_MODEL_PATH="/work/users/g/s/gsonw/BIOS740/final_project/cardiac_anatomy_segmentation/outputs/swin_unetr/best_swin_unetr_ct_cps.pth"
DEFAULT_DATA_ROOT="/work/users/g/s/gsonw/BIOS740/final_project/cardiac_anatomy_segmentation/train_data/label_data"
MODE="${1:-train}"
MODEL_PATH="${2:-$DEFAULT_MODEL_PATH}"
DATA_ROOT="${3:-$DEFAULT_DATA_ROOT}"

mkdir -p "${PROJECT_DIR}/logs"

usage() {
    cat <<EOF
Usage:
  sbatch run_swin.sh train
  sbatch run_swin.sh test [model_path] [data_root]

Examples:
  sbatch run_swin.sh train
  sbatch run_swin.sh test /path/to/best_swin_unetr_ct.pth
  sbatch run_swin.sh test /path/to/best_swin_unetr_ct.pth /path/to/test_data
EOF
}

if [[ "${MODE}" != "train" && "${MODE}" != "test" ]]; then
    echo "Error: mode must be 'train' or 'test'."
    usage
    exit 1
fi

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
# python - <<'PY'
# try:
#     import einops  # noqa: F401
# except ImportError:
#     raise SystemExit(
#         "Missing dependency: einops. Install it in pytorch_env with "
#         "`python -m pip install einops` before running SwinUNETR."
#     )
# PY
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "==== Loaded modules ===="
module list

echo "==== GPU check ===="
nvidia-smi || true

CMD=(python main.py --mode "${MODE}")

if [[ "${MODE}" == "test" ]]; then
    CMD+=(--model_path "${MODEL_PATH}" --data_root "${DATA_ROOT}")
fi

echo "==== Launch command ===="
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Job finished at $(date)"
