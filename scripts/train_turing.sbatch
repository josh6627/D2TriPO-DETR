#!/bin/bash -l
#SBATCH --job-name=d2tripo-train
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ $# -lt 2 || ${1:-} == --help ]]; then
    echo "Usage: sbatch scripts/train_turing.sbatch DATASET_ROOT OUTPUT_DIR [main.py arguments...]"
    echo "Submit from the repository root; activate your environment first or set VENV_PATH."
    exit 2
fi
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this script with sbatch to run on a compute node." >&2
    exit 2
fi

# Slurm copies this script to its spool; its location is not the repository.
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
[[ -f main.py ]] || { echo "Set PROJECT_ROOT to the repository on Turing." >&2; exit 2; }
dataset_root=$1
output_dir=$2
shift 2
for split in train val; do
    for kind in images labels adj; do
        [[ -d $dataset_root/$kind/$split ]] || {
            echo "Missing dataset directory: $dataset_root/$kind/$split" >&2
            exit 2
        }
    done
done

# Choose a versioned module matching the environment and compiled extensions.
module load "${CUDA_MODULE:-cuda}"
if [[ -n ${VENV_PATH:-} ]]; then
    source "$VENV_PATH/bin/activate"
fi
python_bin=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-offline}

"$python_bin" - <<'PY'
import torch
import MultiScaleDeformableAttention
import sort_vertices

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: check the GPU allocation and PyTorch environment.")
if torch.cuda.device_count() != 1:
    raise SystemExit("This script requires exactly one visible GPU; request --gres=gpu:1.")
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}")
PY

mkdir -p "$output_dir"
printf 'Job %s on %s; dataset=%s; output=%s\n' "$SLURM_JOB_ID" "${HOSTNAME:-unknown}" "$dataset_root" "$output_dir"
# main.py interprets SLURM_PROCID as distributed training. Suppress launch
# markers for this single Python child, preserving CUDA_VISIBLE_DEVICES.
exec env -u SLURM_PROCID -u WORLD_SIZE -u RANK -u LOCAL_RANK \
    "$python_bin" -u main.py \
    --dataset_root "$dataset_root" --output_dir "$output_dir" \
    --batch_size 1 --num_workers 4 --amp --no-pred-train "$@"
