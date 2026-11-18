# Training on WPI Turing

`train_turing.sbatch` runs this repository on one GPU through Slurm. Defaults:
6 CPU cores, 32 GB host RAM, 12 hours on `short`, batch size 1, 4 data workers,
AMP, offline W&B, and no legacy machine-specific pretrained path. Learning
rates and epoch count otherwise come from `main.py`; specify them per run.
Submit from the repository root on Turing.

## Prepare once

1. Obtain Turing access, then transfer the intended source snapshot, exported
   dataset, and checkpoint to storage accessible from compute nodes. Datasets
   and weights are ignored by Git and must be transferred separately. A clone
   excludes uncommitted source changes. Keep a dedicated run checkout unchanged
   while jobs run.
2. Create a Linux Python environment on Turing with compatible CUDA-enabled
   PyTorch, torchvision, and `requirements.txt` dependencies. The main README
   gives a legacy Python 3.8 / PyTorch 1.13.1 / CUDA 11.6 recipe; requirements
   alone do not pin a reproducible compatible stack. Check `module avail cuda`
   and choose a toolkit compatible with your PyTorch build.
3. In an interactive **GPU compute allocation**, activate that environment,
   load the same CUDA module used for training, and build both extensions:

   ```bash
   sinteractive --partition=short --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=0-01:00:00
   module load cuda  # or your chosen versioned module
   source /absolute/path/to/venv/bin/activate
   # From the repository root:
   (cd models/dab_deformable_detr/ops && python setup.py build install)
   (cd cuda_op && python setup.py build install)
   (cd models/dab_deformable_detr/ops && python test.py)
   ```

   The deformable attention build requires a visible GPU. Rebuild or set an
   appropriate `TORCH_CUDA_ARCH_LIST` if subsequent jobs use different GPU
   architectures. Keep toolkit/compiler/PyTorch versions consistent. Do not
   copy workstation binaries. The torchvision backbone may download pretrained
   ResNet weights on first use; populate its cache if compute internet is limited.

## Submit fine-tuning

All example paths are placeholders for paths **on Turing**. The dataset needs
`images`, `labels`, and `adj` directories with `train` and `val` splits, plus its
class registry and metadata. Export proximity annotations to a new dataset;
the labeler's `project.json` is not a training export.

```bash
cd /absolute/path/to/D2TriPO-DETR
export VENV_PATH=/absolute/path/to/venv
# Optional: export CUDA_MODULE=your/versioned/cuda/module
sbatch scripts/train_turing.sbatch \
  /absolute/path/to/exported_dataset \
  /absolute/path/to/checkpoints/turing-proximity \
  --pretrain_model_path /absolute/path/to/checkpoint0110.pth \
  --finetune_ignore class_embed adj_embed \
  --lr 1e-5 --lr_backbone 1e-6 --epochs 40
```

This example initializes the current **three-class adjacency branch** from a
binary checkpoint. `adj_embed` reinitializes adjacency layers; `class_embed`
reinitializes object/grasp classifiers for the new class registry. Omit either
keyword only if those weights match your architecture and class IDs. Dataset
metadata controls grasp supervision; add `--no-grasps` for boxes/relationships
training if needed. Without a checkpoint, task heads start fresh.

For Conda, load its module and activate before `sbatch`, leave `VENV_PATH`
unset, and export `PYTHON_BIN` as the absolute environment Python path (check
`which python`). `PROJECT_ROOT` overrides the repository location, which defaults
to `SLURM_SUBMIT_DIR`. Avoid `sbatch --export=NONE`; settings must reach the job.

## Resume and monitor

Resume a checkpoint from the **same architecture and class mapping**:

```bash
sbatch scripts/train_turing.sbatch \
  /absolute/path/to/exported_dataset \
  /absolute/path/to/checkpoints/turing-proximity \
  --resume /absolute/path/to/checkpoints/turing-proximity/checkpoint.pth \
  --epochs 60

squeue --me
tail -f slurm-d2tripo-train-JOBID.out
scancel JOBID
```

`--epochs` is the total target, not additional epochs. Resume restores optimizer
and scheduler state; use `--resume_reset_lr --lr ... --lr_backbone ...` only for
intentional rate changes. Checkpoints save after each training epoch; a timeout
can lose the unfinished epoch. Resubmission/resume is manual. Logs go to the
submission directory; checkpoints/config/metrics go to `OUTPUT_DIR`. Use a new
output directory for each independent experiment.

Override Slurm resources **before** the script name and training options after
the two paths:

```bash
sbatch --time=0-02:00:00 --mem=48G scripts/train_turing.sbatch \
  /absolute/path/to/exported_dataset /absolute/path/to/new-output --epochs 2
```

Keep one node, one task, and one GPU; the script uses single-process training
and preserves Slurm's GPU mapping. Check `idle_gpus` and `scluster` on Turing
before choosing hardware. Add `sbatch --account=...` if your allocation requires
it. Resource defaults are starting points, not measured Turing requirements.

Official references: [submission scripts](https://docs.turing.wpi.edu/getting-started/writing/),
[GPU modules](https://docs.turing.wpi.edu/software/cuda/),
[Python environments](https://docs.turing.wpi.edu/software/python/), and
[Slurm options and monitoring](https://docs.turing.wpi.edu/getting-started/slurm_guide/).
