# Create a smaller VMRD dataset

Run from the repository root on the **converted** dataset (`images/train`,
`images/val`, and matching `labels`, `adj`, and `grasps` folders):

```bash
python scripts/subset_vmrd.py --input data/vmrd --output data/vmrd_5pct --percent 5 --seed 42 --dry-run
python scripts/subset_vmrd.py --input data/vmrd --output data/vmrd_5pct --percent 5 --seed 42
```

`--percent 5` selects 5% of **each** existing split, preserving the original
train-to-validation ratio as closely as whole images allow. For 4,205 training
and 445 validation images this produces **210 train + 22 validation images**.
Counts are rounded to the nearest integer (halves round up). If a percentage
rounds either split to zero, choose a larger percentage. Percentages must be
greater than zero and at most 100.

Selection is random without replacement; the same source files, percentage,
and seed give the same selection. Change `--seed` to draw a different subset.
Original split membership is retained. Sampling is per split, not per class;
rare classes may be absent from a small subset.

Each selected image is copied with its object labels, adjacency matrix, and
grasp annotations. Image symlinks are copied as real files, so the subset is
portable. The script preserves class IDs and task settings, writes the complete
class registry to `classes.json`, and records source counts and selected IDs in
`subset_report.json`. It uses only the Python standard library and does not
start training. Input files are only read; the output must be a new directory.
If your source is still raw VMRD (`JPEGImages`, `Annotations`, `Grasps`), first
follow [VMRD conversion](VMRD_CONVERSION.md).

Use the new folder with your usual training options:

```bash
python main.py --dataset_root data/vmrd_5pct --output_dir results/vmrd_5pct
```

To reduce VMRD's contribution to a combined dataset, pass `data/vmrd_5pct`
instead of `data/vmrd` to [merge_datasets.py](README_merge_datasets.md).

Regression tests:

```bash
python -m unittest discover -s tests -p 'test_subset_vmrd.py'
```
