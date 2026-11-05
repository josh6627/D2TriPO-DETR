# MetaGraspNet conversion

From the repository root, with NumPy and Pillow installed:

```bash
python scripts/convert_metagraspnet.py
python scripts/convert_metagraspnet.py --check
```

Defaults: source `data/MetaGraspNetRaw`, destination `data/metagraspnet`, 80/20
train/validation split by scene, seed 42. The source may be an extracted archive
with nested directories or the directory directly containing `scene*` folders.
RGB images are linked, preserving PNG pixels. The source must remain available.
Rerunning is safe: existing outputs must match. Use a fresh destination when
changing conversion options. `--dry-run` audits without writing.

The output has the same four directories as `data/vmrd`, each with `train` and
`val` subdirectories. Filenames have numeric stems because the loader calls
`int(image_id)`. `conversion_report.json` maps stems back to source scenes/views,
lists classes per instance, and records skipped views and rejected grasps.

## Class mapping

The converter imports the exact 0–30 class order from `convert_vmrd.py`.
`metagraspnet_classes.json` contains all upstream semantic category IDs and an
editable mapping to VMRD class names. Boxes, bottles, cans, cups, screwdrivers,
knives, and markers are grouped into their corresponding VMRD categories.
Unmatched categories share ID **31**, labeled `novel` in the conversion report.
This uses the remaining slot in the model's 32-class heads. **The repository
does not explicitly define a novel-object convention; ID 31 is an assumption,
and the existing checkpoint is not thereby trained to recognize novel objects.**
No per-object novel classes are added, and background mask ID 0 is excluded.

Category IDs come from the [upstream category table](https://github.com/maximiliangilles/MetaGraspNet/blob/master/models/ifl_labels.py).
Pass `--class-map path/to/mapping.json` to adjust the mapping using the same JSON
schema; targets must be one of the existing class names or `novel`.

## Annotation conventions and limitations

- Labels: `class xmin ymin xmax ymax width height`, in original image pixels.
  Boxes enclose the visible instance mask with exclusive upper bounds. Amodal
  boxes are not synthesized. Fully invisible objects are recorded in the report
  and omitted from labels and adjacency rows.
- Adjacency: rows follow increasing source instance IDs, matching label rows.
  `adj[above, below] = 1`, using `-source_order.T`, as in the
  [upstream layout visualizer](https://github.com/maximiliangilles/MetaGraspNet/blob/master/Scripts/visualize_layout.py).
  Rows/columns are selected using `instance_id - 1`; no transitive edges are added.
- Grasps: `x1 y1 x2 y2 x3 y3 x4 y4 difficulty instance_id class_id`.
  MetaGraspNet provides two contact points, illustrated by its
  [2D visualizer](https://github.com/maximiliangilles/MetaGraspNet/blob/master/Scripts/visualize_2d.py),
  rather than VMRD rectangles. These become opposite jaw-edge midpoints; the
  angle computed by the model loader follows the contact axis. Rectangle
  height defaults to **20 source pixels**, configurable with `--grasp-height`.
  This height is a conversion parameter, not measured ground truth. Corners
  are preserved even if the expanded rectangle extends beyond the image.
- Source grasps lack instance IDs. The rounded midpoint's visible instance
  assigns each grasp; background/out-of-image midpoints and invalid contact
  pairs are rejected and reported. This is a heuristic: review associations
  for objects with holes or overlapping objects. No nearest-object guesses
  are made. Suction annotations are not parallel-jaw grasp targets.
- VMRD easy/hard difficulty is unavailable. Default `u` maps to 0 in
  `CustomDataset.map_difficulty`. Use `--difficulty e` or `h` only if you intend
  to assign that constant to every grasp.
- A view requires its own mask, grasp file, and order CSV. Other camera views
  are skipped rather than inheriting another view's 2D annotations. A sample
  with invalid annotations or no assignable grasps is withheld (exit status 2).

To load the converted data directly:

```python
from datasets.vmrd import CustomDataset, make_vmrd_transforms

root = "data/metagraspnet"
split = "val"
dataset = CustomDataset(
    f"{root}/images/{split}", f"{root}/labels/{split}",
    f"{root}/adj/{split}", f"{root}/grasps/{split}",
    transform=make_vmrd_transforms(split),
)
```

For training via `main.py`, set `dataset_root` in `datasets/vmrd.py:build` to
`data/metagraspnet` as described by the repository's training instructions.

Run converter regression tests with:

```bash
python -m unittest discover -s tests -p 'test_convert_metagraspnet.py'
```
