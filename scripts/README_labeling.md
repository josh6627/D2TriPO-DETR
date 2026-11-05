# Label the top-down renders

From the repository root:

```bash
conda activate d2tripo
python scripts/label_dataset.py
```

This opens a local Tk desktop app using Pillow. It needs no model, GPU, server,
or network connection. The existing `d2tripo` environment contains Tk and Pillow.
On Linux, if its Tk build only supports bitmap fonts, the launcher checks
`/usr/bin/python3` for Tk, Pillow, and scalable fonts and uses that renderer
automatically. Project paths and command options are preserved. The training
environment is unchanged. You can also launch directly with
`/usr/bin/python3 scripts/label_dataset.py --no-grasps` on this machine.
Run it from a graphical desktop (or SSH with X forwarding). A display connection
is required; `--export` works without a display. For another environment, install
Pillow and the Python Tk bindings supplied by your Python distribution.

The default input is `data/topdown_renders`: 300 numbered renders. The summary
image `density_map.png` is excluded. Edits save after every action to
`data/topdown_labels/project.json`; reopen the same command to resume. Source
images are only read. Back up the project JSON along with the images.

## Label and train without grasps

Launch with:

```bash
python scripts/label_dataset.py --no-grasps
```

Or uncheck **Require grasps for completion** in the **Grasps** tab. This setting
applies to the whole project and persists when reopening. You can mark images
complete using object boxes and reviewed relationships alone. If no objects
overlap, leave relationships empty and check **Relationships reviewed**; export
creates an all-zero adjacency matrix. Existing grasp annotations remain saved.

Export normally. The export's `manifest.json` records that grasp supervision is
disabled, so `main.py --dataset_root data/topdown_labeled` automatically trains
only object detection and relationships. The fine-tuning command below works
unchanged. For another dataset without this metadata, pass `--no-grasps` to
`main.py` explicitly. Missing grasp folders are allowed in this training mode.

Grasp matching and all grasp losses are skipped, including auxiliary losses.
The model still computes grasp outputs, but this run does not train them; grasp
predictions, especially for new classes, should not be treated as learned from
this dataset. This mode uses the default `dab_deformable_detr` architecture;
the repository's `dab_detr` alternative is grasp-only and is rejected.

To return to full multitask labeling, check **Require grasps** again. Previously
completed images that lack a grasp for any object become incomplete. Re-export
to a new directory after changing the mode.

## Label an image

The workspace has tools above the image and an inspector with **Objects**,
**Relations**, and **Grasps** tabs. Progress is shown in the header.

Type part of a name in the search field in **Objects** to filter the class picker
(case insensitive). The first match is selected automatically; use the picker
for other matches, or press Enter/Down from the search box to open it. **Clear**
shows all classes again. This also searches custom classes without changing IDs.

Text defaults to 14 points, with larger section headings and canvas labels.
Help and status text use the same readable size, with higher-contrast colors.
For larger text, launch with `--font-size 16`. The controls panel has a vertical
scrollbar so all actions stay accessible with larger fonts or smaller windows.

Switch tools with **S** (select/resize), **B** (new box), **R** (redraw selected
box), and **G** (new grasp). Switching cancels an unfinished shape or resize.
Shortcuts are inactive while typing in search fields, dropdowns, or dialogs;
click the image to return keyboard focus to labeling.

The **Confirm relationships** button beside **Mark complete** checks and saves
the same review flag as **Relationships reviewed** in the **Relations** tab, including
scenes with no relationships. Its text changes to **Relationships confirmed**.

1. Choose a class, select **Box [B]**, and drag a tight axis-aligned box around
   each object. **Add class** appends a new named class immediately. Use the
   object list or **Select [S]** to select an existing box; repeated clicks
   cycle overlapping boxes. Change its class with **Apply to selected**, or use
   **Redraw [R]** to replace its bounds.
   In **Select [S]** mode, click a box, then drag any of its four
   sides or square midpoint handles to resize it. The opposite side stays fixed;
   bounds stay inside the image. Release to save, Escape to cancel, or Ctrl+Z
   to undo the resize. Grasp positions and object IDs stay unchanged.
2. If **Require grasps** is checked, select an object and **Grasp [G]**. Click the two opposing jaw-contact
   midpoints, then a third point to set rectangle height (distance from the
   contact axis). The preview shows the result; red edges show the two jaw
   faces. Choose Easy or Hard before drawing. Add multiple valid grasps per
   object when appropriate. Delete and redraw an incorrect grasp.
3. For stacked objects, select the upper object, choose the lower object from
   the dropdown, and add the relationship. The arrow points **above → below**,
   meaning grasp the upper object first. Cycles are rejected. Review all pairs
   and check **Relationships reviewed**, including scenes without overlaps.
4. Click **Mark complete**, then **Next unfinished**. Completion requires at
   least one object and reviewed relationships, plus one grasp per object when
   **Require grasps** is checked. Partial
   work remains saved but is excluded from exports. Further edits clear the
   completion flag. Finish all visible objects and valid relationships before
   marking complete; the app cannot judge annotation accuracy from pixels.

Use **Undo** / Ctrl+Z for edits on the current image; history resets when changing
images. Escape cancels an unfinished shape. The +/- controls zoom; scrollbars
pan the image. **Fit image** scales the image to the available workspace. Grasp corners may extend beyond the image, matching the existing
VMRD convention; object boxes must stay inside it. Adding classes never changes
an existing class ID. Deleting an object removes its grasps and relationships.

## Export

**Export dataset** asks for a parent folder and a new dataset folder name.
At least two completed images are required. Export validates all completed
samples, copies RGB images into a portable dataset, and creates a deterministic
80/20 train/validation split. Existing output directories are never overwritten.
Re-export to a new folder after edits. CLI export supports another split/seed:

```bash
python scripts/label_dataset.py --export data/topdown_labeled --val-fraction 0.2 --seed 42
```

Custom input/project:

```bash
python scripts/label_dataset.py --images path/to/images --project path/to/project.json
```

The output contains `images`, `labels`, `adj`, and `grasps`, each with `train`
and `val` folders. Numeric output filenames satisfy the loader's integer image
ID requirement; `manifest.json` maps them back to source filenames and instance
IDs. `classes.json` is an ordered JSON list, where position is semantic class ID.
The app preserves VMRD's 31 names at IDs **0–30**. New names start at **31**, then
32, 33, etc. ID 31 here is a specific new class, not a generic novel category;
do not combine it with the MetaGraspNet converter's grouped `novel` ID 31
without reconciling the class registries first.

Annotation files use original pixel coordinates and these exact formats:

- Labels: `class_id xmin ymin xmax ymax image_width image_height`.
- Grasps: `x1 y1 x2 y2 x3 y3 x4 y4 difficulty instance_id class_id`.
  Corners 2/3 face contact B, so the loader's angle follows the contact axis.
  Class IDs are looked up from the owner at export, so reclassifying an object
  also reclassifies its grasps.
- Adjacency: an N×N binary matrix aligned with object label rows;
  `A[upper,lower]=1`. No transitive edges are automatically inserted.

The split is by image. If several images depict the same scene, arrange scene
separation before measuring generalization; this tool does not infer scene IDs.

## Train with new classes

`main.py` now accepts `--dataset_root` and infers both semantic class heads from
`classes.json`, with a minimum of 32 channels for compatibility. You can override
with `--num_classes N` (at least the registry length). Existing training commands
and checkpoints without these settings retain 32 channels. Names and class count
are saved in checkpoint arguments; `infer_unlabeled_dataset.py` reads those names
for predictions from the resulting checkpoint.

For fine-tuning the existing checkpoint on the exported labels:

```bash
python main.py --dataset_root data/topdown_labeled \
  --pretrain_model_path checkpoints/checkpoint0110.pth \
  --finetune_ignore class_embed --no-pred-train \
  --output_dir results/topdown_finetune
```

Use the model/backbone settings matching your source checkpoint. The example
initializes both semantic classifiers anew (the `class_embed` keyword matches
object and grasp heads), while loading the remaining weights. This avoids size
mismatches when adding class channels. New classes require labeled examples and
training; the old checkpoint does not recognize them just because a name was
added. This command does not start training until you run it.

Use `--resume` only for a checkpoint with the same class registry and head sizes,
including `--dataset_root` again. `--no-pred-train` disables the legacy hardcoded
`pred_train_dir` load; explicit `--pretrain_model_path` / `--resume` also take
precedence over that load. Preserve the same class ID meaning when mixing data.

Run validation tests (including the real loader/transforms in `d2tripo`):

```bash
python -m unittest discover -s tests -p 'test_label*.py'
```
