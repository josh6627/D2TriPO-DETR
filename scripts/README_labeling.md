# Label the top-down renders

## Start a separate project with model predictions

`prelabel_dataset.py` runs inference and writes a **new** project compatible with
the existing labeler. It does not change the labeler, existing annotations, or
source images. Use a flat folder containing only the new images you want to label.

```bash
conda activate d2tripo
python prelabel_dataset.py \
  --images path/to/new_images \
  --project data/labels/new_images/project.json
python scripts/label_dataset.py \
  --images path/to/new_images \
  --project data/labels/new_images/project.json
```

Defaults use `checkpoints/real-cluttered-lab_finetune/checkpoint.pth` and the exact
ordered registry in `data/real_cluttered_lab/classes.json`. All 43 class names and
their order must match the checkpoint's saved `class_names`; otherwise the script
stops. Classes are never sorted, renamed, appended, or inferred from detections.
Another training registry or existing project can be supplied with
`--classes-from PATH`; another checkpoint with `--checkpoint PATH`.

Existing output projects are never overwritten. To try a few images first, use
`--limit 5` and a separate trial output path. Other images in the folder remain
available as empty samples in the labeler. Run in your GPU environment: the
model's compiled deformable-attention operator requires CUDA. The default device
is `cuda`; use `--device cuda:1` to select another GPU.

Predicted boxes/classes and above→below relationships are editable immediately.
Every image starts incomplete with relationships unreviewed. Missing objects
still need adding; incorrect or duplicate detections need fixing/removing.
The defaults are `--score-threshold 0.4 --relation-threshold 0.5 --topk 100`.
Degenerate boxes and unnamed classifier channels are skipped. Relationships are
considered in descending confidence order, rejecting edges that would form a
cycle (including contradictory pairs); skipped counts are printed per image.

Grasps are not generated. The new project defaults to boxes/relationships mode;
pass `--require-grasps` if you want to add grasps manually before completion.
Review and mark images complete, select test images, and export normally.
Old and new exports can be combined with the dataset merge script; keep test
images held out. Keeping this class registry preserves the meaning of the
existing checkpoint's class channels and does not itself require retraining.

## Launch the manual labeler

From the repository root:

```bash
conda activate d2tripo
python scripts/label_dataset.py
```

This opens a local Tk desktop app using Pillow. It needs no model, GPU, server,
or network connection. The existing `d2tripo` environment contains Tk and Pillow.
The app uses the Python/Tk environment you launch it with; it does not switch
interpreters or renderers automatically.
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

Or uncheck **Require grasps for completion** at the top of the sidebar. This setting
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

The classic layout has navigation and completion buttons at the top, the image
on the left, and all labeling controls in one scrollable sidebar on the right.

Type part of a name in **Search object classes** to filter the class picker
(case insensitive). The first match is selected automatically; use the picker
for other matches, or press Enter/Down from the search box to open it. **Clear**
shows all classes again. This also searches custom classes without changing IDs.

Text defaults to 16 points, with larger section headings and canvas labels.
The original native theme and fonts are retained.
For larger text, launch with `--font-size 18`. The controls panel has a vertical
scrollbar so all actions stay accessible with larger fonts or smaller windows.

The background image is cached during selection and annotation edits. Zoom is
limited to 8 million displayed pixels and 8,192 pixels per side to bound memory
use; original image pixels and exported coordinates are unchanged. On 1920×1080
images this limits zoom to about 196%. Undo retains the last 100 edits per image.
Drag previews reuse a single solid outline and coalesce pointer movements to
about 30 frames per second. Release always saves the current mouse position,
independently of the last preview frame. Cancel, image changes, zoom, and closing
the app cancel pending preview callbacks. Unchanged test-progress text and
cursors are not rewritten during ordinary pointer motion or selection.

A rotating diagnostic log is saved as `labeler.log` beside `project.json`.
It records image loads, selections, resizes, zooms, and callback errors; it keeps
up to three files of 2 MiB each. Override its location with `--log-file PATH`.
See [freeze investigation](LABELER_FREEZE_DIAGNOSIS.md) for the November 6 findings.

Switch tools with **S** (select/resize), **B** (new box), **R** (redraw selected
box), and **G** (new grasp). Switching cancels an unfinished shape or resize.
Shortcuts are inactive while typing in search fields, dropdowns, or dialogs;
click the image to return keyboard focus to labeling.

The **Confirm relationships** button beside **Mark complete** checks and saves
the same review flag as **Relationships reviewed** in the sidebar, including
scenes with no relationships. Its text changes to **Relationships confirmed**.

1. Choose a class, select **New box [B]**, and drag a tight axis-aligned box around
   each object. **Add class** appends a new named class immediately. Use the
   object list or **Select [S]** to select an existing box; repeated clicks
   cycle overlapping boxes. Change its class with **Apply class to selected object**, or use
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

For multiple exports or mixing lab images with VMRD, use the
[dataset merge script](README_merge_datasets.md). It reconciles classes by name
and gives each output sample a unique numeric filename.

Choose test images using **Use this image for TEST** above the image. The
choice is saved immediately, independently of annotation completion and undo.
Marked images show `[TEST]` in the image dropdown. You can plan test selections
before labeling; only completed images count toward an export.

The live counter shows the percentage of **completed images** assigned to test,
the target count, and whether more or fewer selections are needed. It turns green
and says **TEST TARGET REACHED** at the desired count. It also shows the number
marked across the whole project, including unfinished images. The target updates
as more images become complete. The labeler never chooses test images randomly.

Defaults are **70% training, 10% validation, 20% test**. For 100 completed images,
choose 20 for test; the other 80 are randomly split into 70 train and 10 val.
Use **Split ratios...** to edit the three percentages; settings persist in the
project. Counts round to the nearest whole image (half up), with at least one
image in each split. Very small exports can therefore differ from the requested
percentages. The counter's target count is authoritative.

**Export completed** asks for a parent folder and a new dataset folder name.
At least **three** completed images are required, and the number of completed
test selections must match the displayed target. Export validates every sample
and writes `train`, `val`, and `test` folders. Test membership is exactly your
selection. Training/validation membership is shuffled with seed 42 by default,
so repeated exports of the same completed images and choices are reproducible.
Existing output directories are never overwritten; choose a new folder after
edits. Existing projects receive the new defaults without changing annotations.

CLI export uses the saved test choices and split ratios:

```bash
python scripts/label_dataset.py --export data/topdown_labeled_v2 --seed 42
```

CLI ratio overrides can also be saved with `--val-fraction 0.1 --test-fraction 0.2`;
the training fraction is the remainder. `manifest.json` records requested
fractions and actual split counts. Training continues to use only `train` and
`val`; test stays held out. The dataset builder supports `build_dataset('test',
args=...)` with evaluation transforms for later testing.

Custom input/project:

```bash
python scripts/label_dataset.py --images path/to/images --project path/to/project.json
```

The output contains `images`, `labels`, `adj`, and `grasps`, each with `train`,
`val`, and `test` folders. Numeric output filenames satisfy the loader's integer image
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
