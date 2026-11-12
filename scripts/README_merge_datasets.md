# Combine exported datasets

From the repository root, with Pillow installed (the `d2tripo` environment works):

```bash
python scripts/merge_datasets.py \
  --inputs data/real_lab_dataset data/synth_lab_dataset data/topdown_dataset data/vmrd_2.5pct \
  --output data/combined_dataset \
  --cross-split-duplicates prefer-heldout
```

For a read-only audit before creating the output, add `--dry-run` to the command.
The output must be a new directory. Inputs are only read, and source images are
copied, including the targets of any image symlinks. A temporary output is renamed
into place only when complete. The script does not load a model or use a GPU.

## Classes

Classes are matched by **name**, ignoring capitalization and repeated/leading/
trailing whitespace. Source numeric IDs can be completely different. Both
object labels and grasp semantic labels are remapped. Box coordinates, object
row order, adjacency direction/entries, grasp corners, difficulty, and grasp
instance IDs are preserved. The shared class always has the same output ID.

VMRD names retain IDs 0–30 regardless of input order. New names are appended in
input registry order. The output `classes.json` is the ordered combined registry.
For example, `plate` and `scissors` can both have source ID 31 in different
exports and still become different correctly labeled output classes. Conversely,
`bowl` gets one combined ID even if its source ID differs between exports.

Each input needs `classes.json` (a name list), or a `conversion_report.json`
containing the converter's `class_ids` name-to-ID map. The local VMRD conversion
already has the latter. Unknown class IDs and missing registries are errors;
the script never guesses class order. A custom registry can be supplied with:

```bash
python scripts/merge_datasets.py --inputs path/to/first path/to/second \
  --class-map path/to/first=path/to/first_classes.json \
  --output data/combined_v2
```

`--class-map` is repeatable. JSON may be an ordered name list or a name-to-ID
object with contiguous zero-based IDs.

Different words are kept separate: `can` and `cans` do not automatically match.
To intentionally merge synonyms, supply a JSON file such as
`{"can": "cans", "squeeze_bottle": "squeeze bottle"}` using `--aliases FILE`.
Aliases map directly to canonical names; they are not recursively expanded.
VMRD's canonical names cannot be renamed by aliases.

## Splits and duplicates

The source `images`, `labels`, `adj`, and `grasps` train/val/test layout is preserved. Older sources without a test directory
are accepted; their samples are not reassigned to test. Test selections from new
labeler exports remain held out. The merge does not resplit the combined data,
so its overall ratios depend on the input datasets.
Output samples receive unique numeric stems across all sources and splits, so
repeated names like `000001.png` cannot overwrite each other. Other supported
image formats (BMP, TIFF, WebP) are converted to RGB PNG; JPEG/PNG bytes are copied.

The script hashes image file bytes and checks for identical files spanning
any two splits (including test). `--cross-split-duplicates` controls the result:

- `error` (default): stop and identify the conflicting samples.
- `prefer-heldout`: retain copies in test first, then validation, then training;
  exclude identical copies from lower-priority splits.
- `prefer-val`: compatibility alias for `prefer-heldout` (also protects test).
- `keep`: retain both with an explicit warning and a record in the manifest.

The requested local inputs contain one such pair inside VMRD: training image
`10945.jpg` and validation image `03733.jpg`. The example command keeps the
validation image and excludes the training duplicate in the combined output.
Original VMRD files are unchanged. Duplicates within one split are retained and
counted. Hashing detects identical file bytes; it does not group related camera
views or recognize identical pixels in differently encoded files.

The output `manifest.json` includes source paths, all class-ID mappings, input
and retained counts, original filenames/metadata, hashes, exclusions, and output
sample IDs. This provides the provenance for every merged annotation.

## Grasp supervision

Available grasp annotations are always copied and their semantic IDs remapped.
Missing grasp files become empty output files; no grasps are fabricated.

`--grasp-policy auto` is the default. If any input declares grasp supervision
false or any sample lacks grasps, the combined manifest sets `tasks.grasps=false`.
`main.py --dataset_root ...` reads this and disables grasp supervision for the
entire combined run. This is the supported mode for VMRD mixed with your
unlabeled-grasp lab exports. VMRD grasps remain available in the merged files.

Use `--grasp-policy require` only when every source is intended for grasp
training and every sample contains grasp labels. It fails on unsupervised
inputs. `--grasp-policy disable` explicitly disables grasp training, retaining
any existing annotations. Mixed per-image grasp supervision is not implemented.

## Verified result for the current local inputs

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| real_lab_dataset | 52 | 8 | 15 |
| synth_lab_dataset | 76 | 11 | 22 |
| topdown_dataset | 73 | 11 | 21 |
| vmrd_2.5pct | 105 | 11 | 0 |
| **Combined** | **306** | **41** | **58** |

Total: **405 images and 52 classes**, with grasp training disabled. These
counts describe the exports present on November 7, 2026; later exports can differ.
Every output image and annotation bundle was checked against its source in a
temporary merge, and all 58 test samples were read through the dataset loader.
Replacing the subset with full `data/vmrd` also passed a read-only audit:
4,405 train, 475 validation, and 58 test images after excluding the one VMRD
training duplicate described above.

Train using the same fine-tuning setup as the labeling guide, with the merged
root substituted:

```bash
python main.py --dataset_root data/combined_dataset \
  --pretrain_model_path checkpoints/checkpoint0110.pth \
  --finetune_ignore class_embed --no-pred-train \
  --output_dir results/combined_finetune
```

The class count is inferred from the combined registry. Preserve the source
checkpoint's model/backbone settings. Reinitializing the semantic heads avoids
size mismatches with the original checkpoint. This merge does not itself start
training. VMRD supplies most samples; the merge does not rebalance source ratios.

Run merge regression tests in `d2tripo`:

```bash
python -m unittest discover -s tests -p 'test_merge_datasets.py'
python -m unittest discover -s tests -p 'test_label_splits.py'
```
