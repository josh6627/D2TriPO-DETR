# VMRD V2 conversion

The converted dataset is in `data/vmrd`. **Conversion is incomplete for 33 source
samples with ambiguous annotations.** All 4,683 split IDs have original images,
XML and grasp TXT files. Complete, unambiguous samples were converted as follows:

| Split source | Output split | Expected | Converted | Withheld | Objects | Grasps |
|---|---|---:|---:|---:|---:|---:|
| `trainval.txt` | `train` | 4,233 | 4,205 | 28 | 14,518 | 97,307 |
| `test.txt` | `val` | 450 | 445 | 5 | 1,519 | 10,351 |

## Rerun

From `/home/josh/Robotics/MER-Lab/D2TriPO-DETR`:

```bash
conda run -n d2tripo python scripts/convert_vmrd.py --root data/vmrd
conda run -n d2tripo python scripts/convert_vmrd.py --root data/vmrd --check
```

The first command creates missing outputs; the second checks existing outputs
without writing. Both return **exit status 2** when source samples are withheld,
even when all converted outputs pass integrity checks. Status 0 means no samples
were withheld. Status 1 means a fatal conversion/output error. Existing output
annotations must match exactly; differing files are never overwritten. Source
files are never changed. The converter uses only Python's standard library and
the already installed Pillow. No model or GPU imports are needed.

## Formats and evidence

The authoritative local consumers are `datasets/vmrd.py`,
`datasets/transforms.py`, `models/dab_deformable_detr/matcher.py`, and the
classification, grasp and adjacency losses in
`models/dab_deformable_detr/dab_deformable_detr.py`.

Object labels contain one row per XML `<object>`, in original XML document order:

```text
class_id xmin ymin xmax ymax image_width image_height
```

Coordinates are the original absolute XML pixel coordinates, without subtracting
one, adding one, resizing, rounding or normalizing. The loader reads `xyxy`
directly, builds size as `[height, width]`, and transforms later convert boxes to
normalized `cxcywh`. XML `<filename>` may name a historical image; split ID and
XML stem identify the actual image in `JPEGImages`.

Class IDs are **zero based**, following `robot/inference.py:draw_rectangle`:

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | banana | 16 | paper |
| 1 | watch | 17 | cup |
| 2 | pliers | 18 | mobile phone |
| 3 | toothbrush | 19 | wrist developer |
| 4 | wrench | 20 | bottle |
| 5 | towel | 21 | headset |
| 6 | knife | 22 | toothpaste |
| 7 | badminton | 23 | glasses |
| 8 | tape | 24 | remote controller |
| 9 | charger | 25 | card |
| 10 | pen | 26 | mouse |
| 11 | box | 27 | cans |
| 12 | umbrella | 28 | screwdriver |
| 13 | apple | 29 | notebook |
| 14 | wallet | 30 | stapler |
| 15 | shaver | | |

The model has 32 class channels, but the repository's named VMRD classes occupy
0–30. No background row or fabricated class 31 is added. The matcher and losses
index class channels directly, for both objects and grasps.

Grasp rows contain **11 fields**:

```text
x1 y1 x2 y2 x3 y3 x4 y4 difficulty object_instance_id class_id
```

Original source rows contain eight coordinates, instance ID, then difficulty.
The converter preserves coordinate strings, vertex order, grasp row order and
instance IDs; it moves difficulty to the third field from the end and appends the
owner object's semantic class. The owner is found using XML `<index>`, never
`index - 1` or row position. The loader reads difficulty at `parts[-3]` (`e` → 1,
`h` → 2) and semantic class at `parts[-1]`. It does not expose the instance ID in
its returned tensors, but that association remains in the converted text.
Unknown characters would map to 0 in the loader; those samples are withheld
because their intended difficulty is unknown. Object-level `<difficult>` is
separate and is not consumed by this loader; it remains in the original XML.

Grasp angle is calculated by the loader from rectangle center toward the midpoint
of vertices 2 and 3, then folded into (-90, 90]. Transforms calculate width and
height from successive rectangle edges after coordinate normalization. Therefore
vertices must not be reordered or replaced with a five-parameter rectangle.
The [dataset authors' annotation description](https://gr.xjtu.edu.cn/zh/web/zeuslan/dataset)
defines the original eight coordinates, instance association and e/h labels
(the original page now redirects; its indexed description was consulted).

Adjacency is an `N × N` integer matrix with rows and columns in XML object order.
`A[i,j] = 1` means object at row `i` is above object at row `j` and must be grasped
first: **child → father**. A `<father>` reference on object i creates edge i→father;
a `<children>` reference on object j confirms edge child→j. The two edge sets
must agree. The diagonal and absent edges are zero. No symmetric or transitive
edges are added. Repeated identical references collapse into the same binary
entry and are reported as warnings (`03291`, `03391`). This direction follows
Section III-B of the [D2TriPO-DETR paper](https://www.softintel.top/papers/2026/pmh_TII.pdf),
which defines position (m,n) as m above n, and is consistent with
`robot/inference.py:grasp_order` selecting the starting row of a longest path.
The adjacency loss requires binary targets aligned to the original object rows.

## Integrity checks and unresolved source issues

Every converted file was reread and compared with its source-derived contents.
Checks cover split uniqueness and disjointness, complete file sets, image dimensions,
class ranges, finite numbers, positive in-bounds object boxes, valid grasp owners,
nonzero grasp edges, adjacency size, reciprocal XML relationships and acyclicity.
All converted samples passed. The real `CustomDataset` and both train/val transform
pipelines were also checked on CPU for `00022`; no model was loaded or run.

`00022` is 1008 × 756 with classes `[30, 27, 13]` in XML order (stapler, cans,
apple), 17 grasps (15 easy, 2 hard), and adjacency:

```text
0 1 1
0 0 0
0 0 0
```

Visual inspection confirms the stapler is above the can and apple. Its first
converted grasp ends with `e 1 30`.

There are 417 original grasp rows with corners outside the image; 416 occur in
converted samples. They are finite, nondegenerate annotations and are preserved
without clipping, which would distort the grasp. All instances are listed as
warnings. These are source geometry warnings, not proof of annotation accuracy.

The 33 withheld samples have these issues (categories overlap):

- 27 images have grasps referencing nonexistent XML object indices.
- `00748`, `03886`, `03917` contain duplicate XML instance indices.
- `00722`, `03549` have inconsistent father/children relationships; `03549` also
  contains self references.
- `00561` has difficulty `r`; `00593` has difficulty `w`. No reliable e/h mapping
  was established.

Withheld train IDs:
`00057, 00561, 00593, 00722, 00748, 01457, 02029, 02030, 02031, 02033, 02035,
02036, 02037, 02038, 02039, 02040, 02041, 02042, 02439, 02695, 02698, 03359,
03549, 03886, 03917, 04049, 10069, 10100`.

Withheld val IDs: `02027, 02028, 02034, 02945, 03050`.

`data/vmrd/conversion_report.json` records converted IDs, counts, warning locations
and the exact reasons/line numbers for every withheld sample. No files in any of
the four processed directories are emitted for withheld samples. Original split
files are unchanged; this converted subset must not be described as the complete
official split. Resolving these samples requires corrected authoritative
annotations or an explicit annotation policy; ownership and labels were not guessed.

## Output layout

```text
data/vmrd/
├── images/{train,val}/*.jpg  # retained images, regular files after cleanup
├── labels/{train,val}/*.txt
├── adj/{train,val}/*.txt
├── grasps/{train,val}/*.txt
└── conversion_report.json
```

Image hard links avoided duplicate image storage during cleanup. The repository's hardcoded
`/data/VMRD/...` paths, source code and checkpoint were left unchanged. No inference,
training, model evaluation or GPU benchmarks were performed.
