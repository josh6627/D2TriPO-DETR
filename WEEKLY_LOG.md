# Weekly log

Week: 2026-W47 (November 16–22, 2026)

## 2026-11-18 — Stop tracking generated results

- Removed 1,628 previously tracked files under `results/` from the Git index using `git rm --cached`; all remain present locally. Existing `results/` ignore rule now applies to them.
- Verified zero tracked result files, all staged-deletion paths still exist, and staged whitespace checks pass. No other files were staged by this task.
- Cleanup is staged but uncommitted/unpushed. Existing Git history still contains these artifacts; future clones of a committed cleanup will omit them from the checked-out tree but ordinary clones retain historical objects.

## 2026-11-18 — Turing training submission

- Added local `scripts/train_turing.sbatch`, a one-node/one-GPU Slurm job with CUDA/environment selection, dataset-directory and GPU/extension preflight checks, AMP, and training argument forwarding.
- Added `scripts/README_turing.md` and README link covering transfer of uncommitted source/data, cluster CUDA builds, binary-to-three-class initialization, submission, monitoring, and manual resume.
- Used current official WPI Turing documentation for Slurm, GPU modules, and environment guidance. Resource defaults remain unmeasured starting points.
- Validation: Bash syntax, mocked launch with spaced paths, argument forwarding, GPU visibility preservation, distributed-marker isolation, missing-data/non-Slurm guards, and whitespace checks passed. No Turing connection, environment installation, actual GPU training, or commit performed.

## 2026-11-18 — Shared project memory

- Added `AGENTS.md` instructions to read this log and append concise entries after substantive tasks.
- Chose brief entries and on-demand archive reading to limit recurring context usage; included weekly rollover and presentation guidance.
- Validation: checked for existing memory files and reviewed repository status. No application code changed; tests were unnecessary.
- Open: existing local code changes and presentation files predate this setup; their purpose and validation have not been reconstructed.

## 2026-11-18 — Lab dataset, fine-tuning, failure analysis, and presentation

- **Data:** collected and labeled **485 lab images** with a labmate (corrected from 440), including classes, bounding boxes, and directed relationships. This is the user-reported collection count, not a verified total for every merged dataset folder.
- **Training:** fine-tuned the pretrained binary-adjacency model, compared checkpoints, and tested lower learning rates. Lower-rate epoch 55 had the best reported class-aware edge F1 on the shared 101-image evaluation set; retain it as the baseline.
- **Diagnosis:** visually reviewed failures. Missed visible and occluded objects were the largest categories; prioritize understanding detection and class errors rather than continuing learning-rate experiments without a specific hypothesis.
- **Adjacency:** three-class head/loss code exists on the separate `proximity-adjacency` branch. The proposed labels now describe **directed manipulation dependencies**, including asymmetric grasp obstruction. Relabeling and retraining remain pending.
- **Artifacts:** created ground-truth visualizations, a ranked failure-review list, and a 28-slide weekly update using the reference decks' WPI template. Checks performed during this conversation are recorded below; repository edits remain local unless otherwise noted.

### Training setup and resume behavior

Dataset: `data/combined_datasets/combined_real_cluttered_lab`. Initial weights: `checkpoints/checkpoint0110.pth`. Initial output: `checkpoints/real-cluttered-lab_finetune-updated`.

Initial settings: model LR `1e-5`, backbone LR `1e-6`, batch size 1, 40 total epochs initially, 4 workers, AMP, and `--no-pred-train`. The run reported `no_grasps=True`, disabling grasp supervision. `--finetune_ignore class_embed` matches both object and grasp classification heads, leaving them freshly initialized. Missing keys for those heads were expected. The empty `Ignore keys: []` startup message is misleading because it is printed before filtering populates the list. Deprecation warnings did not indicate a training failure.

With default `lr_drop=80`, these initial runs did not reach an automatic LR drop. Constant LR was a valid experiment; validation improved initially, then further training became inconsistent. The lower-rate experiment resumed the epoch-50 baseline with model LR `3e-6`, backbone LR `3e-7`, and `--resume_reset_lr`, training to 60 total epochs. Requested output was `checkpoints/real_cluttered-lab_finetune-updated/lower_rate` (underscore after `real`, unlike the initial output path).

Resume with `--resume` rather than reloading pretraining weights or ignoring the heads again. `--epochs` is the total target, not additional epochs. Resume restores optimizer and scheduler state; `--resume_reset_lr` explicitly applies CLI rates and restarts the configured LR schedule while retaining optimizer history. Confirm rates in the startup log. Filenames are zero-based: `checkpoint0049.pth` follows 50 completed epochs; use the exact file evaluated if the user's epoch naming differs.

### Evaluation record

Earlier results used **40 images / 254 objects**. Values below are percentages and were supplied by the user. The lower-rate runs belong to the expanded evaluation only.

| Original-rate epoch | Class-aware precision | Recall | Edge F1 | Image accuracy |
|---|---:|---:|---:|---:|
| 30 | 92.31 | 86.75 | 89.44 | 62.50 |
| 40 | 97.33 | 87.95 | 92.41 | 67.50 |
| 50 | 93.83 | 91.57 | 92.68 | 72.50 |
| 55 | 94.74 | 86.75 | 90.57 | 70.00 |
| 60 | 93.59 | 87.95 | 90.68 | 75.00 |

Expanded evaluation: **101 images / 671 objects / 135 labeled directed edges**, with no failed evaluations in the reported runs. These checkpoints were evaluated on the same expanded set:

| Checkpoint | Class-aware precision | Recall | Edge F1 | Image accuracy | Box-only F1 | Box-only image accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Original-rate epoch 50 | 88.98 | 77.78 | 83.00 | 64.36 | 86.17 | 66.34 |
| Original-rate epoch 60 | 87.72 | 74.07 | 80.32 | 65.35 | 85.94 | 69.31 |
| Lower-rate epoch 55 | **89.08** | **78.52** | **83.46** | 64.36 | **87.40** | 67.33 |
| Lower-rate epoch 60 | 87.93 | 75.56 | 81.27 | 61.39 | 85.26 | 63.37 |

The saved `results/visualizations/combined_real_cluttered_lab_test/results.csv` and `summary.csv` were checked against lower-rate epoch 55: class-aware TP=106, FP=13, FN=29, 28 missing objects, 11 extra detections, and 65/101 fully correct images. Box-only TP=111, FP=8, FN=24. Its class-aware F1 gain over original-rate epoch 50 is only **0.46 percentage points**, corresponding to one additional TP with the same FP count; do not claim a significant improvement or proven overfitting.

IoU means intersection over union; object matches use IoU ≥ 0.5. Class-aware matching also requires matching classes. Box-only matching ignores classes. Micro edge scores pool relationship counts across images; F1 balances precision and recall. Image accuracy measures completely correct images. Different evaluation cohorts cannot establish a training improvement or decline. Because the 101-image set has guided checkpoint selection, treat it as validation despite its `test` folder name; reserve independent scenes for a final test.

### Visual failure review

User's manual **image-level** tally:

| Failure category | Images |
|---|---:|
| Missed visible objects | 10 |
| Missed hidden/occluded objects | 10 |
| Class confusion | 6 |
| Duplicate boxes around the same object | 3 |
| Missed relationships | 3 |
| Extra relationships | 3 |

Categories may overlap; these are not individual object/edge counts. Keep occluded misses distinct from unobstructed misses. Missing objects can also cause missing edges, so do not add these counts as independent failures. Per-image CSVs identify failures; visual review explains scene conditions, class pairs, and possible annotation errors. Inspect object recall by class, size, occlusion, and clutter, plus relationship recall when both endpoints are correctly detected. Use rates with denominators, not raw counts alone.

Selected fully correct examples, ranked by relationship count then object count: **420 (10 objects, 5 edges), 421 (10, 5), 401 (7, 4), 422 (10, 3)**. Failure examples retained: **484** (2 missed objects, 3 missed edges), **434** (all 5 objects detected, scissors–clamp edge missed), **441** (2 missed objects, 1 false edge), **442** (1 missed visible fruit), **456** (fruit beneath scissors missed, along with its edge), and **432** (small USB-labeled prediction at scissors, 1 missed object, 1 extra detection, 1 missed edge). User corrected requested image 465 to **456**. Image 432 also fails box-only matching, so it is not purely a classification error.

Duplicate detections can come from different DETR queries surviving thresholding; the evaluator already deduplicates repeated classes from the same query. Consider overlap filtering only after checking frequency and the risk of suppressing real stacked objects.

### Proposed directed adjacency labeling

Both relationship classes are directional: **`adj[i][j]` describes how row object i affects column object j**.

| Class | Proposed meaning |
|---|---|
| 0 | No outgoing manipulation dependency |
| 1 | Directed overlap/stacking: i is above or occludes j |
| 2 | Directed proximity obstruction: i restricts grasping/manipulating j through clearance limits or potential collision |

For example, `[[0, 2], [0, 0]]` says object 0 obstructs grasping object 1, while the reverse is not true; object 0 may need removal first. Nearby objects are not automatically obstructing. If both types apply to an ordered pair, **class 1 takes priority**. Preserve existing stacking/occlusion labels and ignore self-relations. Annotators must use consistent gripper size, approach direction, feasibility assumptions, and a preferred-grasp convention when multiple grasps are possible. These are approximate visual obstruction labels, not guarantees of physical collision.

Two options discussed: symmetric proximity kept separate from picking-order dependencies, versus the proposed directed grasp-obstruction labels. Simply interpreting symmetric proximity as precedence can introduce cycles. Directed labels are not guaranteed to produce an acyclic or executable plan either. The research question is whether directed obstruction improves picking-order selection and clutter removal.

Read-only branch inspection found `proximity-adjacency` commit `534cec93`: adjacency head output expands from 1 to 3 logits per pair, with cross-entropy replacing binary BCE. This confirms the head/loss changes, not full end-to-end readiness. The user reports those code modifications complete; no further architecture/loss changes were requested or made here. Still needed: relabel images, check matrix/class/direction conventions, verify three-class decoding/evaluation/visualization, retrain, and compare against the binary baseline on the same scenes.

### Repository checks and deliverables

During the earlier training investigation, checkout `main` at `5855d012` matched the startup log; three-class changes were isolated to `proximity-adjacency`. The local model diff only commented out a debug print. A live training process could not be inspected from this session. Existing adjacency activation checkpointing on `main`, batch size 1, and AMP help explain roughly 5 GB tensor-memory usage. The log's `max mem` is peak PyTorch allocated tensor memory, which can differ from `nvidia-smi`. Lower LR and ignored checkpoint heads do not materially explain reduced memory. For concurrent branch work, a separate worktree avoids changing files under a running job; imported Python code usually stays in memory, but later imports/workers/file reads can see changed files.

Created `visualize_ground_truth.py`, reusing the evaluator's box/arrow/panel style with ground-truth labels and no confidence scores. It rendered all 101 images to `results/ground_truth/combined_real_cluttered_lab_test`; 12 existing evaluator tests passed during that change. `evaluate_dataset.py` and `infer_unlabeled_dataset.py` received local renderer support. The ranked list of all 36 failed images is `results/ground_truth/failure_review_epoch55.md`, ordered by FP+FN, then missing+extra objects, with prediction/ground-truth links.

Presentation deliverables in `presentations/weekly_update_2026-11-18/`:

- `Josh Weekly Update - Nov 18.pptx`: **28 slides**.
- `Josh Weekly Updates - Updated Nov 18.pptx`: **70 slides**, preserving the original 42 and appending the update.
- `slide_outline.md`: slide text, sources, and speaker notes.

Used the actual masters/layouts from `/home/josh/Downloads/Josh Weekly Updates.pptx`, with `/home/josh/Downloads/Azura.pptx` as a reference for concise, image-led progress updates. Kept the WPI/Verdana styling, enlarged examples on separate ground-truth/prediction slides, and added failure analysis, the two proximity options, numeric matrices, status, and next steps. No epoch-by-epoch presentation walkthrough. Original reference files were not modified. PowerPoint archives reopened, original slide text and slide bounds were checked, and approximate layout previews used substitute fonts; no native PowerPoint rendering was available. Presentation and visualization artifacts are local working-tree additions, not newly committed work.

## 2026-11-18 — Keep project memory local

- Added root-level `AGENTS.md` and `WEEKLY_LOG.md` to `.gitignore`; both were already untracked, so no index removal was needed.
- Verified ignore rules and whitespace. The existing `proximity-adjacency` branch does not track either path, so ordinary switching between it and `main` should retain the local files. Separate worktrees do not automatically copy them.
- These files have no Git-backed history or backup. A branch that tracks the same paths can replace ignored files; retain an independent backup for recovery. The ignore change is local and uncommitted.

## 2026-11-18 — Three-class adjacency readiness audit

- Audited current `proximity-adjacency` checkout: three-logit head and cross-entropy already support directed classes 0/1/2; no additional architecture/loss change identified for the proposed semantics. No application code changed.
- Focused CPU checks executed actual head/loss methods in isolation: output shape, permuted asymmetric labels against manual CE, finite gradients, empty and partial matches passed. Full model training was not tested.
- Reproduced `inspect_objects.align_relationships` shape failure with three-class output. Inference, evaluation, visualization, and several data validators retain binary assumptions and need updates before end-to-end use.
- Binary checkpoint output-layer sizes differ; initialize through `--pretrain_model_path` with appropriate `--finetune_ignore` adjacency keys rather than direct binary resume. Existing memory tests still construct a one-output head and do not establish three-class readiness. Relabeling/retraining remain pending.

## 2026-11-18 — Three-class evaluation support

- Updated `inspect_objects.py` and `evaluate_dataset.py` locally for directed classes 0/1/2, argmax decoding with softmax confidence, zero self-relations, and typed edge scoring. A wrong relationship type counts as FP plus FN.
- Added per-class counts and summary scores for class-aware and box-only matching; visualization panels identify overlap/stacking versus grasp obstruction. Extra edges count once regardless of class ID.
- Retained binary decoding and strict checkpoint loading by matching adjacency output-layer widths to saved weights. Evaluation predictions use the same detection selection as scoring.
- Validation: all 39 targeted relationship/evaluation tests passed, including new multiclass, checkpoint-loading, CSV, and rendering cases; `git diff --check` passed. Changes remain uncommitted. No trained three-class checkpoint inference or retraining performed; standalone unlabeled/robot inference remains outside this update.

## 2026-11-18 — Directed proximity labeling

- Updated the desktop labeler and project export locally: choose stacking (1) or directed proximity/grasp obstruction (2), with purple solid/orange dashed arrows. Legacy two-ID edges remain stacking; typed edges retain row-to-column direction and export as 0/1/2 matrices.
- Enforced one type per ordered pair and stacking priority. Stacking cycles remain invalid; mutual obstruction is allowed. Relationship edits clear review/completion and support undo; object deletion checks endpoints rather than relationship class IDs.
- Added labeling instructions and export metadata. Existing annotations and split selections were not modified.
- Validation: 31 targeted tests passed, including controller editing, persistence, export and real-loader preservation of class 2; GUI suite passed its render-budget test and skipped eight window tests because no display was available. Whitespace checks passed. Changes remain uncommitted.

## 2026-11-18 — Combined dataset labeling project

- Created ignored local `data/combined_datasets/combined_real_cluttered_lab/project.json`, `labeling_images` links, and `LABELING.md`. Imported all 486 images (343/42/101 train/val/test), 43 classes, 2,271 boxes, and 537 stacking edges using original object IDs from manifest provenance. No grasps were present.
- Marked every image incomplete/unreviewed for proximity annotation. Verified all 1,458 source annotation files stayed unchanged and every imported adjacency matrix matched its source.
- Added fixed imported-split support to the labeler/export, disabling split edits for this project and preserving membership for partial exports. Code changes remain uncommitted.
- Validation: 28 targeted tests and whitespace checks passed. Three real images round-tripped through temporary export with identical pixels/annotation values and original splits. Open using the command in `LABELING.md`; review proximity and export to a new dataset.

## 2026-11-18 — Sidebar wheel scrolling and crash traceability

- Fixed missing wheel handling over right-sidebar labels/controls using a sidebar-only bindtag. Lists keep native scrolling until their boundary; modal grabs are left to Tk. Wheel callbacks do not call redraw, image upload, save, or schedule work.
- Saved exact before/after labeler snapshots, isolated patch, tests and SHA-256 metadata in ignored `results/labeler_sidebar_scroll_2026-11-18/`. Baseline includes earlier proximity rendering and fixed-split changes; changes remain uncommitted.
- Added a startup revision marker and documented comparison artifacts/log locations in `scripts/LABELER_FREEZE_DIAGNOSIS.md`, per the user's request to trace future crashes.
- Validation: 10 tests passed, eight GUI tests skipped without a display; whitespace checks passed. No new crash reported or physical graphics-driver stability test performed. Retain project-adjacent logs if symptoms recur.

## 2026-11-18 — Correct sidebar scrolling startup regression

- User-reported traceback identified a startup error introduced by the wheel patch: its dot-prefixed bindtag was interpreted by Tk as a nonexistent widget path. Corrected the tag with a class-name prefix and moved registration into a directly testable method.
- Added a regression using the reported canvas path and all three wheel bindings. Eleven tests passed; eight GUI tests skipped without a display. Actual window startup remains unverified here; this traceback does not establish a recurrence of the earlier GPU/kernel freeze.
- Updated crash-diagnosis notes and retained both the original failed source and corrected snapshot/patch in the scrolling artifact folder. Startup marker is now `2026-11-18-r2`. Changes remain local and uncommitted; project annotations were not edited.

## 2026-11-18 — Explicit proximity direction

- Defined `#2 → #3` as object #2 obstructing grasping/manipulating object #3; removing #2 may clear access, without guaranteeing a full picking sequence. Reverse obstruction must be labeled separately.
- Updated sidebar source/target instructions, examples, relationship-list wording, labeling guides, and export meanings. Existing edge orientation and arrow rendering remain unchanged; project annotations were not edited.
- Validation: 29 targeted tests passed, including selected source #2/target #3 through display description and exported matrix (class 2 in source row/target column, zero reverse). Whitespace checks passed.
- Preserved a separate direction-clarification source snapshot/patch and added diagnosis notes for crash traceability. Changes remain local/uncommitted; no window stability test performed.

## 2026-11-18 — Next-image crash: confirmed Intel reset recurrence

- User reported a crash after Next image. Previous boot on kernel `7.0.0-42-generic` logged Intel GuC engine-reset failure at 12:16:30 PM EST and forcewake timeouts; no OOM-kill entries found. Final labeler action rendered `000361.png` nine seconds earlier, with no final callback traceback.
- Backed up current project, logs, source and kernel evidence to `results/labeler_freeze_20261118_121630/`. All 486 annotations validate; 350 are complete. Source project was not rewritten.
- Current source matches the saved direction-clarification version; the last image has no proximity edges. Exact trigger remains unknown; no application fix is established.
- Current boot is kernel `-43`; `-38` remains installed. Documented a controlled kernel comparison before further speculative UI changes. No driver/boot configuration or application code changed.
