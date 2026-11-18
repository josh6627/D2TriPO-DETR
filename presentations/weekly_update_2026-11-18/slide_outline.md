# Weekly update — November 18, 2026

28 slides. Original WPI masters, layout, theme, and Verdana fonts preserved. Large image-only viewports crop off baked-in text panels; relationships are repeated as larger editable slide text.

## 1. Dexterous Picking Check-In
Nov 18th 2026

Speaker notes: Weekly update: data collection and labeling, fine-tuning results, visual failure analysis, and the new adjacency representation. Original WPI master and layout are reused directly. The three-class work is distinct from the binary model whose results are presented.

## 2. Current Progress
Collected 485 lab images with my labmate
Labeled object classes, bounding boxes, and relationships
Fine-tuned the model and reviewed its failures
Modified the adjacency code; relabeling and retraining are next

Speaker notes: 485 lab images is the corrected count supplied by the user, replacing the earlier report of 440. The photograph is a representative lab scene, not verified as belonging exclusively to this week’s captures. Adjacency code status is reported by the user, with three-class head/loss changes confirmed on proximity-adjacency branch.

## 3. Labeled Dataset Example
Ground-truth relationships
Green fruit above plate
Orange fruit above plate
lego above wood
Classes, boxes, and directed relationships
Image 000400

Speaker notes: Annotation rows define object IDs. All relationships are drawn from the actual ground-truth adjacency matrix. A[i,j]=1 means i is above j. The evaluation screenshot’s right panel is cropped away and its text is replaced with larger editable slide text. Full scene and annotation pixels are retained. Ground-truth boxes and arrows are not predictions.

## 4. Fine-Tuning the Model
Started from the pretrained D2TriPO-DETR checkpoint
Trained on lab object classes, boxes, and relationships
Used a batch size of 1 with mixed precision
Initial rates: 1×10⁻⁵ for the model, 1×10⁻⁶ for the backbone
Tested lower rates and kept the best-performing checkpoint

Speaker notes: The shown model uses binary adjacency. Class heads were reset using finetune_ignore class_embed; grasp supervision was disabled. Best observed class-aware edge F1 among reported runs is lower-rate epoch 55: rates 3e-6 and 3e-7 after resuming from 50 completed epochs. Later training did not consistently help. The separate proximity-adjacency changes were not used for this run.

## 5. Performance on Lab Images
101 evaluation images · 671 labeled objects · 135 directed relationships
Evaluation | Precision | Recall | Edge F1 | Image accuracy
Class-aware | 89.08% | 78.52% | 83.46% | 64.36%
Box-only | 93.28% | 82.22% | 87.40% | 67.33%
Box-only evaluation ignores classes; both use IoU ≥ 0.5
Lower-rate training gave a small improvement in edge F1
Additional training did not consistently improve performance
Edge F1 pools relationships across images; image accuracy requires a fully correct image.

Speaker notes: Source: lower-rate epoch 55 summary.csv and results.csv, exact totals TP=106 FP=13 FN=29, 65/101 class-aware correct images. Box-only TP=111 FP=8 FN=24, 68/101 correct images. Original epoch 50 class-aware F1=83.00395%; lower-rate epoch 55=83.46457% (+0.46 percentage points). Lower-rate epoch 60=81.27490%. This set has been used for checkpoint selection and functions as validation, despite its test folder name. It is not an untouched final test. No new three-class results are available. Earlier 40-image evaluations are omitted because the cohort changed.

## 6. Successful Inference 1: Image 420
Correct relationships
scissors above bowl
fruit above plate
lego above cans
lego above wood
knife above wood
10 objects detected
5 of 5 relationships correct
Image 000420

Speaker notes: Lower-rate epoch 55. Selected from fully correct images, ranked by class-aware relationship TP count first, then object count. 000420.png: total_objects=10, TP=5, FP=0, FN=0, missing_objects=0, extra_detections=0. Relationship labels are verified against ground truth for this fully correct scene. Two instances can share a class name; prediction query IDs and ground-truth row IDs are independent.

## 7. Successful Inference 2: Image 421
Correct relationships
scissors above plate
fruit above bowl
bowl above box
lego above cans
ball above wood
10 objects detected
5 of 5 relationships correct
Image 000421

Speaker notes: Lower-rate epoch 55. Selected from fully correct images, ranked by class-aware relationship TP count first, then object count. 000421.png: total_objects=10, TP=5, FP=0, FN=0, missing_objects=0, extra_detections=0. Relationship labels are verified against ground truth for this fully correct scene. Two instances can share a class name; prediction query IDs and ground-truth row IDs are independent.

## 8. Successful Inference 3: Image 401
Correct relationships
lego above plate
Right fruit above plate
Left fruit above plate
bowl above wood
7 objects detected
4 of 4 relationships correct
Image 000401

Speaker notes: Lower-rate epoch 55. Selected from fully correct images, ranked by class-aware relationship TP count first, then object count. 000401.png: total_objects=7, TP=4, FP=0, FN=0, missing_objects=0, extra_detections=0. Relationship labels are verified against ground truth for this fully correct scene. Two instances can share a class name; prediction query IDs and ground-truth row IDs are independent.

## 9. Successful Inference 4: Image 422
Correct relationships
banana above bowl
bowl above wood
clamp above plate
10 objects detected
3 of 3 relationships correct
Image 000422

Speaker notes: Lower-rate epoch 55. Selected from fully correct images, ranked by class-aware relationship TP count first, then object count. 000422.png: total_objects=10, TP=3, FP=0, FN=0, missing_objects=0, extra_detections=0. Relationship labels are verified against ground truth for this fully correct scene. Two instances can share a class name; prediction query IDs and ground-truth row IDs are independent.

## 10. Failure Analysis
Manual review: images with each failure mode
Missed visible objects
10
Missed occluded objects
10
Class confusion
6
Duplicate bounding boxes
3
Missed relationships
3
Extra relationships
3
Missed objects were the most common issue in the visual review.
Categories may overlap; these are image counts, not individual error counts.

Speaker notes: User’s manual tally. Hidden/occluded objects were intentionally separated from unobstructed missed objects. Counts may overlap and do not represent an exclusive partition of the 36 failed images. The CSV has four images with missing edges and zero class-aware detection errors; this slide retains the user’s tally of three images instead of silently replacing the visual audit.

## 11. Missed Objects: Ground Truth
Expected relationships
box above fruit
squeeze bottle above ruler
cans above ruler
10 labeled objects
3 labeled relationships
Image 000484

Speaker notes: Actual ground truth for 000484.png. Relationships: box above fruit, squeeze bottle above ruler, cans above ruler. Fruit beneath the box and ruler partly covered by bottle/can are the two objects absent from the prediction, as visible by comparing the annotations to the prediction. The tiny fruit is partially visible through the box region. Keep this slide paired with the next at an identical scale.

## 12. Missed Objects: Model Prediction
What the model missed
Fruit beneath the box
Ruler beneath the bottle and can
No relationships predicted
2 missed objects
3 missed relationships
Image 000484

Speaker notes: Lower-rate epoch 55, 000484.png. total_objects=10, missing_objects=2, extra_detections=0, TP=0 FP=0 FN=3. The image comparison supports the identity of the missing fruit and ruler. Co-occurring missed objects and relationships do not establish the causal contribution of each error without query alignment.

## 13. Missed Relationship: Ground Truth
Expected relationship
Scissors above clamp
5 labeled objects
1 labeled relationship
Image 000434

Speaker notes: 000434.png has five annotated objects. Ground-truth adjacency has the single edge from scissors to clamp. Both endpoints appear in the next slide’s predictions. This illustrates a missed relationship independently of missing or misclassified objects.

## 14. Missed Relationship: Model Prediction
Objects found, edge missed
All 5 objects detected correctly
No extra detections
Scissors–clamp relationship absent
0 of 1 relationships found
Image 000434

Speaker notes: Lower-rate epoch 55, 000434.png: missing_objects=0, extra_detections=0, TP=0 FP=0 FN=1. Box-only metrics are identical. This is a clean example of a missing relation with correct class-aware object detections.

## 15. Image 441: Ground Truth
What is labeled
No relationships annotated
9 labeled objects
9 labeled objects
0 labeled relationships
Image 000441

Speaker notes: User-requested failure example 000441.png. Ground-truth annotation shown at the same scale and crop as its following prediction slide. Focus: Missed Objects and a False Relationship. Images 441 and 442 have no ground-truth edges, so their zero TP/FN counts do not imply successful object detection. All objects remain visible in the full scene. Image 456 replaces 465 following the user correction.

## 16. Image 441: Model Prediction
What went wrong
Orange fruit at the right is missed
Can at the lower right is missed
Bowl-above-can edge is false
2 missed objects
1 false relationship
Image 000441

Speaker notes: Lower-rate epoch 55. 000441.png. Focus: Missed Objects and a False Relationship. Exact CSV: TP=0, FP=1, FN=0, missing_objects=2, extra_detections=0. The named visual issues were checked against ground truth and the saved prediction. For image 432, box-only matching also misses the scissors; its failure is not purely a class mismatch because the USB box covers only a small part of the scissors.

## 17. Image 442: Ground Truth
What is labeled
Orange fruit at the bottom
No relationships annotated
6 labeled objects
0 labeled relationships
Image 000442

Speaker notes: User-requested failure example 000442.png. Ground-truth annotation shown at the same scale and crop as its following prediction slide. Focus: Missed Visible Object. Images 441 and 442 have no ground-truth edges, so their zero TP/FN counts do not imply successful object detection. All objects remain visible in the full scene. Image 456 replaces 465 following the user correction.

## 18. Image 442: Model Prediction
What went wrong
Orange fruit is not detected
Other 5 objects are matched
No relationships predicted
1 missed object
0 relationship errors
Image 000442

Speaker notes: Lower-rate epoch 55. 000442.png. Focus: Missed Visible Object. Exact CSV: TP=0, FP=0, FN=0, missing_objects=1, extra_detections=0. The named visual issues were checked against ground truth and the saved prediction. For image 432, box-only matching also misses the scissors; its failure is not purely a class mismatch because the USB box covers only a small part of the scissors.

## 19. Image 456: Ground Truth
What is labeled
Scissors above fruit
Fruit partly covered by scissors
9 labeled objects
1 labeled relationships
Image 000456

Speaker notes: User-requested failure example 000456.png. Ground-truth annotation shown at the same scale and crop as its following prediction slide. Focus: Missed Occluded Object. Images 441 and 442 have no ground-truth edges, so their zero TP/FN counts do not imply successful object detection. All objects remain visible in the full scene. Image 456 replaces 465 following the user correction.

## 20. Image 456: Model Prediction
What went wrong
Fruit beneath scissors is missed
No relationships predicted
Scissors–fruit edge is missed
1 missed object
1 missed relationship
Image 000456

Speaker notes: Lower-rate epoch 55. 000456.png. Focus: Missed Occluded Object. Exact CSV: TP=0, FP=0, FN=1, missing_objects=1, extra_detections=0. The named visual issues were checked against ground truth and the saved prediction. For image 432, box-only matching also misses the scissors; its failure is not purely a class mismatch because the USB box covers only a small part of the scissors.

## 21. Image 432: Ground Truth
What is labeled
Scissors above clamp
5 labeled objects
5 labeled objects
1 labeled relationships
Image 000432

Speaker notes: User-requested failure example 000432.png. Ground-truth annotation shown at the same scale and crop as its following prediction slide. Focus: Class Confusion and a Missed Relationship. Images 441 and 442 have no ground-truth edges, so their zero TP/FN counts do not imply successful object detection. All objects remain visible in the full scene. Image 456 replaces 465 following the user correction.

## 22. Image 432: Model Prediction
What went wrong
Small USB-labeled box at scissors
Scissors are not correctly detected
Scissors–clamp edge is missed
1 missing · 1 extra object
1 missed relationship
Image 000432

Speaker notes: Lower-rate epoch 55. 000432.png. Focus: Class Confusion and a Missed Relationship. Exact CSV: TP=0, FP=0, FN=1, missing_objects=1, extra_detections=1. The named visual issues were checked against ground truth and the saved prediction. For image 432, box-only matching also misses the scissors; its failure is not purely a class mismatch because the USB box covers only a small part of the scissors.

## 23. Current Adjacency Representation
The current model predicts a binary relationship for each ordered object pair
1 means the row object is above the column object
0 combines nearby objects and objects with no stacking relationship
Example pair | Current label
A above B | 1
A near C | 0
A unrelated to D | 0
Goal: distinguish grasp obstruction from no manipulation dependency.

Speaker notes: Current main code uses one adjacency logit per ordered pair, sigmoid decoding, and binary_cross_entropy_with_logits. This is a conceptual explanation; near does not have a numerical distance rule supplied by the user. Existing binary scores shown earlier do not evaluate proximity. A above B is the current directional meaning; geometric image-box overlap is not automatically a manipulation relation.

## 24. Proximity Direction: Two Options
Stacking has a direction. Physical proximity alone does not.
Option 1: Symmetric proximity
Option 2: Directed obstruction
A is near B, so B is near A
Describes closeness, without telling us which object to move first
Keep proximity separate from picking-order dependencies
A restricts grasping B; B may not restrict grasping A
An edge indicates an outgoing manipulation dependency
Requires consistent gripper and preferred-grasp assumptions
Proposed definition: label directed grasp obstruction.

Speaker notes: User-supplied proposed labeling guidelines distinguish symmetric physical proximity from directed manipulation obstruction. Option 1 records near(A,B) symmetrically and must remain separate from precedence edges; treating both directions as picking-order requirements can create a cycle. Option 2 uses adj[i][j]=2 if object i restricts the feasible gripper approach, clearance, or manipulation of j. This can be asymmetric and may inform which object should be removed first. Obstruction depends on gripper size, approach direction, preferred grasp and scene assumptions; visual labels approximate interference rather than guarantee collision. A dependency graph can still contain cycles if mutual obstruction exists and requires handling by the planner. This slide does not claim directed labels guarantee an acyclic graph or physical feasibility. Directed obstruction is proposed; code changes for the three-class output and loss are already complete. No additional architecture/loss change is requested or performed.

## 25. New Plan: Three Directed Labels
Class | Meaning of row i → column j
0: No dependency | No outgoing manipulation dependency
1: Overlap / stacking | Object i is above or occludes object j
2: Proximity obstruction | Object i restricts grasping or manipulating j
Class 1 takes priority when both types apply to the same pair
Use consistent gripper, approach, and preferred-grasp assumptions

Speaker notes: Exact proposed class mapping from user: 0 no outgoing manipulation dependency; 1 directed overlap/stacking (row object above or occludes column object); 2 directed proximity obstruction (row object restricts column object grasp through insufficient clearance or potential collision). Both 1 and 2 are directional. Preserve existing stacking/occlusion labels; class 1 takes priority for an ordered pair where both apply. Diagonal ignored. Nearby does not automatically mean obstructing; no fixed bounding-box distance is specified. Multiple possible grasps require a consistent preferred-grasp convention. Labels are approximate visual judgments under assumed gripper geometry, approach and feasibility. Research objective is whether these dependencies improve picking-order selection and clutter removal, not an already measured gain. Three-class model code is implemented; relabeling and retraining remain. Existing slides report binary-model results only.

## 26. Example Adjacency Matrices
A is above B; B obstructs grasping C; C does not obstruct B.
Current: binary labels
Proposed: directed labels
From / to | A | B | C
A | – | 1 | 0
B | 0 | – | 0
C | 0 | 0 | –
From / to | A | B | C
A | – | 1 | 0
B | 0 | – | 2
C | 0 | 0 | –
1 = above
0 = no directed stacking edge
1 = overlap / stacking
2 = directed obstruction
0 = no outgoing dependency
Row i affects column j. Self-relationships are ignored. This scene is illustrative.

Speaker notes: Illustrative three-object scene using user’s proposed numeric class mapping. Binary matrix encodes A→B=1 and nothing for the non-stacking obstruction B→C. Proposed matrix preserves A→B=1 and adds B→C=2 while C→B=0. All other off-diagonal entries are 0 in this hypothetical scene. This mirrors the user’s two-object example [[0,2],[0,0]] for B and C. Dashes show ignored self-relations; no inferred new labels are assigned to real images before relabeling. Model output is N×N×3 class scores; decoded class-label matrix is N×N. The edge B→C means B may need removal before C, not that an executable collision-free sequence has been proven.

## 27. Adjacency Update: Current Status
Task | Status
Modify the model and adjacency loss | Code changes complete
Relabel images with the new relationship classes | Next
Retrain on the relabeled dataset | Pending
Evaluate stacking and grasp-obstruction predictions | After retraining
The results shown today are from the existing binary-adjacency model.

Speaker notes: Status: user states code modification is finished but all images need relabeling and model needs retraining. Read-only verification confirms three-output adjacency head and cross-entropy loss on separate proximity-adjacency branch. This does not certify end-to-end inference/labeler compatibility or testing. Relabeling must preserve class IDs, object row ordering and split boundaries. Keep prior binary checkpoint and labels as baseline.

## 28. Next Steps
Apply consistent directed stacking and grasp-obstruction labels
Relabel the images and check adjacency matrix consistency
Retrain the model using the three-class adjacency code
Compare against the current binary-adjacency baseline
Continue investigating missed objects and class confusion

Speaker notes: Relabeling and retraining are upcoming work, not completed results. Apply the proposed 0/1/2 mapping and consistent gripper and preferred-grasp conventions before the labeling pass, and keep the dataset split fixed. Update or verify inference decoding, metrics and visualization for three-class outputs before comparing models. Use stacking-specific comparisons on identical scenes plus separate directed obstruction metrics; a new three-class aggregate cannot be directly equated to old binary edge F1. Keep independent final test scenes because the existing 101-image set has guided tuning.
