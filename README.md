# D2TriPO-DETR

> **Dual-Decoder Triple-Parallel-Output Detection Transformer**

This repository is an official implementation of the paper [D2TriPO-DETR: Dual-Decoder Triple-Parallel-Output Detection Transformer](https://anonymous.4open.science/w/TII-D2TriPO-DETR/).

<p align="center">
  <img src="https://github.com/Embodied-Soft-Intelligence/TII/blob/main/picture/2.png" alt="Fig. 3" />
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Abstract](#abstract)
- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference / Demo](#inference--demo)
- [Real-World Experiments](#real-world-experiments)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Authors & Contact](#authors--contact)
- [Acknowledgements](#acknowledgements)
- [Changelog](#changelog)

---

## Abstract

Vision-based grasping, though widely employed for industrial and household applications, still struggles with object stacking scenarios. Current methods face three major challenges: 1) limited inter-object relationship understanding, 2) poor grasping adaptation across different viewpoints, and 3) error propagation. Inspired by distributed perception and visual adaptation in the human visual attention system, we propose D2TriPO-DETR, a dual-decoder transformer with three outputs, of which are object detection, manipulation relationship, and grasp detection, to address the above challenges. Specifically, a distributed attention perception module and a rotation attention invariance module are designed to address limited inter-object relationship understanding and poor grasping adaptation across different viewpoints. These two modules are respectively integrated into the two parallel decoders to output the triple results simultaneously, partly eliminating task-level error propagation. The evaluation of the Visual Manipulation Relationship Dataset shows that D2TriPO-DETR outperforms existing state-of-the-art methods across all metrics, e.g., a 5.3% higher precision in object detection, a 6.7% image accuracy improvement in manipulation relationship, and a 1.5% higher accuracy in grasp detection. Real-world testing confirms D2TriPO-DETR's effectiveness in grasping stacked objects.

---

## Features

- End-to-end transformer-based architecture
- Dual decoders generating three parallel outputs
- Modules for distributed attention and viewpoint adaptation
- Training and evaluation recipes compatible with Deformable-DETR-style codebases
- Real-robot validation (UR3 + RealSense + Robotiq gripper)

---

## Main Results

<div align="center">

### Object Detection Performance

<table>
<tr>
<td valign="top">

#### Object Detection (VMRD)

<table> <tr><th align="left">Model</th><th align="right">Recall (%)</th><th align="right">Precision (%)</th></tr> <tr><td>VSE[26]</td><td align="right">89.2</td><td align="right">90.2</td></tr> <tr><td>GrRN[20]</td><td align="right">91.9</td><td align="right">94.8</td></tr> <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="right"><strong>98.0</strong></td><td align="right"><strong>97.2</strong></td></tr> </table></td><td valign="top" style="padding-left: 40px;">

#### Manipulation Relationship (VMRD)

<table> <tr><th align="left">Model</th><th align="right">OR (%)</th><th align="right">OP (%)</th><th align="right">IA (%)</th></tr> <tr><td>Mutli-Task CNN[8]</td><td align="right">86.0</td><td align="right">88.8</td><td align="right">67.1</td></tr> <tr><td>VMRN-RN101[4]</td><td align="right">85.4</td><td align="right">85.5</td><td align="right">65.8</td></tr> <tr><td>VMRN-VGG101[4]</td><td align="right">86.3</td><td align="right">88.8</td><td align="right">68.4</td></tr> <tr><td>GVMRN-RF-RN101[27]</td><td align="right">86.9</td><td align="right">87.5</td><td align="right">68.8</td></tr> <tr><td>GVMRN-RF-VGG101[27]</td><td align="right">88.7</td><td align="right">89.5</td><td align="right">70.2</td></tr> <tr><td>Adj-Net-RN50[7]</td><td align="right">88.9</td><td align="right">91.5</td><td align="right">75.0</td></tr> <tr><td>Adj-Net-RN101[7]</td><td align="right">89.8</td><td align="right">91.5</td><td align="right">77.3</td></tr> <tr><td>GrRN[20]</td><td align="right">91.2</td><td align="right">93.1</td><td align="right">78.0</td></tr> <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="right"><strong>92.3</strong></td><td align="right"><strong>94.3</strong></td><td align="right"><strong>84.7</strong></td></tr> </table></td> </tr> </table>

### Detailed Relationship Analysis

<table> <tr> <td valign="top">

#### Manipulation Relationship IA-X (VMRD)

<table> <tr><th align="left">Model</th><th align="right">IA-2 (%)</th><th align="right">IA-3 (%)</th><th align="right">IA-4 (%)</th><th align="right">IA-5 (%)</th></tr> <tr><td>Mutli-Task CNN[8]</td><td align="right">87.7</td><td align="right">64.1</td><td align="right">56.6</td><td align="right">72.9</td></tr> <tr><td>GVMRN-RF-RN101[27]</td><td align="right">91.4</td><td align="right">69.2</td><td align="right">61.2</td><td align="right">57.5</td></tr> <tr><td>GVMRN-RF-VGG101[27]</td><td align="right">92.9</td><td align="right">70.3</td><td align="right">63.8</td><td align="right">60.3</td></tr> <tr><td>Adj-Net-RN50[7]</td><td align="right">87.3</td><td align="right">74.5</td><td align="right">69.8</td><td align="right">72.6</td></tr> <tr><td>Adj-Net-RN101[7]</td><td align="right">88.7</td><td align="right">75.2</td><td align="right">75.0</td><td align="right">76.7</td></tr> <tr><td>GrRN[20]</td><td align="right">92.3</td><td align="right">76.6</td><td align="right">74.5</td><td align="right">74.3</td></tr> <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="right"><strong>93.8</strong></td><td align="right"><strong>82.8</strong></td><td align="right"><strong>81.1</strong></td><td align="right"><strong>87.1</strong></td></tr> </table></td><td valign="top" style="padding-left: 40px;">

#### Grasp detection (VMRD)

<table> <tr><th align="left">Model</th><th align="right">Accuracy (%)</th></tr> <tr><td>FCGN[28]</td><td align="right">54.5</td></tr> <tr><td>VMRN[3]</td><td align="right">70.5</td></tr> <tr><td>SE-ResUNet[29]</td><td align="right">81.2</td></tr> <tr><td>GR-ConvNet[30]</td><td align="right">82.7</td></tr> <tr><td>TFGrasp[31]</td><td align="right">83.3</td></tr> <tr><td>SKGNet[32]</td><td align="right">86.1</td></tr> <tr><td>SPANet[33]</td><td align="right">86.7</td></tr> <tr><td>EGNet[5]</td><td align="right">87.1</td></tr> <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="right"><strong>88.6</strong></td></tr> </table></td> </tr> </table>

### Cross-Dataset Performance

<table> <tr> <td valign="top">

#### Grasp detection (Cornell)

<table> <tr><th align="left">Model</th><th align="center">Input</th><th align="right">Accuracy (%)</th></tr> <tr><td>TFGrasp[31]</td><td align="center">RGB</td><td align="right">96.78</td></tr> <tr><td>TFGrasp[31]</td><td align="center">RGBD</td><td align="right">97.99</td></tr> <tr><td>DSNet[34]</td><td align="center">RGBD</td><td align="right">98.31</td></tr> <tr><td>Yang[35]</td><td align="center">RGB</td><td align="right">98.74</td></tr> <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="center">RGB</td><td align="right"><strong>98.87</strong></td></tr> </table></td> </tr> </table></div>

> All D2TriPO-DETR experiments reported here used a total batch size of 4 (see code for implementation details).

---

## Real-World Experiment Results

<div align="center"> <table> <tr> <th>Models</th> <th>Cluttered Scenes (%)</th> <th>Stacked Scenes (%)</th> </tr> <tr> <td>Mutli-Task CNN[8]</td> <td align="center">90.60</td> <td align="center">65.65</td> </tr> <tr> <td>SMTNet[35]</td> <td align="center">86.13</td> <td align="center">65.00</td> </tr> <tr> <td>EGNet[5]</td> <td align="center">93.60</td> <td align="center">69.60</td> </tr> <tr> <td><strong>D2TriPO-DETR (Ours)</strong></td> <td align="center"><strong>95.00 (95% CI: 87.8%-98.0%, n=80)</strong></td> <td align="center"><strong>91.25 (95% CI:85.9%-94.7%, n=160)</strong></td> </tr> </table> </div>

---

## Requirements

- OS: Linux (recommended)
- CUDA >= 11.6
- Python >= 3.8
- PyTorch >= 1.13.1
- torchvision >= 0.14.1
- Additional: git, cmake, GCC compatible with CUDA
- Optional: 4 × NVIDIA GPUs (the authors used 4 × RTX 4090 in their experiments)

---

## Installation

```bash
conda create -n d2tripo python=3.8 -y
conda activate d2tripo
conda install pytorch=1.13.1 torchvision=0.14.1 cudatoolkit=11.6 -c pytorch -y
pip install -r requirements.txt
```

## Compiling CUDA operators

```bash
cd ./models/dab_deformable_detr/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
# rorate
cd ./cuda_op
python setup.py install
```
---

## Usage
### Evaluating a labeled dataset

`evaluate_dataset.py` requires a dataset folder with `images/<split>`,
`labels/<split>`, and `adj/<split>`. It supports JPG, JPEG, and PNG images;
grasp annotations are not required. For example:

```bash
python evaluate_dataset.py --dataset-root data/topdown_labeled --checkpoint results/topdown_finetune/checkpoint.pth
python evaluate_dataset.py --dataset-root data/vmrd
python evaluate_dataset.py --dataset-root data/real_lab_dataset --split test --checkpoint results/combined_finetune/checkpoint.pth
```

Use `--split both` to evaluate train and val in one run, loading the model once
and saving separate results for each split:

```bash
python evaluate_dataset.py --dataset-root data/topdown_labeled --split both
```

This uses the default checkpoint without fine-tuning. Use `--checkpoint` to
select other weights. The split names describe the dataset folders; neither
split is used to update model weights during evaluation.

Use `--split test` for your held-out test images, `--split val` for validation
(the default, including VMRD), or `--split train` for only train. The selected
split must exist; the script reports a missing or empty split before loading the model.
Use
`--limit 10` to evaluate up to 10 images per split, or `--output-dir results/custom` to change the
output directory. CSV results and error logs are saved alongside the visualization
images, for example `results/visualizations/topdown_labeled_val/results.csv`
and `errors.txt`. Evaluation uses CUDA.
The existing metrics require matching object classes and boxes. Separate
`bbox_*` CSV columns score relationships using one-to-one box matches at IoU
>= 0.5, ignoring predicted object classes. Both endpoints must match and the
arrow direction must be correct. Missing boxes still cause missed relationships;
arrows involving unmatched detections count as false positives. The summary
prints box-only micro precision, recall, F1, and the fraction of images with
all boxes and relationships correct (ignoring classes). Both sets of metrics
use the same confidence-filtered predictions.
Precision is TP / (TP + FP), recall is TP / (TP + FN), and image accuracy
is entirely correct images / evaluated images. Class-aware image accuracy
requires all classes, boxes, and relationships to be correct; box-only image
accuracy requires all boxes and relationships to be correct. These are
calculated separately for each evaluation mode. `results.csv` includes per-image
precision, recall, F1, and image accuracy, with `bbox_` prefixes for box-only
scores. `summary.csv` saves separate `class_aware` and `box_only` rows with
dataset scores computed from summed counts. CSV scores range from 0 to 1;
the console shows percentages. Undefined scores are blank in CSV and `N/A`
in the console. Failed images are excluded from both modes.
Choose a checkpoint trained with the dataset's class IDs, including any added
classes; the default checkpoint is `checkpoints/checkpoint0110.pth`.

Evaluation also saves annotated prediction images to
`<output-dir>/visualizations/<dataset-name>_<split>/`: one `_relationships.jpg` image
with object boxes, class names, confidence scores, and
arrows pointing from upper objects to lower objects. A side panel lists every
predicted relationship as `#ID class above #ID class` with its confidence score;
the IDs match the box labels. Images with no predicted relationships say so.
These use the same drawing
style as `infer_unlabeled_dataset.py` and reuse the evaluation forward pass.
They show predictions without ground-truth filtering, using object confidence
above 0.4 and relationship probability above 0.5. Class names come from the
checkpoint when available. Use `--no-visualizations` to save only metrics and logs.
Set `--visualization-dir /path/to/images` to choose a folder for images, CSV results,
and error logs; dataset/split subfolders are added beneath it. With
`--no-visualizations`, `results.csv` and `errors.txt` still use the same folders.

### Dataset Preparation

Count annotated objects, instances per class, and directed relationships without
loading a model or using a GPU:

```bash
python scripts/dataset_stats.py --dataset-root data/vmrd
python scripts/dataset_stats.py --dataset-root data/real_lab_dataset --split test
python scripts/dataset_stats.py --dataset-root data/real_lab_dataset --csv counts.csv
```

The script reports each existing train/val/test split and combined totals,
including registered classes with zero instances. It reads `classes.json` or
the class mapping in `conversion_report.json`; use `--classes PATH` to supply
a registry explicitly. Without a registry, it reports numeric class IDs.
Instances are appearances across images, not unique physical objects.
Relationships are saved directed adjacency entries equal to 1 (above → below),
without adding inferred edges. Orphan annotations are ignored with a warning;
missing or malformed annotations prevent totals from being reported. CSV export
contains per-class counts by split and requires a new output filename.

To combine exported lab datasets with VMRD while reconciling class IDs by name,
use `scripts/merge_datasets.py`. See the [dataset merge guide](scripts/README_merge_datasets.md)
for the command, class mappings, duplicate handling, and mixed grasp supervision.

For unlabeled images in `data/topdown_renders`, run
`python scripts/label_dataset.py` in the `d2tripo` environment. The desktop
labeler supports object boxes, rotated grasps, manipulation relationships,
saved progress, and new named classes. See the [labeling guide](scripts/README_labeling.md)
for export and training with `--dataset_root` and automatic class counts.
To label boxes and relationships without grasps, use
`python scripts/label_dataset.py --no-grasps`; training automatically reads
this choice from the exported dataset.

To convert the extracted MetaGraspNet real-world data in `data/MetaGraspNetRaw`
to this model's dataset layout, run `python scripts/convert_metagraspnet.py`.
See [MetaGraspNet conversion](scripts/README_metagraspnet.md) for class mapping,
novel objects, grasp conversion assumptions, and loading the resulting dataset.

Please download [VMRD](https://gr.xjtu.edu.cn/zh/web/zeuslan/dataset) dataset and [Cornell](https://journals.sagepub.com/doi/abs/10.1177/0278364914549607) dataset, then organize them as following:

To extract a smaller percentage of converted VMRD while preserving its train/validation
proportions, see the [VMRD subset guide](scripts/README_subset_vmrd.md).

```
code_root/
└── data/
    └── vmrd/
        ├── adj/
        	├── train
        	└── val
               └──00001.txt
        └── grasps/
        	├── train
        	└── val
               └──00001.txt
        └── images/
        	├── train
        	└── val
               └──00001.jpg
        └── labels/
        	├── train
        	└── val
               └──00001.txt
        └── xml/
        	├── train
        	└── val
               └──00001.xml
```

---

### Training

For GPU batch training on WPI's Turing cluster, see the
[Turing setup and submission guide](scripts/README_turing.md) and
[`scripts/train_turing.sbatch`](scripts/train_turing.sbatch).

To train the baseline D2TriPO-DETR on a single node with 4 GPUs for 110 epochs, first modify the dataset path in `./datasets/vmrd.py`, then download the [resnet101_doubledetr](https://drive.google.com/file/d/1AHhbgWUhpmpR2t3Q4XSuO6FvX-na-u5E/view?usp=drive_link) weight file and place it in the project root directory, and finally run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py
```
110 epoch training takes around 15 hours on a single machine with 4 RTX4090. To ease reproduction of our results we provide [results](https://drive.google.com/file/d/1zzWPDbZ9_iApHUBZeaO5p6-Rs48jCv4G/view?usp=drive_link) for the best checkpoint.pth for 110 epoch, achieving 88.6 accuracy.
We train D2TriPO-DETR with the AdamW optimizer, using a learning rate of 1e-4 for the transformer and 1e-5 for the backbone, and a weight decay of 1e-4. Training uses a batch size of 4 for 110 epochs, with the learning rate dropped at epoch 80. Data augmentation includes horizontal flipping, random scaling, and random cropping. The transformer uses a dropout of 0.1, and gradient clipping with a maximum norm of 0.1 is applied to stabilize training.

The command for training on a single GPU is as follows:

```bash
python main.py
```
---

## Real-World Experiments

This repository includes the code used for **real-world robotic grasping experiments**.  
The code path for the real-world experiment is `./robot/inference.py`. To perform experiments using an actual robotic arm, run:

```bash
python inference.py
```

The script `predict.py` connects to a **UR robotic arm** equipped with an **Intel RealSense D435** depth camera and a **Robotiq 2F-85** parallel gripper.  It performs **real-time grasp detection and execution** using the trained D2TriPO-DETR model.  Before running, ensure that the robot, gripper, and camera are properly connected and configured.

---

## Change Log

See [log](https://drive.google.com/file/d/1QNxz4lBXenbfMJAcSCJOY6_A_xyxbAom/view?usp=drive_link) for detailed logs of major changes.

## License

This project is released under the [D2TriPO-DETR](https://mok1170.github.io/TII/). See `LICENSE` for details.

---

