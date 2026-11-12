#!/usr/bin/env python3
"""Inspect the existing dataset preprocessing without loading a model.

Examples (run in the d2tripo environment):
    python inspect_preprocessing.py --dataset-root data/real_lab_dataset --split test --image-id 000012
    python inspect_preprocessing.py --dataset-root data/vmrd --split val --batch-size 2
    python inspect_preprocessing.py --dataset-root data/real_lab_dataset --split train --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from datasets import build_dataset
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import collate_fn


def adjacency_report(raw, target, names):
    """Check matrix structure and preservation; indices refer to label-file rows."""
    for stage, labels in (("before", raw), ("after", target)):
        count = len(labels["labels"])
        matrix = labels["adj"]
        if tuple(matrix.shape) != (count, count):
            raise ValueError(f"Adjacency {stage} preprocessing: expected {(count, count)}, "
                             f"got {tuple(matrix.shape)}")
        if not torch.all((matrix == 0) | (matrix == 1)):
            raise ValueError(f"Adjacency {stage} preprocessing must contain only 0 and 1")
        if torch.any(matrix.diagonal() != 0):
            raise ValueError(f"Adjacency {stage} preprocessing contains self-relationships")
    if not torch.equal(raw["adj"], target["adj"]):
        raise AssertionError("Preprocessing changed the adjacency matrix")
    if not torch.equal(raw["labels"], target["labels"]):
        raise AssertionError("Preprocessing changed class-label row order or object count")
    objects = []
    for index, label in enumerate(target["labels"].tolist()):
        objects.append({"index": index, "class_id": label,
                        "class_name": names[label] if 0 <= label < len(names) else str(label)})
    matrix = target["adj"]
    edges = []
    for above, below in (matrix == 1).nonzero().tolist():
        edges.append({"above_index": above, "below_index": below,
                      "above_class": objects[above]["class_name"],
                      "below_class": objects[below]["class_name"]})
    return {"matrix_before": raw["adj"].tolist(), "matrix_after": matrix.tolist(),
            "matrix_unchanged": True, "class_label_order_unchanged": True,
            "objects_in_annotation_order": objects, "direction": "row above -> column below",
            "edges": edges,
            "bidirectional_pairs": [[i, j] for i in range(len(objects))
                                    for j in range(i + 1, len(objects))
                                    if matrix[i, j] == 1 and matrix[j, i] == 1]}


def draw_adjacency(ax, matrix, labels, names, title):
    count = len(labels)
    ax.set_title(title)
    if not count:
        ax.text(0.5, 0.5, "No objects", ha="center")
        ax.axis("off")
        return
    ax.imshow(matrix.numpy(), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ticks = [f"{i}: {names[c] if 0 <= c < len(names) else c}"
             for i, c in enumerate(labels.tolist())]
    ax.set_xticks(range(count))
    ax.set_yticks(range(count))
    ax.set_xticklabels(ticks, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel("Column: below object")
    ax.set_ylabel("Row: above object")
    if count <= 20:
        for row in range(count):
            for col in range(count):
                value = int(matrix[row, col])
                ax.text(col, row, str(value), ha="center", va="center",
                        color="white" if value else "black", fontsize=8)


def draw_annotations(ax, image, boxes, labels, grasps, names, adjacency=None):
    ax.imshow(image)
    for index, (box, label) in enumerate(zip(boxes.tolist(), labels.tolist())):
        x1, y1, x2, y2 = box
        color = plt.get_cmap("tab10")(label % 10)
        ax.add_patch(
            Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5
            )
        )
        name = names[label] if 0 <= label < len(names) else str(label)
        ax.text(
            x1,
            y1,
            f"{index}: {name}",
            fontsize=8,
            color="white",
            bbox={"facecolor": color, "alpha": 0.8, "pad": 1},
        )
    for grasp in grasps.tolist():
        polygon = grasp + [grasp[0]]
        ax.plot(
            [p[0] for p in polygon],
            [p[1] for p in polygon],
            color="cyan",
            linewidth=0.7,
        )
    if adjacency is not None:
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        for above, below in (adjacency == 1).nonzero().tolist():
            ax.annotate("", xy=centers[below].tolist(), xytext=centers[above].tolist(),
                        arrowprops={"arrowstyle": "-|>", "color": "magenta",
                                    "lw": 1.8, "mutation_scale": 16,
                                    "connectionstyle": "arc3,rad=0.08"})
    ax.axis("off")


def crop_box(image, box):
    """Include a small margin around an object; accept PIL or CHW tensors."""
    width, height = image.size if not torch.is_tensor(image) else image.shape[-2:][::-1]
    x1, y1, x2, y2 = box.tolist()
    margin = max(4, 0.1 * max(x2 - x1, y2 - y1))
    bounds = (
        max(0, int(x1 - margin)),
        max(0, int(y1 - margin)),
        min(width, int(x2 + margin + 1)),
        min(height, int(y2 + margin + 1)),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    if torch.is_tensor(image):
        return (
            image[:, bounds[1] : bounds[3], bounds[0] : bounds[2]]
            .permute(1, 2, 0)
            .numpy()
        )
    return image.crop(bounds)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", "--dataset_root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--image-id", help="Image stem or filename; defaults to the first sorted image."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inspect this image and the following sorted images to show real batch padding.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Reproduce random training transforms."
    )
    parser.add_argument(
        "--with-grasps",
        action="store_true",
        help="Require and inspect grasp labels too.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/preprocessing")
    )
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="Also save exact normalized batch and targets as .pt.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.no_grasps = not args.with_grasps
    dataset = build_dataset(args.split, args)
    dataset.image_files.sort()
    start = 0
    if args.image_id:
        matches = [
            i
            for i, name in enumerate(dataset.image_files)
            if name == args.image_id or Path(name).stem == args.image_id
        ]
        if len(matches) != 1:
            parser.error(
                "--image-id must match exactly one image in the selected split"
            )
        start = matches[0]
    if start + args.batch_size > len(dataset):
        parser.error("Not enough images after the selected image for --batch-size")

    # Capture images/labels immediately before the real transform, including the
    # original annotations. Apply the existing pipeline exactly once per sample.
    pipeline = dataset.transform
    normalization = pipeline.transforms[-1].transforms[-1]
    mean = torch.tensor(normalization.mean).view(3, 1, 1)
    std = torch.tensor(normalization.std).view(3, 1, 1)
    originals = []

    def capture(image, target):
        originals.append((image.copy(), {k: v.clone() for k, v in target.items()}))
        return pipeline(image, target)

    dataset.transform = capture
    items = [dataset[i] for i in range(start, start + args.batch_size)]
    samples, targets = collate_fn(items)
    registry = args.dataset_root / "classes.json"
    names = json.loads(registry.read_text()) if registry.exists() else []
    output = (
        args.output_dir.expanduser().resolve()
        / f"{args.dataset_root.name}_{args.split}"
    )
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "seed": args.seed,
        "with_grasps": args.with_grasps,
        "pipeline": repr(pipeline),
        "mean": normalization.mean,
        "std": normalization.std,
        "batch_shape": list(samples.tensors.shape),
        "mask_shape": list(samples.mask.shape),
        "images": [],
    }
    for i, ((tensor, target), (original, raw)) in enumerate(zip(items, originals)):
        filename = dataset.image_files[start + i]
        h, w = tensor.shape[-2:]
        display = (tensor * std + mean).clamp(0, 1)
        boxes = box_cxcywh_to_xyxy(target["boxes"].reshape(-1, 4)) * torch.tensor(
            [w, h, w, h]
        )
        grasps = target["grasp_points"] * torch.tensor([w, h])
        adjacency = adjacency_report(raw, target, names)
        channels = {}
        for name, channel in zip("RGB", tensor):
            channels[name] = {
                "min": channel.min().item(),
                "max": channel.max().item(),
                "mean": channel.mean().item(),
                "std": channel.std(unbiased=False).item(),
            }
        info = {
            "file": filename,
            "original_wh": list(original.size),
            "label_original_hw": raw["orig_size"].tolist(),
            "label_size_matches_image": raw["orig_size"].tolist()
            == list(original.size[::-1]),
            "tensor_shape_chw": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "channels_after_normalization": channels,
            "top_left_normalized_rgb": tensor[:, 0, 0].tolist(),
            "padding_pixels": int(samples.mask[i].sum()),
            "object_count": len(boxes),
            "adjacency": adjacency,
        }
        report["images"].append(info)
        print(json.dumps(info, indent=2))

        fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
        draw_annotations(
            axes[0, 0],
            original,
            raw["boxes"].reshape(-1, 4),
            raw["labels"],
            raw["grasp_points"],
            names,
            raw["adj"],
        )
        axes[0, 0].set_title(f"Original: {original.width} × {original.height}")
        draw_annotations(
            axes[0, 1],
            display.permute(1, 2, 0).numpy(),
            boxes,
            target["labels"],
            grasps,
            names,
            target["adj"],
        )
        axes[0, 1].set_title(
            f"Processed: {w} × {h} (normalization reversed for display)"
        )

        # Match by object index, so the same object is shown even after flips.
        if len(boxes):
            areas = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
            index = int(areas.argmin())
            for ax, image, box, title in [
                (axes[1, 0], original, raw["boxes"][index], "Original smallest object"),
                (axes[1, 1], display, boxes[index], "Processed same object"),
            ]:
                detail = crop_box(image, box)
                if detail is not None:
                    ax.imshow(detail, interpolation="nearest")
                ax.set_title(title + " (enlarged)")
                ax.axis("off")
        else:
            for ax in axes[1]:
                ax.text(0.5, 0.5, "No labeled objects", ha="center")
                ax.axis("off")

        padded = (samples.tensors[i] * std + mean).clamp(0, 1)
        padded[:, samples.mask[i]] = 0  # Render padding black for clarity.
        axes[2, 0].imshow(padded.permute(1, 2, 0).numpy())
        axes[2, 0].set_title("Batch image (padding shown black)")
        axes[2, 1].imshow(samples.mask[i].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2, 1].set_title("Actual padding mask: white = padding, black = image")
        for ax in axes[2]:
            ax.axis("off")
        fig.suptitle(
            f"{filename} | {args.split} | seed {args.seed}\n"
            "Display RGB is reconstructed; the model receives normalized tensor values.",
            fontsize=13,
        )
        figure_path = output / f"{Path(filename).stem}_comparison.png"
        fig.savefig(figure_path, dpi=130)
        plt.close(fig)
        plt.imsave(
            output / f"{Path(filename).stem}_processed.png",
            display.permute(1, 2, 0).numpy(),
        )
        print(f"Saved {figure_path}")
        adj_fig, adj_axes = plt.subplots(2, 2, figsize=(14, 13), constrained_layout=True)
        for column, image, stage_boxes, stage_grasps, stage_target, title in [
            (0, original, raw["boxes"].reshape(-1, 4), raw["grasp_points"], raw, "Before preprocessing"),
            (1, display.permute(1, 2, 0).numpy(), boxes, grasps, target, "After preprocessing"),
        ]:
            draw_annotations(adj_axes[0, column], image, stage_boxes, stage_target["labels"],
                             stage_grasps, names, stage_target["adj"])
            adj_axes[0, column].set_title(title)
            draw_adjacency(adj_axes[1, column], stage_target["adj"], stage_target["labels"],
                           names, title + " adjacency")
        adj_fig.suptitle(f"{filename}: ground-truth relationships (not model predictions)\n"
                         "Box numbers = zero-based annotation rows. Magenta arrows: above → below.\n"
                         "PASS: adjacency unchanged; class-label row order unchanged.")
        adjacency_path = output / f"{Path(filename).stem}_adjacency.png"
        adj_fig.savefig(adjacency_path, dpi=130)
        plt.close(adj_fig)
        print(f"Saved {adjacency_path}")
        if adjacency["bidirectional_pairs"]:
            print(f"WARNING: contradictory above/below pairs: {adjacency['bidirectional_pairs']}")
    stem = Path(dataset.image_files[start]).stem
    report_path = output / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.save_tensors:
        torch.save(
            {"tensors": samples.tensors, "mask": samples.mask, "targets": targets},
            output / f"{stem}_batch.pt",
        )
    print(
        f"Batch shape: {tuple(samples.tensors.shape)}; mask: {tuple(samples.mask.shape)}"
    )
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
