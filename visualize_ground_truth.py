#!/usr/bin/env python3
"""Render dataset annotations using evaluate_dataset's box/relationship style."""

import argparse
import json
import math
from pathlib import Path

from evaluate_dataset import save_visualizations


def load_annotations(dataset_root, split, image_path, class_names):
    """Keep annotation row order: adjacency rows/columns index these objects."""
    label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
    detections = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{label_path}:{line_number}: expected class ID and pixel xyxy box")
        class_id = int(parts[0])
        box = [float(value) for value in parts[1:5]]
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"{label_path}:{line_number}: unknown class ID {class_id}")
        if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{label_path}:{line_number}: invalid xyxy box {box}")
        detections.append(dict(id=len(detections), class_id=class_id,
                               class_name=class_names[class_id], bbox_xyxy=box))

    adj_path = dataset_root / "adj" / split / f"{image_path.stem}.txt"
    adjacency = [[int(value) for value in line.split()]
                 for line in adj_path.read_text().splitlines() if line.strip()]
    count = len(detections)
    if len(adjacency) != count or any(len(row) != count for row in adjacency):
        raise ValueError(f"{adj_path}: expected a {count} x {count} adjacency matrix")
    if any(value not in (0, 1) for row in adjacency for value in row):
        raise ValueError(f"{adj_path}: expected binary adjacency values")
    if any(adjacency[i][i] != 0 for i in range(count)):
        raise ValueError(f"{adj_path}: adjacency diagonal must be zero")
    relationships = [dict(above_id=i, below_id=j)
                     for i, row in enumerate(adjacency)
                     for j, value in enumerate(row) if value == 1]
    return dict(detections=detections, relationships=relationships)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--dataset_root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ground_truth"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.dataset_root.expanduser().resolve()
    for directory in ("images", "labels", "adj"):
        if not (root / directory / args.split).is_dir():
            parser.error(f"Missing directory: {root / directory / args.split}")
    registry = root / "classes.json"
    if registry.is_file():
        names = json.loads(registry.read_text())
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
            parser.error(f"{registry}: expected a nonempty list of class names")
    else:
        # Same legacy class order used by the prediction visualizer.
        from infer_unlabeled_dataset import VMRD_CLASSES
        names = VMRD_CLASSES
    images = sorted(path for path in (root / "images" / args.split).iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        parser.error("No JPG, JPEG, or PNG images found")
    output = args.output_dir.expanduser() / f"{root.name}_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    for index, image_path in enumerate(images, 1):
        try:
            annotations = load_annotations(root, args.split, image_path, names)
            save_visualizations(image_path, annotations, output, ground_truth=True)
        except Exception as exc:
            errors.append(f"{image_path.name}: {exc}")
            print(f"ERROR: {errors[-1]}")
        if index % 25 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)} images")
    (output / "errors.txt").write_text("\n".join(errors))
    print(f"Saved {len(images) - len(errors)}/{len(images)} ground-truth visualizations: {output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
