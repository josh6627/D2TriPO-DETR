#!/usr/bin/env python3
"""Run D2TriPO-DETR on unlabeled RGB images; save detections and directed relations.

Use from the D2TriPO-DETR repository root in your existing d2tripo environment:
    python infer_unlabeled_dataset.py --input /path/to/images --limit 5
    python infer_unlabeled_dataset.py --input /path/to/images --output results/lab_dataset

No VMRD labels, adjacency files, or Hungarian matching are required.
"""

import argparse
import csv
import json
import math
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T

# VMRD's zero-based category names, matching inspect_objects.py.
VMRD_CLASSES = [
    "banana",
    "watch",
    "pliers",
    "toothbrush",
    "wrench",
    "towel",
    "knife",
    "badminton",
    "tape",
    "charger",
    "pen",
    "box",
    "umbrella",
    "apple",
    "wallet",
    "shaver",
    "paper",
    "cup",
    "mobile phone",
    "wrist developer",
    "bottle",
    "headset",
    "toothpaste",
    "glasses",
    "remote controller",
    "card",
    "mouse",
    "cans",
    "screwdriver",
    "notebook",
    "stapler",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = [
    (49, 196, 141),
    (79, 163, 255),
    (255, 175, 75),
    (226, 117, 228),
    (250, 105, 114),
    (76, 213, 220),
    (255, 220, 90),
    (155, 169, 252),
]


def load_model(checkpoint_path, device):
    # Import lazily so this file can be tested without the compiled DETR operators.
    from models import build_dab_deformable_detr

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # The project's known checkpoint contains Python argparse settings, so load
    # only a checkpoint you trust. weights_only=False is required for those args.
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    model, _, _ = build_dab_deformable_detr(checkpoint["args"])
    model.load_state_dict(checkpoint["model"])
    model.class_names = getattr(checkpoint["args"], "class_names", VMRD_CLASSES)
    model = model.to(device)
    model.eval()
    return model


def make_transform():
    # Same preprocessing as the verified inspect_objects.py workflow.
    return T.Compose(
        [
            T.Resize(800, max_size=1333),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def find_images(input_path, output_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [(input_path, Path(input_path.name))]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input file/folder not found: {input_path}")

    images = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Avoid picking up images from an output directory inside the input tree.
        if output_path == path or output_path in path.parents:
            continue
        images.append((path, path.relative_to(input_path)))
    return images


def cxcywh_to_xyxy_pixels(box, width, height):
    cx, cy, bw, bh = [float(x) for x in box]
    return [
        round(max(0.0, min(float(width), (cx - bw / 2) * width)), 2),
        round(max(0.0, min(float(height), (cy - bh / 2) * height)), 2),
        round(max(0.0, min(float(width), (cx + bw / 2) * width)), 2),
        round(max(0.0, min(float(height), (cy + bh / 2) * height)), 2),
    ]


def decode_predictions(
    outputs, width, height, score_threshold, relation_threshold, topk, class_names=None
):
    """Select query/class predictions; never use GT annotations or matching."""
    class_names = VMRD_CLASSES if class_names is None else class_names
    probabilities = outputs["pred_logits"][0].sigmoid()
    num_classes = probabilities.shape[1]
    k = min(topk, probabilities.numel())
    scores, indices = torch.topk(probabilities.flatten(), k)

    detections = []
    seen_queries = set()
    boxes = outputs["pred_boxes"][0]

    for score, index in zip(scores, indices):
        confidence = float(score.item())
        if confidence <= score_threshold:
            # topk is descending, so all remaining entries also fail.
            break
        flat_index = int(index.item())
        query_id = flat_index // num_classes
        class_id = flat_index % num_classes

        # A single query can score highly for more than one class; take its
        # highest-scoring class occurrence only (same rule as earlier tests).
        if query_id in seen_queries:
            continue
        seen_queries.add(query_id)

        class_name = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"unmapped_class_{class_id}"
        )
        detections.append(
            {
                "id": len(detections),
                "query_id": query_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 6),
                "bbox_xyxy": cxcywh_to_xyxy_pixels(boxes[query_id], width, height),
            }
        )

    query_ids = [det["query_id"] for det in detections]
    n = len(query_ids)
    # Per-image adjacency is indexed by the original DETR query IDs.
    if n:
        logits = outputs["pred_adj"][0][query_ids][:, query_ids]
        probs = logits.sigmoid().detach().cpu()
    else:
        probs = torch.empty((0, 0))

    binary_matrix = [[0] * n for _ in range(n)]
    probabilities_matrix = [[0.0] * n for _ in range(n)]
    relationships = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # No object should have a relationship with itself.
            p = float(probs[i, j].item())
            probabilities_matrix[i][j] = round(p, 6)
            if p > relation_threshold:
                binary_matrix[i][j] = 1
                relationships.append(
                    {
                        "above_id": i,
                        "below_id": j,
                        "above_query_id": query_ids[i],
                        "below_query_id": query_ids[j],
                        "above_class": detections[i]["class_name"],
                        "below_class": detections[j]["class_name"],
                        "confidence": round(p, 6),
                    }
                )

    # Both A[i,j] = 1 and A[j,i] = 1 form a contradictory ordered pair.
    contradictory_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if binary_matrix[i][j] and binary_matrix[j][i]:
                contradictory_pairs.append([i, j])

    return {
        "detections": detections,
        "adjacency_binary": binary_matrix,
        "adjacency_probabilities": probabilities_matrix,
        "relationships": relationships,
        "contradictory_pairs": contradictory_pairs,
    }


def box_center(det):
    x1, y1, x2, y2 = det["bbox_xyxy"]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_arrow(draw, src_xy, dst_xy):
    sx, sy = src_xy
    dx, dy = dst_xy
    ux, uy = dx - sx, dy - sy
    distance = math.hypot(ux, uy)
    if distance < 1:
        return
    ux, uy = ux / distance, uy / distance
    inset = min(24, distance / 4)
    start = (sx + ux * inset, sy + uy * inset)
    end = (dx - ux * inset, dy - uy * inset)
    color = (255, 255, 0)
    draw.line([start, end], fill=color, width=5)
    length, half_width = 16, 8
    base = (end[0] - ux * length, end[1] - uy * length)
    left = (base[0] - uy * half_width, base[1] + ux * half_width)
    right = (base[0] + uy * half_width, base[1] - ux * half_width)
    draw.polygon([end, left, right], fill=color)


def visualize(image, detections, relationships, with_arrows, font=None):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    if with_arrows:
        for rel in relationships:
            src = detections[rel["above_id"]]
            dst = detections[rel["below_id"]]
            draw_arrow(draw, box_center(src), box_center(dst))

    for det in detections:
        i = det["id"]
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = det["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f'{i}: {det["class_name"]}'
        if "confidence" in det:
            label += f' {det["confidence"]:.2f}'
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle(
            (x1, label_y, x1 + text_w + 8, label_y + text_h + 6), fill=(0, 0, 0)
        )
        draw.text(
            (x1 + 4 - text_bbox[0], label_y + 3 - text_bbox[1]),
            label, fill=color, font=font,
        )
    return canvas


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Predict objects and manipulation relationships in unlabeled RGB images"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Image file or directory of RGB images (searched recursively)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/unlabeled"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/checkpoint0110.pth")
    )
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--relation-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--limit", type=int, default=None, help="Test on the first N images"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    if not 0 <= args.score_threshold <= 1 or not 0 <= args.relation_threshold <= 1:
        parser.error("Thresholds must be between 0 and 1")
    if args.topk < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--topk and --limit must be positive")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()

    image_pairs = find_images(input_path, output_path)
    if args.limit is not None:
        image_pairs = image_pairs[: args.limit]
    if not image_pairs:
        raise RuntimeError(f"No supported images found at {input_path}")

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    device = torch.device(requested_device)

    print(f"Loading checkpoint once: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    transform = make_transform()
    print(f"Found {len(image_pairs)} images. Device: {device}")

    summary_rows, relationship_rows, errors = [], [], []
    for counter, (image_path, relative_path) in enumerate(image_pairs, start=1):
        image_key = relative_path.as_posix()
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs = model(input_tensor)
            prediction = decode_predictions(
                outputs,
                width,
                height,
                args.score_threshold,
                args.relation_threshold,
                args.topk,
                class_names=model.class_names,
            )

            data = {
                "image": image_key,
                "width": width,
                "height": height,
                "object_score_threshold": args.score_threshold,
                "relationship_threshold": args.relation_threshold,
                "note": "Predictions only; no ground truth or accuracy metrics",
                **prediction,
            }
            json_file = (output_path / "predictions" / relative_path).with_suffix(
                ".json"
            )
            json_file.parent.mkdir(parents=True, exist_ok=True)
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, allow_nan=False)

            image_file = (output_path / "visualizations" / relative_path).with_suffix(
                ""
            )
            image_file.parent.mkdir(parents=True, exist_ok=True)
            visualize(
                image,
                prediction["detections"],
                prediction["relationships"],
                with_arrows=False,
            ).save(str(image_file) + "_objects.jpg", quality=90)
            visualize(
                image,
                prediction["detections"],
                prediction["relationships"],
                with_arrows=True,
            ).save(str(image_file) + "_relationships.jpg", quality=90)

            summary_rows.append(
                {
                    "image": image_key,
                    "detected_objects": len(prediction["detections"]),
                    "predicted_relationships": len(prediction["relationships"]),
                    "contradictory_pairs": len(prediction["contradictory_pairs"]),
                }
            )
            for rel in prediction["relationships"]:
                relationship_rows.append({"image": image_key, **rel})
            print(
                f"[{counter}/{len(image_pairs)}] {image_key}: "
                f"{len(prediction['detections'])} objects, "
                f"{len(prediction['relationships'])} directed relationships"
            )
        except Exception:
            errors.append(f"{image_key}\n{traceback.format_exc()}")
            print(f"[{counter}/{len(image_pairs)}] ERROR: {image_key}")

    save_csv(
        output_path / "summary.csv",
        summary_rows,
        ["image", "detected_objects", "predicted_relationships", "contradictory_pairs"],
    )
    save_csv(
        output_path / "relationships.csv",
        relationship_rows,
        [
            "image",
            "above_id",
            "below_id",
            "above_query_id",
            "below_query_id",
            "above_class",
            "below_class",
            "confidence",
        ],
    )
    (output_path / "errors.txt").write_text("\n\n".join(errors), encoding="utf-8")

    print("\n=== UNLABELED DATASET INFERENCE ===")
    print(f"Requested images: {len(image_pairs)}")
    print(f"Successfully processed: {len(summary_rows)}")
    print(f"Failed images: {len(errors)}")
    print(f"Predicted detections: {sum(r['detected_objects'] for r in summary_rows)}")
    print(f"Predicted directed relationships: {len(relationship_rows)}")
    print(f"Results: {output_path}")
    print("No OR/OP/IA: no ground-truth annotations supplied.")
    if errors:
        print(f"See {output_path / 'errors.txt'} for failures")


if __name__ == "__main__":
    main()
