import sys
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.dataset_config import class_id_mapping

from models import build_dab_deformable_detr


def evaluate_relationships(ground_truth, predicted):
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(ground_truth)):
        for j in range(len(ground_truth)):
            if i == j:
                continue

            gt = ground_truth[i][j]
            pred = predicted[i][j]

            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 1 and pred == 0:
                fn += 1
            elif gt == 0 and pred == 1:
                fp += 1
    return tp, fp, fn


def match_objects(
    gt_objects, outputs, scores, indices, keep, num_classes, W, H, min_iou=0.5,
    class_agnostic=False,
):

    detections = []
    used_queries = set()

    # Collect valid-confidence detections
    for score, index in zip(scores[keep], indices[keep]):
        query_id = int(index // num_classes)
        class_id = int(index % num_classes)

        # A query may appear multiple times with different classes.
        # Keep only its highest-scoring class.
        if query_id in used_queries:
            continue

        used_queries.add(query_id)

        detections.append(
            {
                "query_id": query_id,
                "class_id": class_id,
                "score": float(score),
            }
        )

    if not detections or not gt_objects:
        return {}, detections

    # Convert predicted boxes from normalized cxcywh to pixel xyxy
    query_ids = [d["query_id"] for d in detections]

    pred_boxes = outputs["pred_boxes"][0, query_ids].detach().cpu()
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
    pred_boxes *= torch.tensor([W, H, W, H])

    # Ground-truth boxes are already in pixel xyxy format
    gt_boxes = torch.tensor([obj["box"] for obj in gt_objects], dtype=torch.float32)

    # Calculate overlap between every GT and predicted box
    iou_matrix, _ = box_iou(gt_boxes, pred_boxes)

    n_gt = len(gt_objects)
    n_pred = len(detections)

    # High cost for invalid matches; dummy columns allow unmatched GT.
    cost = torch.full((n_gt, n_pred + n_gt), 1000.0)
    cost[:, n_pred:] = 10.0

    for i, gt in enumerate(gt_objects):
        for j, det in enumerate(detections):

            same_class = gt["class_id"] == det["class_id"]
            iou = float(iou_matrix[i, j])

            if (class_agnostic or same_class) and iou >= min_iou:
                cost[i, j] = 1.0 - iou

    # Find the best one-to-one assignment
    gt_indices, pred_indices = linear_sum_assignment(cost.numpy())

    matches = {}

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if pred_idx < n_pred and cost[gt_idx, pred_idx] < 10.0:
            matches[int(gt_idx)] = detections[int(pred_idx)]

    return matches, detections


def align_relationships(gt_adj, outputs, matches, detections):
    """Align all directed edges, keeping unmatched detections as extra slots."""
    matched_queries = {det["query_id"] for det in matches.values()}
    extra_queries = [det["query_id"] for det in detections
                     if det["query_id"] not in matched_queries]
    n_gt = len(gt_adj)
    n_total = n_gt + len(extra_queries)
    ground_truth = torch.zeros((n_total, n_total), dtype=gt_adj.dtype)
    ground_truth[:n_gt, :n_gt] = gt_adj
    predicted = torch.zeros_like(ground_truth)
    gt_indices = sorted(matches)
    positions = gt_indices + list(range(n_gt, n_total))
    query_ids = [matches[i]["query_id"] for i in gt_indices] + extra_queries
    if query_ids:
        adjacency = outputs["pred_adj"][0]
        subset = (adjacency[query_ids][:, query_ids].sigmoid() > 0.5).long().cpu()
        idx = torch.tensor(positions, dtype=torch.long)
        predicted[idx[:, None], idx[None, :]] = subset
    predicted.fill_diagonal_(0)
    return ground_truth, predicted


device = torch.device("cuda")


def load_model(checkpoint_path="checkpoints/checkpoint0110.pth"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    args.device = str(device)

    model, _, _ = build_dab_deformable_detr(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.class_names = getattr(args, "class_names", None)
    return model.to(device).eval()


def evaluate_one_image(image_path, model, return_predictions=False):

    # Load RGB image
    image = Image.open(image_path).convert("RGB")
    print("Image:", image_path)
    print("Original size:", image.size)

    # Match the validation preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize(800, max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    print("Network input shape:", tuple(input_tensor.shape))

    # Run inference
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        outputs = model(input_tensor)

    print("\nInference successful!")

    for name, value in outputs.items():
        if torch.is_tensor(value):
            print(f"{name}: {tuple(value.shape)}")

    print(
        f"\nPeak GPU memory: " f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

    # Examine the object detection predictions
    class_names = [
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
        "plate",
        "ball",
        "lego",
        "fruit",
        "clamp",
        "squeeze bottle",
        "lid",
        "tool",
        "bowl",
        "lock",
        "pitcher",
        "scissors",
        "drill",
        "usb",
        "key",
        "button",
        "id",
        "ruler",
        "3dprint",
        "wood",
        "cap",
    ]

    class_names = getattr(model, "class_names", None) or class_names

    probabilities = outputs["pred_logits"][0].sigmoid()
    num_classes = probabilities.shape[1]

    scores, indices = torch.topk(probabilities.flatten(), min(100, probabilities.numel()))
    keep = scores > 0.4

    print("\n=== Object predictions above 0.4 ===")
    print("Number of query/class predictions:", int(keep.sum()))

    for score, index in zip(scores[keep][:15], indices[keep][:15]):
        query_id = int(index // num_classes)
        class_id = int(index % num_classes)

        name = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"unnamed class {class_id}"
        )

        print(f"Query {query_id:3d} | " f"{name:20s} | " f"Score: {score.item():.3f}")

    print("\n=== Predicted vs Ground Truth Boxes ===")

    # Obtain the original image dimensions
    W, H = image.size

    # Determine the annotation path from the input image path
    # Follow the dataset layout, not an image symlink's target in a raw folder.
    image_file = Path(image_path).absolute()
    gt_objects = []

    # Expected VMRD structure: images/<split>/<image_id>.jpg
    if image_file.parent.parent.name == "images":
        split = image_file.parent.name
        image_id = image_file.stem
        dataset_root = image_file.parent.parent.parent

        label_path = dataset_root / "labels" / split / f"{image_id}.txt"
        id_mapping = class_id_mapping(dataset_root, class_names[:num_classes])

        if label_path.exists():
            print("Ground-truth file:", label_path)

            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    source_class_id = int(parts[0])
                    if id_mapping is not None and source_class_id not in id_mapping:
                        raise ValueError(
                            f"{label_path}: class ID {source_class_id} has no matching class in the checkpoint"
                        )
                    class_id = (id_mapping[source_class_id] if id_mapping is not None
                                else source_class_id)

                    gt_objects.append(
                        {
                            "class_id": class_id,
                            "box": [float(v) for v in parts[1:5]],
                        }
                    )
        else:
            print("No ground-truth annotation found:", label_path)
    else:
        print("Image is outside the expected VMRD folder structure.")

    # Display predictions and corresponding ground-truth boxes
    for score, index in zip(scores[keep], indices[keep]):
        query_id = int(index // num_classes)
        class_id = int(index % num_classes)

        cx, cy, w, h = outputs["pred_boxes"][0, query_id].tolist()

        predicted_box = [
            (cx - w / 2) * W,
            (cy - h / 2) * H,
            (cx + w / 2) * W,
            (cy + h / 2) * H,
        ]

        name = class_names[class_id] if class_id < len(class_names) else str(class_id)

        matching_gt = [obj["box"] for obj in gt_objects if obj["class_id"] == class_id]

        print(f"\n{name} (query {query_id})")
        print("  Confidence:", round(score.item(), 3))
        print("  Predicted box:", [round(v, 1) for v in predicted_box])
        print("  Ground-truth boxes:", matching_gt)

    # Load the ground-truth manipulation relationships
    print("\n=== Ground Truth Manipulation Relationships ===")

    adj_path = dataset_root / "adj" / split / f"{image_id}.txt"

    with open(adj_path) as f:
        gt_adj = torch.tensor(
            [[int(value) for value in line.split()] for line in f if line.strip()],
            dtype=torch.int64,
        )
    expected_shape = (len(gt_objects), len(gt_objects))
    if tuple(gt_adj.shape) != expected_shape:
        raise ValueError(f"{adj_path}: expected adjacency shape {expected_shape}, got {tuple(gt_adj.shape)}")
    if not torch.all((gt_adj == 0) | (gt_adj == 1)) or torch.any(gt_adj.diagonal() != 0):
        raise ValueError(f"{adj_path}: adjacency must be binary with a zero diagonal")

    print("Ground-truth adjacency file:", adj_path)

    for row in gt_adj:
        print(row.tolist())
        # Ground-truth object classes, in annotation order
        gt_classes = [obj["class_id"] for obj in gt_objects]

    matches, detections = match_objects(
        gt_objects, outputs, scores, indices, keep, num_classes, W, H
    )

    print("\n=== Automatic Object Matching ===")

    for gt_idx, obj in enumerate(gt_objects):
        if gt_idx in matches:
            det = matches[gt_idx]

            print(
                f"GT object {gt_idx} "
                f"({class_names[obj['class_id']] if obj['class_id'] < len(class_names) else str(obj['class_id'])}) "
                f"→ Query {det['query_id']}"
            )
        else:
            print(f"GT object {gt_idx} → NOT MATCHED")

    matched_query_ids = {det["query_id"] for det in matches.values()}

    unmatched_detections = [
        det for det in detections if det["query_id"] not in matched_query_ids
    ]

    matched_gt_indices = sorted(matches.keys())

    matched_query_ids = [matches[i]["query_id"] for i in matched_gt_indices]

    gt_subset = gt_adj[matched_gt_indices][:, matched_gt_indices]

    pred_subset = outputs["pred_adj"][0]
    pred_subset = pred_subset[matched_query_ids][:, matched_query_ids]
    pred_subset = (pred_subset.sigmoid() > 0.5).int()

    print("Matched ground-truth indices:", matched_gt_indices)
    print("Matched query IDs:", matched_query_ids)

    print("\nGT relationships among matched objects:")
    print(gt_subset.tolist())

    print("\nPredicted relationships among matched objects:")
    print(pred_subset.tolist())

    # Number of annotated and extra predicted objects
    n_gt = len(gt_objects)
    n_extra = len(unmatched_detections)
    gt_for_scoring, predicted_binary = align_relationships(
        gt_adj, outputs, matches, detections
    )

    missing_objects = n_gt - len(matches)

    print("\nMissing GT objects:", missing_objects)
    print("Unmatched detections:", n_extra)

    print("\nFull predicted adjacency:")
    for row in predicted_binary:
        print(row.tolist())

    tp, fp, fn = evaluate_relationships(
        gt_for_scoring.tolist(), predicted_binary.tolist()
    )

    # False-positive directed edges involving extra detections
    extra_relation_fp = int(
        predicted_binary[n_gt:, :].sum().item()
        + predicted_binary[:n_gt, n_gt:].sum().item()
    )

    print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")
    print("FP edges involving extra detections:", extra_relation_fp)

    or_score = tp / (tp + fn) if (tp + fn) > 0 else None
    op_score = tp / (tp + fp) if (tp + fp) > 0 else None

    print("Provisional edge recall:", or_score)
    print("Provisional edge precision:", op_score)

    image_correct = (
        missing_objects == 0
        and n_extra == 0
        and torch.equal(predicted_binary, gt_for_scoring)
    )

    print("Image correct:", image_correct)

    result = {
        "total_objects": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "missing_objects": missing_objects,
        "extra_detections": len(unmatched_detections),
        "image_correct": int(image_correct),
        "extra_relation_fp": extra_relation_fp,
    }
    # Independently match geometry so class mistakes do not hide correct edges.
    bbox_matches, _ = match_objects(
        gt_objects, outputs, scores, indices, keep, num_classes, W, H,
        class_agnostic=True,
    )
    bbox_gt, bbox_pred = align_relationships(gt_adj, outputs, bbox_matches, detections)
    bbox_tp, bbox_fp, bbox_fn = evaluate_relationships(bbox_gt.tolist(), bbox_pred.tolist())
    bbox_missing = n_gt - len(bbox_matches)
    bbox_extra = len(detections) - len(bbox_matches)
    result.update(
        bbox_tp=bbox_tp,
        bbox_fp=bbox_fp,
        bbox_fn=bbox_fn,
        bbox_missing_objects=bbox_missing,
        bbox_extra_detections=bbox_extra,
        bbox_extra_relation_fp=int(
            bbox_pred[n_gt:, :].sum().item() + bbox_pred[:n_gt, n_gt:].sum().item()
        ),
        bbox_image_correct=int(
            bbox_missing == 0 and bbox_extra == 0 and torch.equal(bbox_gt, bbox_pred)
        ),
    )
    print(f"Box-only TP: {bbox_tp}, FP: {bbox_fp}, FN: {bbox_fn}")
    if return_predictions:
        from infer_unlabeled_dataset import decode_predictions

        result["prediction"] = decode_predictions(
            outputs,
            W,
            H,
            score_threshold=0.4,
            relation_threshold=0.5,
            topk=100,
            class_names=class_names,
        )
    return result


if __name__ == "__main__":
    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("data/vmrd/images/val/00291.jpg")
    )

    result = evaluate_one_image(image_path, load_model())
    print("\nEvaluation result:", result)
