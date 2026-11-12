import argparse
import csv
import io
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def save_visualizations(image_path, prediction, output_dir):
    from PIL import Image, ImageDraw, ImageFont
    from infer_unlabeled_dataset import visualize

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        label_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except OSError:
        label_font = font = ImageFont.load_default()
    annotated = visualize(
        image, prediction["detections"], prediction["relationships"],
        with_arrows=True, font=label_font,
    )

    # Keep the complete relationship list outside the image so it cannot cover
    # objects or become unreadable when many arrows overlap.
    detections = {det["id"]: det for det in prediction["detections"]}
    relationships = prediction["relationships"]
    lines = [f"Predicted relationships ({len(relationships)})"]
    for rel in relationships:
        above_id, below_id = rel["above_id"], rel["below_id"]
        lines.append(
            f'#{above_id} {detections[above_id]["class_name"]} above '
            f'#{below_id} {detections[below_id]["class_name"]}'
            f'  ({rel["confidence"]:.2f})'
        )
    if not relationships:
        lines.append("No relationships predicted.")

    padding = 16
    measure = ImageDraw.Draw(annotated)
    bounds = [measure.textbbox((0, 0), line, font=font) for line in lines]
    line_height = max(box[3] - box[1] for box in bounds) + 10
    panel_width = max(box[2] - box[0] for box in bounds) + 2 * padding
    canvas = Image.new(
        "RGB",
        (annotated.width + panel_width,
         max(annotated.height, 2 * padding + len(lines) * line_height)),
        (24, 24, 24),
    )
    canvas.paste(annotated, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (line, box) in enumerate(zip(lines, bounds)):
        draw.text(
            (annotated.width + padding - box[0],
             padding + index * line_height - box[1]),
            line,
            font=font,
            fill=(255, 220, 90) if index == 0 else (240, 240, 240),
        )
    canvas.save(output_dir / f"{image_path.name}_relationships.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate boxes and relationships in a labeled dataset folder."
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        "--folder",
        type=Path,
        required=True,
        help="Dataset folder containing images/<split>, labels/<split>, and adj/<split>.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test", "both"),
        help="Split to evaluate (default: val); both runs train and val with one model load.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/checkpoint0110.pth")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Image output folder; defaults to <output-dir>/visualizations. Dataset/split subfolders are added.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum images per split."
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip saving annotated object and relationship images.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    splits = ("train", "val") if args.split == "both" else (args.split,)
    images_by_split = {}
    for split in splits:
        image_dir = dataset_root / "images" / split
        for directory in (
            image_dir,
            dataset_root / "labels" / split,
            dataset_root / "adj" / split,
        ):
            if not directory.is_dir():
                parser.error(f"Required dataset folder does not exist: {directory}")

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if args.limit is not None:
            images = images[: args.limit]

        if not images:
            parser.error(f"No JPG, JPEG, or PNG images found in {image_dir}")

        for image_path in images:
            for kind in ("labels", "adj"):
                annotation = dataset_root / kind / split / f"{image_path.stem}.txt"
                if not annotation.is_file():
                    parser.error(f"Missing annotation: {annotation}")

        images_by_split[split] = images

    # Validate inputs before importing the CUDA evaluation backend or loading weights.
    from inspect_objects import evaluate_one_image, load_model

    model = load_model(args.checkpoint.expanduser())
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, images in images_by_split.items():
        print(f"\n=== {split.upper()} SPLIT ===")
        evaluate_split(
            images,
            model,
            evaluate_one_image,
            output_dir,
            f"{dataset_root.name}_{split}",
            save_images=not args.no_visualizations,
            visualization_dir=(
                args.visualization_dir.expanduser()
                if args.visualization_dir is not None
                else None
            ),
        )


def evaluate_split(
    images,
    model,
    evaluate_one_image,
    output_dir,
    result_prefix,
    save_images=True,
    visualization_dir=None,
):
    image_dir = images[0].parent
    image_output_dir = (
        visualization_dir
        if visualization_dir is not None
        else output_dir / "visualizations"
    ) / result_prefix
    image_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(images)} images from {image_dir}")

    rows = []
    errors = []
    visualization_errors = []

    for i, image_path in enumerate(images, start=1):
        debug_output = io.StringIO()

        try:
            # Hide the extensive single-image debug printing
            with redirect_stdout(debug_output):
                if save_images:
                    result = evaluate_one_image(
                        image_path, model, return_predictions=True
                    )
                else:
                    result = evaluate_one_image(image_path, model)

            rows.append(
                {
                    "image": image_path.name,
                    "tp": int(result["tp"]),
                    "fp": int(result["fp"]),
                    "fn": int(result["fn"]),
                    "missing_objects": int(result["missing_objects"]),
                    "extra_detections": int(result["extra_detections"]),
                    "extra_relation_fp": int(result["extra_relation_fp"]),
                    "image_correct": int(result["image_correct"]),
                }
            )
            if save_images:
                try:
                    save_visualizations(
                        image_path, result["prediction"], image_output_dir
                    )
                except Exception:
                    # A rendering failure must not discard valid evaluation results.
                    visualization_errors.append(
                        f"\nVISUALIZATION: {image_path.name}\n{traceback.format_exc()}\n"
                    )
                    print(f"VISUALIZATION ERROR: {image_path.name}")

        except Exception:
            errors.append(
                f"\nIMAGE: {image_path.name}\n"
                f"{traceback.format_exc()}\n"
                f"DEBUG OUTPUT:\n{debug_output.getvalue()[-5000:]}\n"
            )
            print(f"ERROR: {image_path.name}")

        if i % 25 == 0 or i == len(images):
            print(f"Processed {i}/{len(images)} images")

    # Save the individual results
    csv_path = image_output_dir / "results.csv"

    # Always replace the CSV, even if all images failed, so old scores cannot
    # masquerade as results from the current checkpoint.
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "image",
                "tp",
                "fp",
                "fn",
                "missing_objects",
                "extra_detections",
                "extra_relation_fp",
                "image_correct",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save error details separately
    error_path = image_output_dir / "errors.txt"
    error_path.write_text("\n".join(errors + visualization_errors))

    # Aggregate counts across all successfully evaluated images
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)

    total_missing = sum(r["missing_objects"] for r in rows)
    total_extra = sum(r["extra_detections"] for r in rows)
    correct_images = sum(r["image_correct"] for r in rows)

    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else None
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else None
    strict_accuracy = correct_images / len(rows) if rows else None

    print("\n=== DATASET RESULTS ===")
    print(f"Images requested: {len(images)}")
    print(f"Images evaluated: {len(rows)}")
    print(f"Images failed: {len(errors)}")
    if visualization_errors:
        print(f"Visualizations failed: {len(visualization_errors)} (metrics retained)")
    print(f"Total missing objects: {total_missing}")
    print(f"Total extra detections: {total_extra}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Provisional micro edge recall: {recall * 100.0 if recall is not None else 'N/A'}")
    print(f"Provisional micro edge precision: {precision * 100.0 if precision is not None else 'N/A'}")
    print(f"Strict correct-image fraction: {strict_accuracy * 100.0 if strict_accuracy is not None else 'N/A'}")
    print(f"\nResults saved: {csv_path}")
    print(f"Error log: {error_path}")
    if save_images:
        print(f"Visualizations: {image_output_dir}")

    if errors:
        print("WARNING: Metrics exclude images that failed evaluation.")


if __name__ == "__main__":
    main()
