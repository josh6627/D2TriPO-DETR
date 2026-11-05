import csv
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout, redirect_stderr

import evaluate_dataset


class EvaluateDatasetTests(unittest.TestCase):
    def test_custom_visualization_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            for kind in ("images", "labels", "adj"):
                folder = root / kind / "val"
                folder.mkdir(parents=True)
                (folder / ("one.png" if kind == "images" else "one.txt")).touch()
            prediction = {"detections": [], "relationships": []}
            backend = SimpleNamespace(
                load_model=Mock(return_value=object()),
                evaluate_one_image=Mock(return_value=dict(
                    tp=0, fp=0, fn=0, missing_objects=0, extra_detections=0,
                    extra_relation_fp=0, image_correct=1, prediction=prediction)),
            )
            output = Path(directory) / "metrics"
            images = Path(directory) / "custom_images"
            with patch.object(sys, "argv", ["evaluate_dataset.py", "--dataset-root",
                              str(root), "--output-dir", str(output),
                              "--visualization-dir", str(images)]), \
                    patch.dict(sys.modules, inspect_objects=backend), \
                    patch.object(evaluate_dataset, "save_visualizations") as save, \
                    redirect_stdout(io.StringIO()):
                evaluate_dataset.main()
            save.assert_called_once_with(root / "images/val/one.png", prediction,
                                         images / "dataset_val")
            self.assertTrue((output / "dataset_val_results.csv").is_file())

    def test_both_splits_load_once_and_limit_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            for split in ("train", "val"):
                for kind in ("images", "labels", "adj"):
                    folder = root / kind / split
                    folder.mkdir(parents=True)
                    for name in ("one", "two"):
                        (folder / f"{name}.{'png' if kind == 'images' else 'txt'}").touch()
            model = object()
            backend = SimpleNamespace(
                load_model=Mock(return_value=model),
                evaluate_one_image=Mock(return_value=dict(
                    tp=1, fp=0, fn=0, missing_objects=0, extra_detections=0,
                    extra_relation_fp=0, image_correct=1)),
            )
            output = Path(directory) / "results"
            argv = ["evaluate_dataset.py", "--dataset-root", str(root),
                    "--split", "both", "--limit", "1", "--output-dir", str(output),
                    "--no-visualizations"]
            with patch.object(sys, "argv", argv), patch.dict(
                sys.modules, inspect_objects=backend
            ), redirect_stdout(io.StringIO()):
                evaluate_dataset.main()
            backend.load_model.assert_called_once()
            self.assertEqual(backend.evaluate_one_image.call_count, 2)
            for call, split in zip(backend.evaluate_one_image.call_args_list, ("train", "val")):
                self.assertEqual(call.args, (root / "images" / split / "one.png", model))
                with (output / f"dataset_{split}_results.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["tp"], "1")
                self.assertEqual((output / f"dataset_{split}_errors.txt").read_text(), "")

    def test_folder_is_required_before_backend_import(self):
        with patch.object(sys, "argv", ["evaluate_vmrd.py"]), redirect_stderr(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit) as error:
                evaluate_dataset.main()
        self.assertEqual(error.exception.code, 2)

    def test_png_dataset_and_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "topdown_labeled"
            for kind in ("images", "labels", "adj"):
                (root / kind / "train").mkdir(parents=True)
            for name in ("one.png", "two.JPEG", "three.jpg"):
                (root / "images/train" / name).touch()
                for kind in ("labels", "adj"):
                    (root / kind / "train" / f"{Path(name).stem}.txt").touch()
            backend = SimpleNamespace(
                load_model=Mock(return_value=object()),
                evaluate_one_image=Mock(
                    return_value=dict(
                        tp=1,
                        fp=0,
                        fn=0,
                        missing_objects=0,
                        extra_detections=0,
                        extra_relation_fp=0,
                        image_correct=1,
                    )
                ),
            )
            output = Path(directory) / "output/nested"
            argv = [
                "evaluate_vmrd.py",
                "--dataset-root",
                str(root),
                "--split",
                "train",
                "--checkpoint",
                "custom.pth",
                "--output-dir",
                str(output),
                "--no-visualizations",
            ]
            with patch.object(sys, "argv", argv), patch.dict(
                sys.modules, inspect_objects=backend
            ), redirect_stdout(io.StringIO()):
                evaluate_dataset.main()
            backend.load_model.assert_called_once_with(Path("custom.pth"))
            self.assertEqual(backend.evaluate_one_image.call_count, 3)
            with (output / "topdown_labeled_train_results.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                {row["image"] for row in rows}, {"one.png", "two.JPEG", "three.jpg"}
            )
            self.assertEqual(
                (output / "topdown_labeled_train_errors.txt").read_text(), ""
            )

    def test_missing_annotations_fail_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("images", "labels", "adj"):
                (root / kind / "val").mkdir(parents=True)
            (root / "images/val/example.png").touch()
            backend = SimpleNamespace(load_model=Mock(), evaluate_one_image=Mock())
            stderr = io.StringIO()
            with patch.object(
                sys, "argv", ["evaluate_vmrd.py", "--folder", str(root)]
            ), patch.dict(sys.modules, inspect_objects=backend), redirect_stderr(
                stderr
            ):
                with self.assertRaises(SystemExit) as error:
                    evaluate_dataset.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Missing annotation", stderr.getvalue())
            backend.load_model.assert_not_called()

    def test_default_visualizations_use_predictions_and_keep_split_outputs_separate(self):
        import torch
        from PIL import Image, ImageChops
        from infer_unlabeled_dataset import decode_predictions

        outputs = {
            "pred_logits": torch.tensor([[[-9., 9.], [9., -9.]]]),
            "pred_boxes": torch.tensor([[[.2, .3, .2, .2], [.8, .7, .2, .2]]]),
            "pred_adj": torch.tensor([[[-9., 9.], [-9., -9.]]]),
        }
        prediction = decode_predictions(outputs, 320, 240, .4, .5, 100,
                                        class_names=["original", "new class"])
        self.assertEqual(prediction["detections"][0]["class_name"], "new class")
        self.assertEqual(len(prediction["relationships"]), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            for split in ("train", "val"):
                for kind in ("images", "labels", "adj"):
                    folder = root / kind / split
                    folder.mkdir(parents=True)
                    if kind == "images":
                        Image.new("RGB", (320, 240), "white").save(folder / "one.png")
                    else:
                        (folder / "one.txt").touch()
            model = object()
            backend = SimpleNamespace(
                load_model=Mock(return_value=model),
                evaluate_one_image=Mock(return_value=dict(
                    tp=1, fp=0, fn=0, missing_objects=0, extra_detections=0,
                    extra_relation_fp=0, image_correct=1, prediction=prediction)),
            )
            output = Path(directory) / "results"
            with patch.object(sys, "argv", ["evaluate_dataset.py", "--dataset-root",
                              str(root), "--split", "both", "--output-dir", str(output)]), \
                    patch.dict(sys.modules, inspect_objects=backend), \
                    redirect_stdout(io.StringIO()):
                evaluate_dataset.main()
            backend.load_model.assert_called_once()
            self.assertEqual(backend.evaluate_one_image.call_count, 2)
            for split in ("train", "val"):
                backend.evaluate_one_image.assert_any_call(
                    root / "images" / split / "one.png", model, return_predictions=True)
                images = output / "visualizations" / f"dataset_{split}"
                with Image.open(images / "one.png_objects.jpg") as objects, \
                        Image.open(images / "one.png_relationships.jpg") as relationships:
                    self.assertEqual(objects.size, (320, 240))
                    self.assertIsNotNone(ImageChops.difference(objects, relationships).getbbox())
                with (output / f"dataset_{split}_results.csv").open() as file:
                    self.assertEqual(next(csv.DictReader(file))["tp"], "1")
                self.assertEqual((output / f"dataset_{split}_errors.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
