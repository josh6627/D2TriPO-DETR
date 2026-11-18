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


BBOX_RESULT = dict(total_objects=2, bbox_tp=1, bbox_fp=0, bbox_fn=0, bbox_missing_objects=0,
                   bbox_extra_detections=0, bbox_extra_relation_fp=0,
                   bbox_image_correct=1)


class EvaluateDatasetTests(unittest.TestCase):
    def test_typed_relation_counts_are_saved_and_aggregated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = dict(tp=1, fp=1, fn=1, missing_objects=0,
                          extra_detections=0, extra_relation_fp=0, image_correct=0,
                          **BBOX_RESULT)
            for prefix in ('', 'bbox_'):
                result.update({f'{prefix}relation_1_tp': 1, f'{prefix}relation_1_fp': 1,
                               f'{prefix}relation_1_fn': 0, f'{prefix}relation_2_tp': 0,
                               f'{prefix}relation_2_fp': 0, f'{prefix}relation_2_fn': 1})
            with redirect_stdout(io.StringIO()):
                evaluate_dataset.evaluate_split(
                    [root/'one.png', root/'two.png'], object(), Mock(return_value=result),
                    root, 'test', save_images=False)
            folder = root/'visualizations/test'
            with (folder/'results.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]['relation_2_fn'], '1')
            self.assertEqual(rows[0]['bbox_relation_1_fp'], '1')
            with (folder/'summary.csv').open() as file:
                summary = {row['evaluation']: row for row in csv.DictReader(file)}
            for prefix in ('class_aware', 'box_only'):
                stacking = summary[f'{prefix}_relation_1']
                obstruction = summary[f'{prefix}_relation_2']
                self.assertEqual((stacking['tp'], stacking['fp'], stacking['fn']), ('2', '2', '0'))
                self.assertAlmostEqual(float(stacking['f1']), 2/3)
                self.assertEqual((obstruction['tp'], obstruction['fn']), ('0', '2'))
                self.assertEqual(obstruction['image_accuracy'], '')

    def test_visualization_names_obstruction_and_preserves_direction(self):
        from PIL import Image, ImageDraw
        from infer_unlabeled_dataset import visualize

        objects = [dict(id=0, class_name='tool', confidence=0.9, bbox_xyxy=[10,10,30,30]),
                   dict(id=1, class_name='cup', confidence=0.9, bbox_xyxy=[70,70,90,90])]
        rel = dict(above_id=0, below_id=1, relation_class=2, confidence=0.7)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'one.png'
            Image.new('RGB', (100,100), 'white').save(image)
            original_text = ImageDraw.ImageDraw.text
            text_lines = []
            def capture(draw, xy, text, *args, **kwargs):
                text_lines.append(text)
                return original_text(draw, xy, text, *args, **kwargs)
            with patch.object(ImageDraw.ImageDraw, 'text', capture), \
                    patch('infer_unlabeled_dataset.visualize', wraps=visualize) as render:
                evaluate_dataset.save_visualizations(
                    image, dict(detections=objects, relationships=[rel]), root/'output')
            self.assertTrue(any('[2] #0 tool obstructs grasp of #1 cup' in line for line in text_lines))
            self.assertEqual(render.call_args.args[2], [rel])
            self.assertTrue((root/'output/one.png_relationships.jpg').exists())

    def test_box_only_metrics_are_saved_and_aggregated_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = dict(total_objects=3, tp=0, fp=1, fn=1, missing_objects=1,
                          extra_detections=1, extra_relation_fp=1, image_correct=0,
                          bbox_tp=1, bbox_fp=0, bbox_fn=0, bbox_missing_objects=0,
                          bbox_extra_detections=0, bbox_extra_relation_fp=0,
                          bbox_image_correct=1)
            second_result = dict(result, total_objects=5, tp=1, fp=0, fn=2,
                                 bbox_tp=2, bbox_fp=1, bbox_fn=1, bbox_image_correct=0)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                evaluate_dataset.evaluate_split(
                    [root / 'one.png', root / 'two.png'], object(),
                    Mock(side_effect=[result, second_result]), root, 'test', save_images=False)
            with (root / 'visualizations/test/results.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row['total_objects'] for row in rows], ['3', '5'])
            self.assertIn('Total objects: 8\nTotal missing objects: 2', stdout.getvalue())
            self.assertEqual(rows[0]['tp'], '0')
            self.assertEqual(rows[0]['bbox_tp'], '1')
            self.assertEqual(rows[0]['bbox_image_correct'], '1')
            self.assertEqual(rows[0]['precision'], '0.0')
            self.assertEqual(rows[0]['recall'], '0.0')
            self.assertEqual(rows[0]['image_accuracy'], '0.0')
            self.assertEqual(rows[0]['bbox_precision'], '1.0')
            self.assertEqual(rows[0]['bbox_recall'], '1.0')
            self.assertEqual(rows[0]['bbox_image_accuracy'], '1.0')
            with (root / 'visualizations/test/summary.csv').open() as file:
                summary = {row['evaluation']: row for row in csv.DictReader(file)}
            self.assertEqual(float(summary['class_aware']['recall']), 0.25)
            self.assertEqual(float(summary['class_aware']['precision']), 0.5)
            self.assertEqual(float(summary['class_aware']['image_accuracy']), 0.0)
            self.assertEqual(float(summary['box_only']['recall']), 0.75)
            self.assertEqual(float(summary['box_only']['precision']), 0.75)
            self.assertEqual(float(summary['box_only']['image_accuracy']), 0.5)
            self.assertIn('Class-aware micro edge recall: 25.0', stdout.getvalue())
            self.assertIn('Class-aware micro edge precision: 50.0', stdout.getvalue())
            self.assertIn('Class-aware image accuracy: 0.0', stdout.getvalue())
            self.assertIn('Box-only image accuracy: 50.0', stdout.getvalue())
            for metric in ('recall', 'precision', 'F1'):
                self.assertIn(f'Box-only micro edge {metric}: 75.0', stdout.getvalue())

    def test_undefined_scores_remain_unavailable(self):
        self.assertEqual(evaluate_dataset.calculate_scores(0, 0, 0, 0, 0),
                         dict(recall=None, precision=None, f1=None, image_accuracy=None))
        self.assertEqual(evaluate_dataset.calculate_scores(0, 0, 0, 1, 1),
                         dict(recall=None, precision=None, f1=None, image_accuracy=1.0))

    def test_visualization_failure_does_not_remove_successful_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'one.png'
            image.touch()
            evaluate = Mock(return_value=dict(tp=1, fp=0, fn=0, missing_objects=0,
                                             extra_detections=0, extra_relation_fp=0,
                                             image_correct=1, **BBOX_RESULT, prediction={}))
            stdout = io.StringIO()
            with patch.object(evaluate_dataset, 'save_visualizations', side_effect=OSError('disk error')), \
                    redirect_stdout(stdout):
                evaluate_dataset.evaluate_split([image], object(), evaluate, root, 'test')
            with (root / 'visualizations/test/results.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['tp'], '1')
            self.assertIn('Images evaluated: 1', stdout.getvalue())
            self.assertIn('Images failed: 0', stdout.getvalue())
            self.assertIn('VISUALIZATION: one.png', (root / 'visualizations/test/errors.txt').read_text())

    def test_all_failed_images_replace_stale_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / 'visualizations/test/results.csv'
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text('image,tp\nold.png,99\n')
            with redirect_stdout(io.StringIO()):
                evaluate_dataset.evaluate_split([root / 'one.png'], object(),
                                                Mock(side_effect=ValueError('bad annotation')),
                                                root, 'test', save_images=False)
            with csv_path.open() as file:
                self.assertEqual(list(csv.DictReader(file)), [])
            with (csv_path.parent / 'summary.csv').open() as file:
                for row in csv.DictReader(file):
                    self.assertEqual(row['images_evaluated'], '0')
                    self.assertEqual(row['precision'], '')
                    self.assertEqual(row['recall'], '')
                    self.assertEqual(row['image_accuracy'], '')
            self.assertNotIn('old.png', csv_path.read_text())
            self.assertIn('bad annotation', (root / 'visualizations/test/errors.txt').read_text())

    def test_test_split_uses_test_images_annotations_and_output_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            for split in ("val", "test"):
                for kind in ("images", "labels", "adj"):
                    folder = root / kind / split
                    folder.mkdir(parents=True)
                    (folder / ("one.png" if kind == "images" else "one.txt")).touch()
            model = object()
            prediction = {"detections": [], "relationships": []}
            backend = SimpleNamespace(
                load_model=Mock(return_value=model),
                evaluate_one_image=Mock(return_value=dict(
                    tp=1, fp=0, fn=0, missing_objects=0, extra_detections=0,
                    extra_relation_fp=0, image_correct=1, **BBOX_RESULT, prediction=prediction)),
            )
            output = Path(directory) / "results"
            argv = ["evaluate_dataset.py", "--dataset-root", str(root),
                    "--split", "test", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, inspect_objects=backend), \
                    patch.object(evaluate_dataset, "save_visualizations") as save, \
                    redirect_stdout(io.StringIO()):
                evaluate_dataset.main()
            backend.load_model.assert_called_once()
            backend.evaluate_one_image.assert_called_once_with(
                root / "images/test/one.png", model, return_predictions=True)
            save.assert_called_once_with(root / "images/test/one.png", prediction,
                                         output / "visualizations/dataset_test")
            self.assertTrue((output / "visualizations/dataset_test/results.csv").is_file())
            self.assertEqual((output / "visualizations/dataset_test/errors.txt").read_text(), "")
            self.assertFalse((output / "visualizations/dataset_val/results.csv").exists())

            # Existing validation labels cannot substitute for missing test labels.
            (root / "labels/test/one.txt").unlink()
            backend.load_model.reset_mock()
            stderr = io.StringIO()
            with patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, inspect_objects=backend), \
                    redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                evaluate_dataset.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(str(root / "labels/test/one.txt"), stderr.getvalue())
            backend.load_model.assert_not_called()

    def test_missing_test_split_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images/val").mkdir(parents=True)
            backend = SimpleNamespace(load_model=Mock(), evaluate_one_image=Mock())
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["evaluate_dataset.py", "--dataset-root",
                              str(root), "--split", "test"]), \
                    patch.dict(sys.modules, inspect_objects=backend), \
                    redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                evaluate_dataset.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(str(root / "images/test"), stderr.getvalue())
            backend.load_model.assert_not_called()

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
                    extra_relation_fp=0, image_correct=1, **BBOX_RESULT, prediction=prediction)),
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
            self.assertTrue((images / "dataset_val/results.csv").is_file())
            self.assertTrue((images / "dataset_val/errors.txt").is_file())
            self.assertFalse((output / "dataset_val_results.csv").exists())

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
                    extra_relation_fp=0, image_correct=1, **BBOX_RESULT)),
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
                with (output / "visualizations" / f"dataset_{split}" / "results.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["tp"], "1")
                self.assertEqual((output / "visualizations" / f"dataset_{split}" / "errors.txt").read_text(), "")

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
                        image_correct=1, **BBOX_RESULT,
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
            with (output / "visualizations/topdown_labeled_train/results.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                {row["image"] for row in rows}, {"one.png", "two.JPEG", "three.jpg"}
            )
            self.assertEqual(
                (output / "visualizations/topdown_labeled_train/errors.txt").read_text(), ""
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
                    extra_relation_fp=0, image_correct=1, **BBOX_RESULT, prediction=prediction)),
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
                self.assertEqual({p.name for p in images.iterdir()},
                                 {"one.png_relationships.jpg", "results.csv", "summary.csv", "errors.txt"})
                with Image.open(images / "one.png_relationships.jpg") as relationships:
                    self.assertGreater(relationships.width, 320)
                    self.assertGreaterEqual(relationships.height, 240)
                    self.assertIsNotNone(ImageChops.difference(
                        Image.new("RGB", (320, 240), "white"),
                        relationships.crop((0, 0, 320, 240))).getbbox())
                with (images / "results.csv").open() as file:
                    self.assertEqual(next(csv.DictReader(file))["tp"], "1")
                self.assertEqual((images / "errors.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
