"""A merged checkpoint must match source annotations by class name, not raw ID."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

import inspect_objects
from inspect_objects import evaluate_relationships, match_objects
from scripts.convert_vmrd import CLASSES
from util.dataset_config import class_id_mapping


class EvaluationClassMappingTests(unittest.TestCase):
    def test_evaluator_reads_source_labels_in_checkpoint_class_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in ('images', 'labels', 'adj'):
                (root / kind / 'test').mkdir(parents=True)
            (root / 'classes.json').write_text(json.dumps(['plate', 'cup']))
            image_path = root / 'images/test/one.png'
            Image.new('RGB', (100, 100), 'white').save(image_path)
            (root / 'labels/test/one.txt').write_text('0 10 10 30 30 100 100\n1 60 60 80 80 100 100\n')
            (root / 'adj/test/one.txt').write_text('0 1\n0 0\n')
            logits = torch.full((1, 5, 32), -20.)
            logits[0, 0, 31] = 10.  # plate in checkpoint
            logits[0, 1, 17] = 10.  # cup in checkpoint
            relationships = torch.full((1, 5, 5), -20.)
            relationships[0, 0, 1] = 10.
            outputs = {'pred_logits': logits, 'pred_adj': relationships,
                       'pred_boxes': torch.tensor([[[.2, .2, .2, .2], [.7, .7, .2, .2],
                                                    [.5, .5, .1, .1], [.5, .5, .1, .1], [.5, .5, .1, .1]]])}
            model = Mock(return_value=outputs)
            model.class_names = [*CLASSES, 'plate']
            with patch.object(inspect_objects, 'device', torch.device('cpu')), \
                    patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                    patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = inspect_objects.evaluate_one_image(image_path, model)
            self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 0, 0))
            self.assertEqual(result['missing_objects'], 0)
            self.assertEqual(result['extra_detections'], 0)
            self.assertEqual(result['image_correct'], 1)

    def test_reordered_source_ids_match_objects_and_relationships(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'classes.json').write_text(json.dumps(['plate', 'cup']))
            mapping = class_id_mapping(root, ['cup', 'plate'])
            self.assertEqual(mapping, {0: 1, 1: 0})
            boxes = [[10., 10., 30., 30.], [60., 60., 80., 80.]]
            outputs = {'pred_boxes': torch.tensor([[[.2, .2, .2, .2], [.7, .7, .2, .2]]])}
            # Query 0 predicts plate (checkpoint ID 1), query 1 predicts cup (ID 0).
            indices = torch.tensor([1, 2])
            scores = torch.tensor([.99, .98])
            keep = torch.tensor([True, True])
            source_objects = [{'class_id': i, 'box': box} for i, box in enumerate(boxes)]
            wrong, _ = match_objects(source_objects, outputs, scores, indices, keep, 2, 100, 100)
            self.assertEqual(wrong, {})
            objects = [{**obj, 'class_id': mapping[obj['class_id']]} for obj in source_objects]
            matched, _ = match_objects(objects, outputs, scores, indices, keep, 2, 100, 100)
            self.assertEqual({i: obj['query_id'] for i, obj in matched.items()}, {0: 0, 1: 1})
            # Correct class mapping keeps the annotated object/adjacency row order.
            self.assertEqual(evaluate_relationships([[0, 1], [0, 0]], [[0, 1], [0, 0]]), (1, 0, 0))

    def test_conversion_registry_normalization_and_legacy_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIsNone(class_id_mapping(root, ['cup']))
            (root / 'conversion_report.json').write_text(json.dumps({'class_ids': {' CUP ': 0, 'plate': 1}}))
            self.assertEqual(class_id_mapping(root, ['plate', 'cup']), {0: 1, 1: 0})

    def test_unknown_classes_are_not_silently_matched_by_numeric_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'classes.json').write_text(json.dumps(['plate', 'cup']))
            self.assertEqual(class_id_mapping(root, ['banana', 'cup']), {1: 1})
            (root / 'classes.json').write_text(json.dumps(['cup', ' CUP ']))
            with self.assertRaisesRegex(ValueError, 'unique'):
                class_id_mapping(root, ['cup'])

    def test_correct_image_without_edges_contributes_no_edge_true_positives(self):
        self.assertEqual(evaluate_relationships([[0]], [[0]]), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
