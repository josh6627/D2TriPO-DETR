"""Exercise the real evaluator with known predictions, independent of a GPU."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

import inspect_objects


BOX_A = [10., 10., 30., 30.]
BOX_B = [60., 60., 80., 80.]
BOX_C = [40., 10., 50., 20.]


class RelationshipMetricTests(unittest.TestCase):
    def evaluate(self, gt_objects, gt_adj, detections, edges, symlink=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'dataset'
            for kind in ('images', 'labels', 'adj'):
                (root / kind / 'test').mkdir(parents=True)
            (root / 'classes.json').write_text(json.dumps(['a', 'b', 'c']))
            image_path = root / 'images/test/one.png'
            if symlink:
                original = Path(temp) / 'original.png'
                Image.new('RGB', (100, 100)).save(original)
                image_path.symlink_to(original)
            else:
                Image.new('RGB', (100, 100)).save(image_path)
            (root / 'labels/test/one.txt').write_text(''.join(
                ' '.join(map(str, [class_id, *box, 100, 100])) + '\n'
                for class_id, box in gt_objects))
            (root / 'adj/test/one.txt').write_text(''.join(
                ' '.join(map(str, row)) + '\n' for row in gt_adj))
            n = len(detections)
            logits = torch.full((1, n, 3), -20.)
            boxes = torch.empty((1, n, 4))
            for query, (class_id, box) in enumerate(detections):
                logits[0, query, class_id] = 10. - query * .1
                x1, y1, x2, y2 = box
                boxes[0, query] = torch.tensor([(x1+x2)/200, (y1+y2)/200,
                                                (x2-x1)/100, (y2-y1)/100])
            adj = torch.full((1, n, n), -20.)
            for above, below in edges:
                adj[0, above, below] = 10.
            model = Mock(return_value={'pred_logits': logits, 'pred_boxes': boxes, 'pred_adj': adj})
            model.class_names = ['a', 'b', 'c']
            with patch.object(inspect_objects, 'device', torch.device('cpu')), \
                    patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                    patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = inspect_objects.evaluate_one_image(image_path, model, return_predictions=True)
            # Displayed and scored arrows must describe exactly the same predictions.
            self.assertEqual(len(result['prediction']['relationships']), result['tp'] + result['fp'])
            self.assertEqual(result['tp'] + result['fn'], sum(map(sum, gt_adj)))
            self.assertEqual(result['total_objects'], len(gt_objects))
            self.assertEqual(result['bbox_tp'] + result['bbox_fn'], sum(map(sum, gt_adj)))
            self.assertEqual(result['bbox_tp'] + result['bbox_fp'],
                             len(result['prediction']['relationships']))
            return result

    def test_correct_arrow_with_permuted_queries(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(1, BOX_B), (0, BOX_A)], [(1, 0)])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['image_correct']), (1, 0, 0, 1))

    def test_reversed_arrow_is_false_positive_and_false_negative(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(0, BOX_A), (1, BOX_B)], [(1, 0)])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))

    def test_wrong_class_with_correct_geometry_is_not_a_true_positive(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(2, BOX_A), (1, BOX_B)], [(0, 1)])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))
        self.assertEqual((result['missing_objects'], result['extra_detections']), (1, 1))
        self.assertEqual((result['bbox_tp'], result['bbox_fp'], result['bbox_fn']), (1, 0, 0))
        self.assertEqual((result['bbox_missing_objects'], result['bbox_extra_detections'],
                          result['bbox_image_correct']), (0, 0, 1))

    def test_box_only_reversed_edge_with_wrong_classes(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(2, BOX_B), (2, BOX_A)], [(0, 1)])
        self.assertEqual((result['bbox_tp'], result['bbox_fp'], result['bbox_fn']), (0, 1, 1))
        self.assertEqual(result['bbox_image_correct'], 0)

    def test_box_only_requires_overlap_even_when_classes_match(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(0, BOX_C), (1, BOX_B)], [(0, 1)])
        self.assertEqual((result['bbox_tp'], result['bbox_fp'], result['bbox_fn']), (0, 1, 1))
        self.assertEqual(result['bbox_extra_relation_fp'], 1)

    def test_box_only_duplicate_boxes_cannot_match_same_object_twice(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(2, BOX_A), (2, BOX_A), (2, BOX_B)], [(0, 2), (1, 2)])
        self.assertEqual((result['bbox_tp'], result['bbox_fp'], result['bbox_fn']), (1, 1, 0))
        self.assertEqual(result['bbox_extra_detections'], 1)
        self.assertEqual(result['bbox_extra_relation_fp'], 1)

    def test_missing_endpoint_and_no_predictions(self):
        for detections, missing in (([(0, BOX_A)], 1), ([], 2)):
            with self.subTest(detections=detections):
                result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]], detections, [])
                self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 0, 1))
                self.assertEqual(result['missing_objects'], missing)
                self.assertEqual((result['bbox_tp'], result['bbox_fp'], result['bbox_fn']), (0, 0, 1))
                self.assertEqual(result['bbox_missing_objects'], missing)

    def test_edge_between_two_extra_detections_is_counted_once(self):
        result = self.evaluate([(0, BOX_A)], [[0]], [(0, BOX_A), (2, BOX_B), (2, BOX_C)], [(1, 2)])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['extra_relation_fp']), (0, 1, 0, 1))

    def test_same_class_instances_use_boxes_and_correct_query_relationship(self):
        result = self.evaluate([(0, BOX_A), (0, BOX_B)], [[0, 1], [0, 0]],
                               [(0, BOX_B), (0, BOX_A)], [(1, 0)])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['image_correct']), (1, 0, 0, 1))

    def test_single_object_correct_without_edges_and_self_arrows_ignored(self):
        result = self.evaluate([(0, BOX_A)], [[0]], [(0, BOX_A)], [(0, 0)])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['image_correct']), (0, 0, 0, 1))

    def test_symlinked_image_uses_dataset_annotations(self):
        result = self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 1], [0, 0]],
                               [(0, BOX_A), (1, BOX_B)], [(0, 1)], symlink=True)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 0, 0))

    def test_malformed_adjacency_rejected(self):
        with self.assertRaisesRegex(ValueError, 'adjacency shape'):
            self.evaluate([(0, BOX_A), (1, BOX_B)], [[0]], [(0, BOX_A)], [])
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.evaluate([(0, BOX_A), (1, BOX_B)], [[0, 2], [0, 0]], [(0, BOX_A)], [])


if __name__ == '__main__':
    unittest.main()
