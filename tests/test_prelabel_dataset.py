import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from prelabel_dataset import check_classes, make_sample, read_classes, write_project
from scripts.label_project import Project
from scripts.convert_vmrd import CLASSES


class PrelabelTests(unittest.TestCase):
    def test_registry_order_is_exact_and_missing_metadata_rejected(self):
        names = list(CLASSES) + ['wood', 'plate']
        check_classes(names, names.copy())
        for wrong in (None, names[:-1], list(CLASSES) + ['plate', 'wood']):
            with self.assertRaisesRegex(ValueError, 'exactly match'):
                check_classes(names, wrong)

    def test_predictions_reopen_in_unmodified_labeler(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / 'images'
            images.mkdir()
            Image.new('RGB', (100, 80)).save(images / '1.png')
            names = list(CLASSES) + ['wood', 'plate']
            source = root / 'classes.json'
            source.write_text(json.dumps(names))
            project = Project(images, root / 'project.json')
            project.data['classes'] = read_classes(source)
            project.data['require_grasps'] = False
            prediction = {'detections': [
                {'id': 0, 'class_id': 32, 'bbox_xyxy': [1, 2, 30, 40]},
                {'id': 1, 'class_id': 31, 'bbox_xyxy': [10, 20, 60, 70]},
                {'id': 2, 'class_id': 99, 'bbox_xyxy': [1, 2, 30, 40]},
                {'id': 3, 'class_id': 0, 'bbox_xyxy': [10, 2, 10, 40]},
            ], 'relationships': [
                {'above_id': 0, 'below_id': 1, 'confidence': .9},
                {'above_id': 1, 'below_id': 0, 'confidence': .6},
                {'above_id': 0, 'below_id': 2, 'confidence': .8},
            ]}
            original = copy.deepcopy(prediction)
            sample, skipped, rejected = make_sample(prediction, names, (100, 80))
            self.assertEqual(prediction, original)
            self.assertEqual((skipped, rejected), (2, 2))
            self.assertEqual([o['class_id'] for o in sample['objects']], [32, 31])
            self.assertEqual(sample['edges'], [[1, 2]])
            self.assertFalse(sample['complete'])
            self.assertFalse(sample['relations_reviewed'])
            self.assertEqual(sample['grasps'], [])
            project.data['samples']['1.png'] = sample
            write_project(project.path, project.data)
            reopened = Project(images, project.path)
            self.assertEqual(reopened.data, project.data)
            self.assertEqual(read_classes(project.path), names)
            before = project.path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_project(project.path, {})
            self.assertEqual(project.path.read_bytes(), before)

    def test_three_object_cycle_is_removed(self):
        prediction = {
            'detections': [{'id': i, 'class_id': 0, 'bbox_xyxy': [0, 0, 20, 20]}
                           for i in range(3)],
            'relationships': [{'above_id': a, 'below_id': b, 'confidence': c}
                              for a, b, c in [(2, 0, .6), (0, 1, .9), (1, 2, .8)]]}
        sample, _, rejected = make_sample(prediction, list(CLASSES), (100, 80))
        self.assertEqual(sample['edges'], [[1, 2], [2, 3]])
        self.assertEqual(rejected, 1)


if __name__ == '__main__':
    unittest.main()
