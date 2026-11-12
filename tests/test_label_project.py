import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
from scripts.label_project import Project, grasp_rectangle, validate
from util.dataset_config import configure_classes, get_num_classes


class LabelProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        for name in ('1.png', '2.png', '3.png', 'density_map.png'):
            Image.new('RGB', (100, 80)).save(self.images / name)
        self.project = Project(self.images, self.root / 'project.json')

    def annotate(self, name):
        sample = self.project.sample(name)
        sample.update(objects=[{'id': 4, 'class_id': 31, 'box': [5, 8, 45, 60]},
                               {'id': 9, 'class_id': 0, 'box': [50, 10, 90, 70]}],
                      grasps=[{'object_id': 4, 'difficulty': 'e', 'points': grasp_rectangle([10, 20], [35, 35], 5)},
                              {'object_id': 9, 'difficulty': 'h', 'points': grasp_rectangle([55, 20], [85, 20], 5)}],
                      edges=[[9, 4]], relations_reviewed=True, complete=True)
        return sample

    def ready(self):
        self.project.add_class('custom part')
        for name in self.project.files:
            self.annotate(name)
        self.project.set_test_image('3.png', True)

    def test_registry_and_progress_survive_reopen(self):
        self.assertEqual(self.project.files, ['1.png', '2.png', '3.png'])
        self.assertEqual(self.project.add_class('custom part'), 31)
        self.assertEqual(self.project.add_class('second part'), 32)
        with self.assertRaises(ValueError):
            self.project.add_class('CUSTOM PART')
        self.annotate('1.png')
        self.project.save()
        reopened = Project(self.images, self.project.path)
        self.assertEqual(reopened.data, self.project.data)
        wrong = self.root / 'other'
        wrong.mkdir()
        Image.new('RGB', (10, 10)).save(wrong / '1.png')
        with self.assertRaisesRegex(ValueError, 'folder'):
            Project(wrong, self.project.path)

    def test_export_alignment_and_grasp_reclassification(self):
        self.ready()
        self.project.sample('1.png')['objects'][0]['class_id'] = 17
        output = self.project.export(self.root / 'dataset')
        self.assertEqual(json.loads((output / 'classes.json').read_text())[31], 'custom part')
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual({v['split'] for v in manifest['samples'].values()}, {'train', 'val', 'test'})
        for stem, entry in manifest['samples'].items():
            split = entry['split']
            self.assertTrue(stem.isdigit())
            self.assertEqual(entry['object_ids_in_row_order'], [4, 9])
            self.assertEqual((output / 'adj' / split / f'{stem}.txt').read_text(), '0 0\n1 0\n')
            label = (output / 'labels' / split / f'{stem}.txt').read_text().splitlines()[0].split()
            grasp = (output / 'grasps' / split / f'{stem}.txt').read_text().splitlines()[0].split()
            self.assertEqual(len(label), 7)
            self.assertEqual(len(grasp), 11)
            self.assertEqual(grasp[-2], '4')
            self.assertEqual(grasp[-1], label[0])
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            self.project.export(output)
        other = self.project.export(self.root / 'dataset2')
        self.assertEqual(manifest, json.loads((other / 'manifest.json').read_text()))

    def test_incomplete_not_exported_and_invalid_completed_rejected(self):
        self.ready()
        self.project.sample('2.png')['complete'] = False
        with self.assertRaisesRegex(ValueError, 'three images'):
            self.project.export(self.root / 'dataset')
        self.project.sample('2.png')['complete'] = True
        self.project.sample('2.png')['grasps'] = []
        with self.assertRaisesRegex(ValueError, '2.png.*grasp'):
            self.project.export(self.root / 'dataset')
        self.assertFalse((self.root / 'dataset').exists())

    def test_invalid_annotations_rejected(self):
        self.ready()
        sample = self.project.sample('1.png')
        for mutation, message in [
            (lambda s: s['edges'].append([4, 9]), 'cycle'),
            (lambda s: s['edges'].append([4, 4]), 'different'),
            (lambda s: s['edges'].append([4, 8]), 'existing'),
            (lambda s: s['objects'][0].update(box=[5, 5, 101, 50]), 'Boxes'),
            (lambda s: s['objects'][0].update(class_id=500), 'class'),
            (lambda s: s['grasps'][0].update(object_id=123), 'existing'),
            (lambda s: s['grasps'][0].update(points=[[1, 1]]*4), 'edges'),
            (lambda s: s.update(relations_reviewed=False), 'reviewed'),
        ]:
            with self.subTest(message=message):
                bad = copy.deepcopy(sample)
                mutation(bad)
                with self.assertRaisesRegex(ValueError, message):
                    validate(bad, self.project.classes, (100, 80), complete=True)

    def test_rectangle_angle_and_corners(self):
        points = grasp_rectangle([10, 10], [40, 40], 8)
        center = [sum(p[j] for p in points)/4 for j in range(2)]
        midpoint = [(points[1][j]+points[2][j])/2 for j in range(2)]
        self.assertEqual(center, [25, 25])
        self.assertEqual(midpoint, [40, 40])
        with self.assertRaises(ValueError):
            grasp_rectangle([10, 10], [10, 10], 5)

    def test_class_configuration(self):
        self.ready()
        self.project.add_class('another part')
        output = self.project.export(self.root / 'dataset')
        args = SimpleNamespace(dataset_root=output, num_classes=None)
        self.assertEqual(configure_classes(args), 33)
        self.assertEqual(args.class_names[32], 'another part')
        self.assertEqual(get_num_classes(SimpleNamespace()), 32)
        args.num_classes = 32
        with self.assertRaises(ValueError):
            configure_classes(args)

    def test_real_dataset_loader_and_both_transforms(self):
        # This verifies the consumer, including sparse object IDs and corner order.
        from datasets.vmrd import CustomDataset, make_vmrd_transforms
        self.ready()
        output = self.project.export(self.root / 'dataset')
        for split in ('train', 'val', 'test'):
            dataset = CustomDataset(*(str(output / folder / split) for folder in
                                      ('images', 'labels', 'adj', 'grasps')),
                                    transform=make_vmrd_transforms(split))
            image, target = dataset[0]
            self.assertEqual(image.shape[0], 3)
            self.assertEqual(target['boxes'].shape, (2, 4))
            self.assertEqual(target['grasp_points'].shape, (2, 4, 2))
            self.assertEqual(target['labels'].tolist(), [31, 0])
            self.assertEqual(target['grasp_classes'].tolist(), [31, 0])
            self.assertEqual(target['adj'].tolist(), [[0, 0], [1, 0]])
            self.assertTrue(((target['grasp_angles_cls'] >= 0) & (target['grasp_angles_cls'] <= 18)).all())


if __name__ == '__main__':
    unittest.main()
