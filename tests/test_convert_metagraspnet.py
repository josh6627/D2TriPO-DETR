import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.convert_metagraspnet import (
    CLASS_IDS, convert_order, convert_sample, grasp_rectangle, load_classes, main,
)


class ConversionTests(unittest.TestCase):
    def test_contact_axis_matches_loader_angle_and_opening(self):
        for pair in ([[2, 4], [12, 4]], [[7, 2], [7, 12]], [[2, 3], [8, 11]]):
            pair = np.array(pair, dtype=float)
            rectangle = grasp_rectangle(pair, 4)
            np.testing.assert_allclose(rectangle.mean(axis=0), pair.mean(axis=0))
            np.testing.assert_allclose((rectangle[0] + rectangle[3]) / 2, pair[0])
            np.testing.assert_allclose((rectangle[1] + rectangle[2]) / 2, pair[1])
            self.assertAlmostEqual(np.linalg.norm(rectangle[1] - rectangle[0]), 10)
            self.assertAlmostEqual(np.linalg.norm(rectangle[2] - rectangle[1]), 4)

    def test_order_direction_and_noncontiguous_visible_instances(self):
        # Instance 3 sits above instance 1, while instance 2 is invisible.
        order = np.array([[0, 0, -1], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(convert_order(order, [1, 3]), [[0, 0], [1, 0]])
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            convert_order([[0, -1], [-1, 0]], [1, 2])
        with self.assertRaisesRegex(ValueError, 'square'):
            convert_order([[0, 1], [0, 0]], [1, 2])

    def test_classes_keep_vmrd_ids_and_group_novel(self):
        _, mapping = load_classes(Path('scripts/metagraspnet_classes.json'))
        self.assertEqual(mapping[5], CLASS_IDS['banana'])
        self.assertEqual(mapping[67], CLASS_IDS['box'])
        self.assertEqual(mapping[19], CLASS_IDS['screwdriver'])
        self.assertEqual(mapping[20], CLASS_IDS['screwdriver'])
        self.assertEqual(mapping[69], 31)  # tennis ball
        self.assertEqual(mapping[79], 31)  # clamp

    def make_scene(self, root, name):
        scene = root / name
        scene.mkdir(parents=True)
        Image.new('RGB', (20, 12)).save(scene / '3_rgb.png')
        instances = np.zeros((12, 20))
        instances[2:7, 1:8] = 1
        instances[3:10, 11:19] = 3
        semantic = np.where(instances == 1, 5, np.where(instances == 3, 69, 0))
        np.savez(scene / '3.npz', instances_objects=instances, instances_semantic=semantic)
        np.savez(scene / '3_grasps.npz', parallel_jaw_2d=np.array([
            [[2, 4], [6, 4]], [[12, 6], [18, 6]], [[8, 0], [10, 0]],
        ]))
        np.savetxt(scene / '3_order.csv', [[0, 0, -1], [0, 0, 0], [0, 0, 0]],
                   delimiter=',', fmt='%d')
        return scene

    def test_sample_alignment_and_background_grasp_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(Path(directory), 'scene1')
            annotations, info = convert_sample(scene / '3_rgb.png', {5: 0, 69: 31}, 4, 'u')
            self.assertEqual(annotations['labels'].splitlines(),
                             ['0 1 2 8 7 20 12', '31 11 3 19 10 20 12'])
            self.assertEqual(annotations['adj'], '0 0\n1 0\n')
            self.assertEqual([row.split()[-3:] for row in annotations['grasps'].splitlines()],
                             [['u', '1', '0'], ['u', '3', '31']])
            self.assertEqual(info['invisible_instance_ids'], [2])
            self.assertEqual(info['dropped_grasps'][0]['grasp_index'], 2)

    def test_cli_scene_split_numeric_ids_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory) / 'raw'
            output = Path(directory) / 'converted'
            for name in ('scene1', 'scene2'):
                scene = self.make_scene(root, name)
                Image.new('RGB', (20, 12)).save(scene / '47_rgb.png')
            args = ['--root', str(root), '--output', str(output), '--val-fraction', '0.5']
            self.assertEqual(main(args + ['--dry-run']), 0)
            self.assertFalse(output.exists())
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args + ['--check']), 0)
            self.assertEqual(main(args), 0)
            report = json.loads((output / 'conversion_report.json').read_text())
            self.assertEqual(len(report['skipped']), 2)
            self.assertEqual(len(report['samples']), 2)
            self.assertEqual({s['split'] for s in report['samples'].values()}, {'train', 'val'})
            for stem, sample in report['samples'].items():
                int(stem)
                image = output / 'images' / sample['split'] / f'{stem}.png'
                self.assertTrue(image.is_symlink())
                self.assertTrue(image.exists())
            # Changed output must be rejected, not silently overwritten.
            label = next((output / 'labels').rglob('*.txt'))
            label.write_text('corrupted\n')
            with self.assertRaisesRegex(ValueError, 'Destination differs'):
                main(args)


if __name__ == '__main__':
    unittest.main()
