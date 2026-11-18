"""Directed relationship editing and export, without a graphical display."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from scripts.label_dataset import Labeler
from scripts.label_project import Project, validate, edge_description


class LabelRelationshipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        for name in ('1.png', '2.png', '3.png'):
            Image.new('RGB', (100, 80)).save(self.path / name)
        self.project = Project(self.path, self.path/'project.json', require_grasps=False)
        for name in self.project.files:
            self.project.sample(name).update(
                objects=[dict(id=2, class_id=0, box=[5, 5, 25, 25]),
                         dict(id=7, class_id=1, box=[35, 5, 55, 25]),
                         dict(id=11, class_id=2, box=[65, 5, 85, 25])],
                edges=[[7, 11]], relations_reviewed=True, complete=True,
            )
        self.project.set_test_image('3.png', True)
        # Exercise real controller callbacks, validation, saving and undo while
        # replacing only the widgets that require a window server.
        self.app = Labeler.__new__(Labeler)
        self.app.project, self.app.index, self.app.selected = self.project, 0, 2
        self.app.image = Image.new('RGB', (100, 80))
        self.app.history, self.app.pending, self.app.resizing = [], [], None
        self.app.below, self.app.relation_type, self.app.edges = Mock(), Mock(), Mock()
        self.app.below.current.return_value = 1
        self.app.relation_type.current.return_value = 1
        self.app.refresh = Mock()

    def validate(self, sample):
        validate(sample, self.project.classes, (100, 80), require_grasps=False)

    def test_add_proximity_persists_direction_and_requires_review(self):
        self.app.add_edge()
        self.assertEqual(self.app.sample['edges'], [[7, 11], [2, 7, 2]])
        self.assertFalse(self.app.sample['complete'])
        self.assertFalse(self.app.sample['relations_reviewed'])
        saved = Project(self.path, self.project.path).sample('1.png')
        self.assertEqual(saved, self.app.sample)
        self.assertEqual(self.project.data['test_images'], ['3.png'])

    def test_two_to_three_means_two_obstructs_three_in_display_and_export(self):
        for name in self.project.files:
            sample = self.project.sample(name)
            sample['objects'][1]['id'] = 3
            sample['edges'] = []
        self.app.add_edge()  # Selected ID 2, target dropdown selects ID 3.
        self.assertEqual(self.app.sample['edges'], [[2, 3, 2]])
        self.assertEqual(edge_description(self.app.sample['edges'][0]),
                         '#2 → #3: #2 obstructs grasping #3')
        self.assertEqual(edge_description([2, 3]), '#2 → #3: #2 is above / occludes #3')
        self.assertEqual(edge_description([3, 2, 2]), '#3 → #2: #3 obstructs grasping #2')
        self.app.sample.update(complete=True, relations_reviewed=True)
        output = self.project.export(self.path/'direction_export')
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['adjacency']['meanings']['2'],
                         'Source obstructs grasping/manipulating target')
        stem, entry = next((stem, entry) for stem, entry in manifest['samples'].items()
                           if entry['image'] == '1.png')
        self.assertEqual(entry['object_ids_in_row_order'], [2, 3, 11])
        self.assertEqual((output/'adj'/entry['split']/f'{stem}.txt').read_text(),
                         '0 2 0\n0 0 0\n0 0 0\n')

    def test_duplicate_is_noop_and_undo_restores_original(self):
        before = copy.deepcopy(self.app.sample)
        self.app.add_edge()
        self.app.add_edge()
        self.assertEqual(len(self.app.history), 1)
        self.app.undo()
        self.assertEqual(self.app.sample, before)

    def test_stacking_replaces_proximity_and_takes_priority(self):
        self.app.add_edge()
        self.app.relation_type.current.return_value = 0
        self.app.add_edge()
        self.assertEqual(self.app.sample['edges'], [[7, 11], [2, 7, 1]])
        self.app.relation_type.current.return_value = 1
        with self.assertRaisesRegex(ValueError, 'priority'):
            self.app.add_edge()
        self.assertEqual(self.app.sample['edges'], [[7, 11], [2, 7, 1]])

    def test_self_edge_and_stacking_cycle_roll_back(self):
        before = copy.deepcopy(self.app.sample)
        self.app.below.current.return_value = 0
        with self.assertRaisesRegex(ValueError, 'different'):
            self.app.add_edge()
        self.assertEqual(self.app.sample, before)
        self.app.selected = 11
        self.app.below.current.return_value = 1
        self.app.relation_type.current.return_value = 0
        with self.assertRaisesRegex(ValueError, 'cycle'):
            self.app.add_edge()
        self.assertEqual(self.app.sample, before)
        self.assertEqual(self.app.history, [])

    def test_mutual_proximity_and_mixed_cycles_are_allowed(self):
        self.app.add_edge()
        self.app.selected = 7
        self.app.below.current.return_value = 0
        self.app.add_edge()
        self.assertIn([7, 2, 2], self.app.sample['edges'])
        self.app.selected = 11
        self.app.below.current.return_value = 1
        self.app.add_edge()  # Reverse of existing stacking edge, class 2.
        self.validate(self.app.sample)

    def test_delete_edge_requires_review_and_can_be_undone(self):
        self.app.sample['edges'] = [[2, 7, 2]]
        before = copy.deepcopy(self.app.sample)
        self.app.edges.curselection.return_value = (0,)
        self.app.delete_edge()
        self.assertEqual(self.app.sample['edges'], [])
        self.assertFalse(self.app.sample['relations_reviewed'])
        self.app.undo()
        self.assertEqual(self.app.sample, before)

    def test_deleting_object_two_does_not_delete_unrelated_class_two_edge(self):
        self.app.sample['edges'] = [[2, 7, 2], [7, 11, 2]]
        before = copy.deepcopy(self.app.sample)
        self.app.delete_object()
        self.assertEqual(self.app.sample['edges'], [[7, 11, 2]])
        self.app.undo()
        self.assertEqual(self.app.sample, before)

    def test_invalid_edges_are_rejected(self):
        for edge, message in [([2, 7, 0], 'class'), ([2, 7, 3], 'class'),
                              ([2, 7, True], 'class'), ([2, 7, '2'], 'class'),
                              ([2, 7, 2, 1], 'optional'), ([2, 99, 2], 'existing'),
                              ([2, 2, 2], 'different')]:
            with self.subTest(edge=edge):
                bad = copy.deepcopy(self.app.sample)
                bad['edges'] = [edge]
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(bad)
        bad = copy.deepcopy(self.app.sample)
        bad['edges'] = [[2, 7], [2, 7, 2]]
        with self.assertRaisesRegex(ValueError, 'one relationship'):
            self.validate(bad)

    def test_legacy_project_is_preserved_when_reopened(self):
        self.project.save()
        before = self.project.path.read_text()
        reopened = Project(self.path, self.project.path)
        self.validate(reopened.sample('1.png'))
        self.assertEqual(reopened.sample('1.png')['edges'], [[7, 11]])
        self.assertEqual(self.project.path.read_text(), before)

    def test_export_and_loader_preserve_class_two_and_object_row_order(self):
        from datasets.vmrd import CustomDataset, make_vmrd_transforms

        for name in self.project.files:
            self.project.sample(name)['edges'] = [[7, 11], [2, 7, 2], [7, 2, 2]]
        output = self.project.export(self.path/'export')
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertIn('2', manifest['adjacency']['classes'])
        for stem, entry in manifest['samples'].items():
            self.assertEqual(entry['object_ids_in_row_order'], [2, 7, 11])
            self.assertEqual((output/'adj'/entry['split']/f'{stem}.txt').read_text(),
                             '0 2 0\n2 0 1\n0 0 0\n')
        for split in ('train', 'val', 'test'):
            dataset = CustomDataset(
                *(str(output/folder/split) for folder in ('images', 'labels', 'adj', 'grasps')),
                require_grasps=False, transform=make_vmrd_transforms(split),
            )
            _, target = dataset[0]
            self.assertEqual(target['adj'].tolist(), [[0, 2, 0], [2, 0, 1], [0, 0, 0]])

    def test_canvas_draws_typed_directional_arrows(self):
        self.app.sample['edges'] = [[7, 11], [2, 7, 2]]
        self.app.canvas, self.app.mode = Mock(), Mock()
        self.app.mode.get.return_value = 'box'
        self.app.clear_preview = Mock()
        self.app.scale = 1
        self.app._photo_key = (id(self.app.image), self.app.image.size)
        self.app.font_family, self.app.font_size = 'sans', 16
        self.app.draw()
        arrows = self.app.canvas.create_line.call_args_list
        self.assertEqual(len(arrows), 2)
        self.assertEqual(arrows[0].kwargs['fill'], '#e0aaff')
        self.assertEqual(arrows[0].kwargs['dash'], ())
        self.assertEqual(arrows[1].kwargs['fill'], '#ffad42')
        self.assertEqual(arrows[1].kwargs['dash'], (7, 4))
        self.assertEqual(arrows[1].kwargs['arrow'], 'last')
        self.assertEqual(arrows[1].args, (15, 15, 45, 15))


if __name__ == '__main__':
    unittest.main()
