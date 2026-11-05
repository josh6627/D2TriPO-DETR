"""GUI regressions. Run with a graphical or virtual display; otherwise skipped."""
import copy
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest

from PIL import Image
from scripts.label_dataset import Labeler
from scripts.label_project import Project, grasp_rectangle


class LabelerDeleteTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'GUI display unavailable: {exc}')
        self.addCleanup(self.root.destroy)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name)
        Image.new('RGB', (512, 512)).save(path / '1.png')
        self.project = Project(path, path / 'project.json', require_grasps=False)
        sample = self.project.sample('1.png')
        sample.update(objects=[{'id': 1, 'class_id': 0, 'box': [20, 20, 120, 120]},
                               {'id': 4, 'class_id': 1, 'box': [200, 200, 300, 300]}],
                      grasps=[{'object_id': 1, 'difficulty': 'e',
                               'points': grasp_rectangle([40, 60], [90, 60], 5)}],
                      edges=[[1, 4]], relations_reviewed=True, complete=True)
        self.project.save()
        self.app = Labeler(self.root, self.project)
        self.errors = []
        self.root.report_callback_exception = lambda kind, value, tb: self.errors.append(value)
        self.root.update()
        self.app.selected = 1
        self.app.refresh()

    def test_delete_clears_handles_and_associations_before_redraw(self):
        self.app.delete_object()
        self.root.update()
        self.assertIsNone(self.app.selected)
        self.assertIsNone(self.app.resizing)
        self.assertEqual([o['id'] for o in self.app.sample['objects']], [4])
        self.assertEqual(self.app.sample['grasps'], [])
        self.assertEqual(self.app.sample['edges'], [])
        self.assertFalse(self.app.sample['relations_reviewed'])
        saved = Project(self.project.images, self.project.path).sample('1.png')
        self.assertEqual(saved, self.app.sample)
        self.app.motion(SimpleNamespace(x=30, y=30))
        self.app.press(SimpleNamespace(x=240, y=240))
        self.assertEqual(self.app.selected, 4)
        self.assertEqual(self.errors, [])

    def test_delete_last_object_then_draw_new_box(self):
        self.app.delete_object()
        self.app.selected = 4
        self.app.refresh()
        self.app.delete_object()
        self.assertEqual(self.app.sample['objects'], [])
        self.app.motion(SimpleNamespace(x=30, y=30))
        self.app.mode.set('box')
        self.app.press(SimpleNamespace(x=40, y=40))
        self.app.release(SimpleNamespace(x=160, y=160))
        self.root.update()
        self.assertEqual(len(self.app.sample['objects']), 1)
        self.assertEqual(self.errors, [])

    def test_undo_restores_deleted_box_grasps_and_relationships(self):
        original = copy.deepcopy(self.app.sample)
        self.app.delete_object()
        self.app.undo()
        self.root.update()
        self.assertEqual(self.app.sample, original)
        self.assertEqual(self.errors, [])


if __name__ == '__main__':
    unittest.main()
