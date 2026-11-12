"""GUI regressions. Run with a graphical or virtual display; otherwise skipped."""
import copy
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageTk
from scripts.label_dataset import Labeler, bounded_zoom, MAX_RENDER_PIXELS, MAX_RENDER_EDGE
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

    def test_manual_test_selection_progress_and_undo_are_independent(self):
        self.app.test_checkbox.invoke()
        self.assertEqual(self.project.data['test_images'], ['1.png'])
        self.assertIn('TEST TARGET REACHED', self.app.test_status.get())
        self.assertTrue(self.app.sample['complete'])
        self.assertIn('[TEST]', self.app.image_choice.get())
        self.assertEqual(Project(self.project.images, self.project.path).data['test_images'], ['1.png'])
        self.app.reclassify()
        self.assertNotIn('TEST TARGET REACHED', self.app.test_status.get())
        self.app.undo()
        self.assertTrue(self.app.use_for_test.get())
        self.assertIn('TEST TARGET REACHED', self.app.test_status.get())
        self.app.test_checkbox.invoke()
        self.assertIn('Choose 1 more', self.app.test_status.get())
        self.assertEqual(self.project.data['test_images'], [])
        self.root.update()
        self.assertEqual(self.errors, [])

    def test_resize_preview_is_reused_throttled_and_commits_release_position(self):
        self.app.mode.set('select')
        with patch('scripts.label_dataset.time.monotonic', return_value=1.0), \
             patch.object(self.app.canvas, 'create_rectangle', wraps=self.app.canvas.create_rectangle) as create, \
             patch.object(self.root, 'after', wraps=self.root.after) as schedule:
            self.app.press(SimpleNamespace(x=120, y=70))
            self.app.drag(SimpleNamespace(x=121, y=70))
            item = self.app._preview_item
            for index in range(500):
                self.app.drag(SimpleNamespace(x=122+index%50, y=70))
            self.assertEqual(create.call_count, 1)
            self.assertEqual(schedule.call_count, 1)
            self.assertEqual(self.app.canvas.find_withtag('preview'), (item,))
            # Let the single queued frame consume the latest pointer coordinates.
            self.root.after(50, self.root.quit)
            self.root.mainloop()
            self.assertIsNone(self.app._preview_after)
            self.assertEqual(create.call_count, 1)
            self.assertAlmostEqual(self.app.canvas.coords(item)[2], 171)
        # Saving uses the release event, even if no preview displayed that position.
        self.app.release(SimpleNamespace(x=223, y=70))
        self.assertAlmostEqual(self.app.require_object()['box'][2], 223)
        self.assertFalse(self.app.canvas.find_withtag('preview'))
        self.assertIsNone(self.app._preview_after)
        self.root.update()
        self.assertEqual(self.errors, [])

    def test_escape_and_zoom_cancel_queued_preview_frames(self):
        for operation in (self.app.cancel, lambda: self.app.zoom(1)):
            with patch('scripts.label_dataset.time.monotonic', return_value=1.0):
                self.app.press(SimpleNamespace(x=120, y=70))
                self.app.drag(SimpleNamespace(x=125, y=70))
                self.app.drag(SimpleNamespace(x=130, y=70))
                scheduled = self.app._preview_after
                self.assertIsNotNone(scheduled)
                operation()
            self.assertIsNone(self.app._preview_after)
            self.assertFalse(self.app.canvas.find_withtag('preview'))
            self.assertNotIn(scheduled, self.root.tk.call('after', 'info'))
            self.assertEqual(self.app.require_object()['box'], [20, 20, 120, 120])
        self.root.update()
        self.assertEqual(self.errors, [])

    def test_unchanged_progress_and_cursor_do_not_reconfigure_widgets(self):
        writes = []
        self.app.test_status.trace_add('write', lambda *args: writes.append(self.app.test_status.get()))
        with patch.object(self.app.test_status_label, 'configure', wraps=self.app.test_status_label.configure) as configure:
            for index in range(50):
                self.app.selected = 1 if index % 2 else 4
                self.app.refresh()
            self.assertEqual(writes, [])
            self.assertEqual(configure.call_count, 0)
            self.app.test_checkbox.invoke()
            self.assertEqual(len(writes), 1)
            self.assertEqual(configure.call_count, 1)
        self.app.selected = 1
        with patch.object(self.app.canvas, 'configure', wraps=self.app.canvas.configure) as configure:
            for _ in range(100):
                self.app.motion(SimpleNamespace(x=120, y=70))
            self.assertEqual(configure.call_count, 1)
            self.app.motion(SimpleNamespace(x=60, y=60))
            self.assertEqual(configure.call_count, 2)

    def test_repeated_selection_and_edits_reuse_background(self):
        # Match the higher-resolution lab images involved in the reported freezes.
        self.app.image = Image.new('RGB', (1920, 1080))
        self.app.draw()
        photo = self.app.photo
        item = self.app._image_item
        image_count = len(self.root.tk.call('image', 'names'))
        with patch('scripts.label_dataset.ImageTk.PhotoImage', wraps=ImageTk.PhotoImage) as upload:
            for index in range(150):
                self.app.selected = 1 if index % 2 else 4
                self.app.refresh()
                self.app.require_object()['box'][0] += .001
                self.app.draw()
            self.assertEqual(upload.call_count, 0)
            self.assertIs(self.app.photo, photo)
            self.assertEqual(self.app._image_item, item)
            del photo  # Do not artificially retain the old Tk image across zooms.
            self.app.zoom(1.25)
            self.assertEqual(upload.call_count, 1)
            self.app.load(0)
            self.assertEqual(upload.call_count, 2)
        self.root.update()
        self.assertEqual(len(self.root.tk.call('image', 'names')), image_count)
        self.assertEqual(self.errors, [])


class RenderBudgetTests(unittest.TestCase):
    def test_zoom_stays_within_display_pixel_and_edge_limits(self):
        for size in ((512, 512), (1920, 1080), (8000, 6000), (100000, 100)):
            with self.subTest(size=size):
                scale = bounded_zoom(size, 5)
                width, height = (int(v*scale) for v in size)
                self.assertLessEqual(width*height, MAX_RENDER_PIXELS)
                self.assertLessEqual(max(width, height), MAX_RENDER_EDGE)
                self.assertGreater(scale, 0)
        self.assertEqual(bounded_zoom((512, 512), 5), 5)
        self.assertLess(bounded_zoom((1920, 1080), 5), 2)


if __name__ == '__main__':
    unittest.main()
