"""Renderer fallback must preserve CLI options and keep working if unavailable."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.label_dataset import prefer_smooth_font_renderer


class FontRendererTests(unittest.TestCase):
    def setUp(self):
        self.root = Mock()
        self.platform = patch('scripts.label_dataset.sys.platform', 'linux')
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def test_usable_font_keeps_current_renderer(self):
        with patch('scripts.label_dataset.tkfont.families', return_value=('DejaVu Sans',)), \
             patch('scripts.label_dataset.subprocess.run') as probe, \
             patch('scripts.label_dataset.os.execv') as restart:
            prefer_smooth_font_renderer(self.root)
        probe.assert_not_called()
        restart.assert_not_called()
        self.root.destroy.assert_not_called()

    def test_failed_probe_keeps_window_open(self):
        with patch('scripts.label_dataset.tkfont.families', return_value=('fixed',)), \
             patch('scripts.label_dataset.sys.executable', '/tmp/bitmap-python'), \
             patch('scripts.label_dataset.Path.is_file', return_value=True), \
             patch('scripts.label_dataset.subprocess.run', return_value=SimpleNamespace(returncode=1)), \
             patch('scripts.label_dataset.os.execv') as restart:
            prefer_smooth_font_renderer(self.root)
        restart.assert_not_called()
        self.root.destroy.assert_not_called()

    def test_fallback_preserves_labeler_options(self):
        options = ['label_dataset.py', '--project', '/tmp/labels/project.json', '--no-grasps', '--font-size', '16']
        with patch('scripts.label_dataset.tkfont.families', return_value=('fixed',)), \
             patch('scripts.label_dataset.sys.executable', '/tmp/bitmap-python'), \
             patch('scripts.label_dataset.sys.argv', options), \
             patch('scripts.label_dataset.Path.is_file', return_value=True), \
             patch('scripts.label_dataset.subprocess.run', return_value=SimpleNamespace(returncode=0)), \
             patch('scripts.label_dataset.os.execv') as restart:
            prefer_smooth_font_renderer(self.root)
        self.root.destroy.assert_called_once()
        executable, command = restart.call_args.args
        self.assertEqual(executable, '/usr/bin/python3')
        self.assertEqual(command[0], executable)
        self.assertTrue(command[1].endswith('/scripts/label_dataset.py'))
        self.assertEqual(command[2:], options[1:])


if __name__ == '__main__':
    unittest.main()
