"""Classic UI startup keeps the chosen Python/Tk and honors text size."""
import unittest
from unittest.mock import Mock, patch

from scripts.label_dataset import main


class ClassicStartupTests(unittest.TestCase):
    def test_native_renderer_and_larger_default_text(self):
        self.check_startup([], 16)

    def test_custom_text_size(self):
        self.check_startup(['--font-size', '20'], 20)

    def check_startup(self, options, expected_size):
        project, root = Mock(), Mock()
        with patch('scripts.label_dataset.sys.argv', ['label_dataset.py', *options]), \
             patch('scripts.label_dataset.Project', return_value=project), \
             patch('scripts.label_dataset.tk.Tk', return_value=root), \
             patch('scripts.label_dataset.configure_diagnostics'), \
             patch('scripts.label_dataset.Labeler') as labeler, \
             patch('scripts.label_dataset.os.execv') as restart:
            main()
        labeler.assert_called_once_with(root, project, font_size=expected_size)
        root.mainloop.assert_called_once()
        restart.assert_not_called()


if __name__ == '__main__':
    unittest.main()
