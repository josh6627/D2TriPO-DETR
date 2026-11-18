"""Sidebar wheel routing must not trigger image rendering or annotation edits."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, call

from scripts.label_dataset import Labeler


class SidebarScrollTests(unittest.TestCase):
    def setUp(self):
        self.app = Labeler.__new__(Labeler)
        self.app.root = Mock()
        self.app.root._w = '.'
        self.app.root.tk.call.return_value = ''
        self.app.side_canvas = Mock()
        self.app.side_canvas.yview.return_value = (.2, .8)
        self.app.draw, self.app.refresh, self.app.project = Mock(), Mock(), Mock()
        self.widget = Mock()
        self.widget.winfo_class.return_value = 'TLabel'

    def event(self, **kwargs):
        return SimpleNamespace(widget=self.widget, **kwargs)

    def test_registration_uses_a_class_tag_instead_of_a_nonexistent_widget_path(self):
        self.app.side_canvas = MagicMock()
        self.app.side_canvas.__str__.return_value = '.!frame4.!frame.!canvas'
        self.app.bind_sidebar_scroll = Mock()
        sidebar = Mock()
        self.app.setup_sidebar_scroll(sidebar)
        tag = self.app._sidebar_wheel_tag
        self.assertFalse(tag.startswith('.'), 'Tk treats dot-prefixed tags as window paths')
        self.assertIn(str(self.app.side_canvas), tag)
        self.assertEqual(self.app.side_canvas.bind_class.call_args_list, [
            call(tag, sequence, self.app.scroll_sidebar)
            for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>')
        ])
        self.app.bind_sidebar_scroll.assert_called_once_with(sidebar)

    def test_linux_and_mousewheel_events_scroll_in_both_directions(self):
        for event, units in ((dict(num=4), -3), (dict(num=5), 3),
                             (dict(delta=120), -3), (dict(delta=-120), 3),
                             (dict(delta=240), -6), (dict(delta=1), -3),
                             (dict(delta=-1), 3)):
            with self.subTest(event=event):
                self.app.side_canvas.yview_scroll.reset_mock()
                self.assertEqual(self.app.scroll_sidebar(self.event(**event)), 'break')
                self.app.side_canvas.yview_scroll.assert_called_once_with(units, 'units')

    def test_repeated_wheel_events_do_not_redraw_save_or_schedule_callbacks(self):
        for _ in range(500):
            self.app.scroll_sidebar(self.event(num=5))
        self.assertEqual(self.app.side_canvas.yview_scroll.call_count, 500)
        self.app.draw.assert_not_called()
        self.app.refresh.assert_not_called()
        self.app.project.save.assert_not_called()
        self.app.root.after.assert_not_called()
        self.app.side_canvas.create_image.assert_not_called()

    def test_panel_boundaries_and_zero_delta_do_not_scroll(self):
        for view, event in (((0., .5), dict(num=4)), ((.5, 1.), dict(num=5)),
                            ((0., 1.), dict(delta=120)), ((.2, .8), dict(delta=0))):
            with self.subTest(view=view, event=event):
                self.app.side_canvas.yview.return_value = view
                self.app.scroll_sidebar(self.event(**event))
        self.app.side_canvas.yview_scroll.assert_not_called()

    def test_lists_keep_native_scroll_and_hand_off_at_boundary(self):
        self.widget.winfo_class.return_value = 'Listbox'
        self.widget.yview.return_value = (.2, .8)
        for number in (4, 5):
            self.assertIsNone(self.app.scroll_sidebar(self.event(num=number)))
        self.app.side_canvas.yview_scroll.assert_not_called()
        for view, number, units in (((0., .5), 4, -3), ((.5, 1.), 5, 3), ((0., 1.), 5, 3)):
            with self.subTest(view=view):
                self.widget.yview.return_value = view
                self.app.side_canvas.yview_scroll.reset_mock()
                self.assertEqual(self.app.scroll_sidebar(self.event(num=number)), 'break')
                self.app.side_canvas.yview_scroll.assert_called_once_with(units, 'units')

    def test_modal_grabs_are_left_to_tk_without_resolving_internal_widgets(self):
        self.app.root.tk.call.return_value = '.!combobox.popdown'
        self.assertIsNone(self.app.scroll_sidebar(self.event(num=5)))
        self.app.side_canvas.yview_scroll.assert_not_called()
        self.app.root.focus_get.assert_not_called()
        self.app.root.grab_current.assert_not_called()

    def test_closed_combobox_scroll_is_consumed_before_native_value_change(self):
        self.widget.winfo_class.return_value = 'TCombobox'
        self.assertEqual(self.app.scroll_sidebar(self.event(num=5)), 'break')
        self.widget.set.assert_not_called()

    def test_bindtags_cover_sidebar_descendants_once_and_preserve_other_tags(self):
        self.app._sidebar_wheel_tag = 'sidebar.wheel'
        def widget(path, kind, children):
            result = Mock()
            tags = [(path, kind, '.', 'all')]
            def bindtags(*args):
                if args:
                    tags[0] = args[0]
                return tags[0]
            result.bindtags.side_effect = bindtags
            result.winfo_children.return_value = children
            return result
        label = widget('label', 'TLabel', [])
        button = widget('button', 'TButton', [])
        panel = widget('panel', 'Canvas', [label, button])
        outside = widget('image', 'Canvas', [])
        self.app.bind_sidebar_scroll(panel)
        self.app.bind_sidebar_scroll(panel)
        for target, name, kind in ((panel, 'panel', 'Canvas'), (label, 'label', 'TLabel'),
                                    (button, 'button', 'TButton')):
            self.assertEqual(target.bindtags(), (name, 'sidebar.wheel', kind, '.', 'all'))
            target.bind_all.assert_not_called()
        outside.bindtags.assert_not_called()


if __name__ == '__main__':
    unittest.main()
