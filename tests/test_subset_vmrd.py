import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.subset_vmrd import main, percentage, plan_subset


class SubsetVMRDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'source'
        for split, count in (('train', 95), ('val', 10)):
            for folder in ('images', 'labels', 'adj', 'grasps'):
                (self.root / folder / split).mkdir(parents=True)
            for index in range(count):
                stem = str(index + (1000 if split == 'val' else 0))
                (self.root / 'images' / split / f'{stem}.jpg').write_bytes(stem.encode())
                for folder in ('labels', 'adj', 'grasps'):
                    (self.root / folder / split / f'{stem}.txt').write_text(f'{folder} {stem}\n')
        (self.root / 'conversion_report.json').write_text(json.dumps({'class_ids': {'cup': 1, 'banana': 0}}))

    def run_subset(self, output='subset', extra=()):
        destination = self.base / output
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(['--input', str(self.root), '--output', str(destination),
                           '--percent', '10', *extra])
        self.assertEqual(result, 0)
        return destination

    def test_repeatable_selection_and_rounding(self):
        first, counts = plan_subset(self.root, percentage(10), 42)
        second, _ = plan_subset(self.root, percentage(10), 42)
        third, _ = plan_subset(self.root, percentage(10), 123)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(counts['train'], {'available': 95, 'selected': 10})
        self.assertEqual(counts['val'], {'available': 10, 'selected': 1})
        full, _ = plan_subset(self.root, percentage(100), 42)
        self.assertEqual(len(full['train']), 95)
        self.assertEqual(len(full['val']), 10)

    def test_copies_complete_bundles_and_metadata_without_changing_source(self):
        (self.root / 'manifest.json').write_text(json.dumps({'tasks': {'grasps': False}}))
        # Replace a source image with a symlink, like the converted real VMRD.
        image = self.root / 'images' / 'val' / '1000.jpg'
        original = self.base / 'original.jpg'
        image.rename(original)
        image.symlink_to(original)
        output = self.run_subset(extra=['--percent', '100'])
        report = json.loads((output / 'subset_report.json').read_text())
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertFalse(manifest['tasks']['grasps'])
        self.assertEqual(json.loads((output / 'classes.json').read_text()), ['banana', 'cup'])
        for split, ids in report['selected_ids'].items():
            for stem in ids:
                for folder in ('images', 'labels', 'adj', 'grasps'):
                    filename = stem + ('.jpg' if folder == 'images' else '.txt')
                    copied = output / folder / split / filename
                    self.assertEqual(copied.read_bytes(), (self.root / folder / split / filename).read_bytes())
                    self.assertFalse(copied.is_symlink())
                self.assertEqual(manifest['samples'][stem]['split'], split)
        self.assertTrue(image.is_symlink())

    def test_preview_and_existing_destination_protection(self):
        destination = self.run_subset(extra=['--dry-run'])
        self.assertFalse(destination.exists())
        self.run_subset()
        before = (destination / 'subset_report.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_subset()
        self.assertEqual(before, (destination / 'subset_report.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'contain one another'):
            self.run_subset(output='source/nested')

    def test_missing_bundle_rejected_before_writing(self):
        chosen, _ = plan_subset(self.root, percentage(10), 42)
        (self.root / 'adj' / 'train' / (chosen['train'][0].stem + '.txt')).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing selected sample file'):
            self.run_subset()
        self.assertFalse((self.base / 'subset').exists())

    def test_invalid_percentage_and_empty_split(self):
        import argparse
        for value in ('0', '-1', '101', 'nan', 'inf', 'oops'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                percentage(value)
        with self.assertRaisesRegex(ValueError, 'rounds to zero val'):
            plan_subset(self.root, percentage(1), 42)

    def test_split_overlap_rejected(self):
        (self.root / 'images' / 'val' / '1000.jpg').rename(self.root / 'images' / 'val' / '0.jpg')
        # All annotation bundles are complete so the overlap itself is checked.
        for folder in ('labels', 'adj', 'grasps'):
            (self.root / folder / 'val' / '1000.txt').rename(self.root / folder / 'val' / '0.txt')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            plan_subset(self.root, percentage(100), 42)


if __name__ == '__main__':
    unittest.main()
