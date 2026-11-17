import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.dataset_stats import collect_stats, load_classes, main


class DatasetStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ['cup', 'bottle', 'unused']
        (self.root / 'classes.json').write_text(json.dumps(self.names))
        for split in ('train', 'test'):
            for kind in ('images', 'labels', 'adj'):
                (self.root / kind / split).mkdir(parents=True)
        self.sample('train', 'a', [0, 0, 1], '0 1 0\n0 0 1\n0 0 0\n')
        self.sample('test', 'b', [1], '0\n')

    def sample(self, split, stem, ids, adjacency):
        # Statistics need only filenames; decoding images should never be required.
        (self.root / 'images' / split / f'{stem}.JPG').touch()
        (self.root / 'labels' / split / f'{stem}.txt').write_text(
            ''.join(f'{i} 0 0 10 10 20 20\n' for i in ids))
        (self.root / 'adj' / split / f'{stem}.txt').write_text(adjacency)

    def test_counts_directed_edges_repeated_classes_and_orphans(self):
        (self.root / 'labels/train/orphan.txt').write_text('bad data')
        results, warnings = collect_stats(self.root, ['train', 'test'], self.names)
        self.assertEqual(results['combined']['counts'], {0: 2, 1: 2})
        self.assertEqual(results['combined']['relationships'], 2)
        self.assertEqual(results['combined']['images'], 2)
        self.assertEqual(results['test']['relationships'], 0)
        self.assertEqual(len(warnings), 1)

    def test_cli_csv_includes_absent_classes_and_preserves_source(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        output = self.root / 'counts.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--dataset-root', str(self.root), '--csv', str(output)]), 0)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['combined'], '2')
        self.assertEqual(rows[2], {'class_id': '2', 'class_name': 'unused',
                                   'train': '0', 'test': '0', 'combined': '0'})
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))

    def test_rejects_missing_or_invalid_annotations_without_totals(self):
        path = self.root / 'adj/train/a.txt'
        for content in ('0\n', '0 2 0\n0 0 0\n0 0 0\n',
                        '1 0 0\n0 0 0\n0 0 0\n'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, 'No totals produced'):
                    collect_stats(self.root, ['train'], self.names)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'No totals produced'):
            collect_stats(self.root, ['train'], self.names)

    def test_rejects_bad_labels_and_duplicate_image_stems(self):
        path = self.root / 'labels/train/a.txt'
        for content in ('9 0 0 10 10 20 20', '0 nan 0 10 10 20 20', '0 0 0'):
            path.write_text(content)
            with self.assertRaisesRegex(ValueError, 'No totals produced'):
                collect_stats(self.root, ['train'], self.names)
        (self.root / 'images/train/a.png').touch()
        with self.assertRaisesRegex(ValueError, 'duplicate image stems'):
            collect_stats(self.root, ['train'], self.names)

    def test_registry_report_fallback_and_numeric_ids(self):
        self.assertEqual(load_classes(self.root), self.names)
        (self.root / 'classes.json').unlink()
        self.assertIsNone(load_classes(self.root))
        results, _ = collect_stats(self.root, ['train'], None)
        self.assertEqual(results['combined']['counts'], {0: 2, 1: 1})
        (self.root / 'conversion_report.json').write_text(
            json.dumps({'class_ids': {'bottle': 1, 'cup': 0}}))
        self.assertEqual(load_classes(self.root), ['cup', 'bottle'])

    def test_selected_missing_split_and_empty_scene(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Missing image directory'):
                main(['--dataset-root', str(self.root), '--split', 'val'])
        self.sample('test', 'empty', [], '')
        results, _ = collect_stats(self.root, ['test'], self.names)
        self.assertEqual(results['combined']['images'], 2)
        self.assertEqual(sum(results['combined']['counts'].values()), 1)


if __name__ == '__main__':
    unittest.main()
