import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
from scripts.convert_vmrd import CLASSES
from scripts.label_project import grasp_rectangle
from scripts.merge_datasets import audit, main
from util.dataset_config import configure_classes, configure_tasks


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0

    def dataset(self, name, classes=('drill', 'cup', 'clamp'), grasps=True, vmrd_report=False,
                splits=('train', 'val')):
        self.counter += 1
        root = self.root / name
        root.mkdir()
        if vmrd_report:
            (root / 'conversion_report.json').write_text(json.dumps({'class_ids': {n:i for i,n in enumerate(classes)}}))
        else:
            (root / 'classes.json').write_text(json.dumps(classes))
        # Reusing the stem across splits also exercises globally unique output IDs.
        manifest = {'tasks': {'grasps': grasps}}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        for i, split in enumerate(splits):
            for folder in ('images', 'labels', 'adj', 'grasps'):
                (root / folder / split).mkdir(parents=True)
            Image.new('RGB', (100, 80), (self.counter*30, i*100, 80)).save(root / 'images' / split / '1.png')
            (root / 'labels' / split / '1.txt').write_text('0 5 5 40 60 100 80\n1 50 10 95 70 100 80\n')
            (root / 'adj' / split / '1.txt').write_text('0 1\n0 0\n')
            rectangles = [grasp_rectangle([10, 20], [30, 30], 4), grasp_rectangle([60, 30], [80, 30], 5)]
            rows = [' '.join(map(str, [*(v for point in points for v in point), 'e', owner, class_id]))
                    for points, owner, class_id in zip(rectangles, [4, 9], [0, 1])]
            (root / 'grasps' / split / '1.txt').write_text('\n'.join(rows)+'\n' if grasps else '')
        return root

    def run_merge(self, inputs, output=None, extra=()):
        output = output or self.root / 'combined'
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(['--inputs', *map(str, inputs), '--output', str(output), *extra])
        self.assertEqual(result, 0)
        return output

    def test_reordered_names_remap_boxes_and_grasps_preserving_rows(self):
        a = self.dataset('first')
        b = self.dataset('second', ('clamp', 'drill', 'cup'))
        output = self.run_merge([a, b])
        names = json.loads((output / 'classes.json').read_text())
        self.assertEqual(names[:31], list(CLASSES))
        self.assertEqual(names[31:], ['drill', 'clamp'])
        report = json.loads((output / 'manifest.json').read_text())
        self.assertTrue(report['tasks']['grasps'])
        self.assertEqual(report['counts']['train']['images'], 2)
        self.assertEqual(len(report['samples']), 4)
        for stem, entry in report['samples'].items():
            int(stem)
            split = entry['split']
            labels = (output / 'labels' / split / f'{stem}.txt').read_text().splitlines()
            grasps = (output / 'grasps' / split / f'{stem}.txt').read_text().splitlines()
            self.assertEqual([row.split()[0] for row in labels], ['31', '17'] if entry['source']=='first' else ['32', '31'])
            self.assertEqual([row.split()[-1] for row in grasps], [row.split()[0] for row in labels])
            self.assertEqual([row.split()[-2] for row in grasps], ['4', '9'])
            self.assertEqual((output / 'adj' / split / f'{stem}.txt').read_text(), '0 1\n0 0\n')
            source = a if entry['source']=='first' else b
            self.assertEqual((output / 'images' / split / f'{stem}.png').read_bytes(), (source / 'images' / split / '1.png').read_bytes())
            self.assertEqual(' '.join(labels[0].split()[1:]), '5 5 40 60 100 80')

    def test_vmrd_report_and_case_whitespace_matching(self):
        a = self.dataset('vmrd', CLASSES, vmrd_report=True)
        b = self.dataset('lab', (' BANANA ', 'watch'))
        samples, report = audit([b, a])
        self.assertEqual(report['classes'], list(CLASSES))
        self.assertEqual(report['sources'][0]['class_id_map'], {0:0, 1:1})
        self.assertEqual(len(samples), 4)

    def test_mixed_two_and_three_split_sources_preserve_test_annotations(self):
        legacy = self.dataset('legacy', CLASSES, vmrd_report=True)
        modern = self.dataset('modern', ('clamp', 'cup', 'drill'),
                              splits=('train', 'val', 'test'))
        output = self.run_merge([legacy, modern])
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual({s: c['images'] for s, c in manifest['counts'].items()},
                         {'train': 2, 'val': 2, 'test': 1})
        self.assertEqual(manifest['sources'][0]['counts']['test']['images'], 0)
        self.assertEqual(len(manifest['samples']), 5)
        for stem, entry in manifest['samples'].items():
            split = entry['split']
            source = legacy if entry['source'] == 'legacy' else modern
            self.assertEqual((output / 'images' / split / f'{stem}.png').read_bytes(),
                             (source / 'images' / split / entry['image']).read_bytes())
            boxes = (output / 'labels' / split / f'{stem}.txt').read_text().splitlines()
            grasps = (output / 'grasps' / split / f'{stem}.txt').read_text().splitlines()
            expected = ['0', '1'] if source == legacy else ['31', '17']
            self.assertEqual([row.split()[0] for row in boxes], expected)
            self.assertEqual([row.split()[-1] for row in grasps], expected)
            for folder in ('labels', 'grasps'):
                original = (source / folder / split / '1.txt').read_text().splitlines()
                copied = (output / folder / split / f'{stem}.txt').read_text().splitlines()
                # Only semantic IDs change: geometry and grasp owners stay paired.
                without_class = lambda row: row.split()[1:] if folder == 'labels' else row.split()[:-1]
                self.assertEqual(list(map(without_class, original)), list(map(without_class, copied)))
            self.assertEqual((output / 'adj' / split / f'{stem}.txt').read_bytes(),
                             (source / 'adj' / split / '1.txt').read_bytes())

    def test_test_duplicates_take_priority_over_val_in_both_policy_names(self):
        source = self.dataset('modern', splits=('train', 'val', 'test'))
        other = self.dataset('other')  # Retains a separate usable validation sample.
        shutil.copyfile(source / 'images' / 'test' / '1.png', source / 'images' / 'val' / '1.png')
        with self.assertRaisesRegex(ValueError, 'different splits'):
            audit([source, other])
        for policy in ('prefer-heldout', 'prefer-val'):
            with self.subTest(policy=policy):
                samples, report = audit([source, other], duplicate_policy=policy)
                self.assertEqual(report['counts']['test']['images'], 1)
                self.assertEqual(report['counts']['val']['images'], 1)
                self.assertEqual([(s.source, s.split) for s in samples if s.source == 'modern'],
                                 [('modern', 'train'), ('modern', 'test')])
                self.assertEqual(report['excluded_samples'][0]['split'], 'val')

    def test_mixed_grasp_supervision_and_training_configuration(self):
        a = self.dataset('lab', grasps=False)
        b = self.dataset('annotated')
        output = self.run_merge([a, b])
        report = json.loads((output / 'manifest.json').read_text())
        self.assertFalse(report['tasks']['grasps'])
        self.assertEqual(report['counts']['train']['grasps'], 2)
        args = SimpleNamespace(dataset_root=output, no_grasps=None, num_classes=None)
        self.assertTrue(configure_tasks(args))
        self.assertEqual(configure_classes(args), 33)
        with self.assertRaisesRegex(ValueError, 'without grasp supervision'):
            audit([a, b], grasp_policy='require')
        _, disabled = audit([b], grasp_policy='disable')
        self.assertFalse(disabled['tasks']['grasps'])

    def test_cross_split_duplicates_error_exclusion_and_explicit_keep(self):
        a = self.dataset('first')
        b = self.dataset('second')
        shutil.copyfile(a / 'images' / 'train' / '1.png', b / 'images' / 'val' / '1.png')
        with self.assertRaisesRegex(ValueError, 'Identical image'):
            audit([a, b])
        samples, report = audit([a, b], duplicate_policy='prefer-val')
        self.assertEqual(len(samples), 3)
        self.assertEqual(report['excluded_samples'][0]['source'], 'first')
        self.assertEqual(report['sources'][0]['counts']['train']['images'], 0)
        self.assertEqual(report['sources'][0]['input_counts']['train']['images'], 1)
        samples, report = audit([a, b], duplicate_policy='keep')
        self.assertEqual(len(samples), 4)
        self.assertEqual(len(report['cross_split_duplicates']), 1)

    def test_dry_run_and_output_protection(self):
        a = self.dataset('first')
        output = self.run_merge([a], extra=['--dry-run'])
        self.assertFalse(output.exists())
        self.run_merge([a])
        original = (output / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_merge([a])
        self.assertEqual((output / 'manifest.json').read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'contain one another'):
            self.run_merge([a], a/'nested')

    def test_invalid_sources_fail_before_output(self):
        a = self.dataset('first')
        (a / 'labels' / 'train' / '1.txt').write_text('500 5 5 40 60 100 80\n1 50 10 95 70 100 80\n')
        with self.assertRaisesRegex(ValueError, 'absent from the source registry'):
            self.run_merge([a])
        self.assertFalse((self.root/'combined').exists())
        (a / 'labels' / 'train' / '1.txt').write_text('0 5 5 40 60 100 80\n1 50 10 95 70 100 80\n')
        (a / 'adj' / 'val' / '1.txt').write_text('0 1\n1 0\n')
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            self.run_merge([a])
        self.assertFalse((self.root/'combined').exists())

    def test_missing_registry_can_be_supplied_explicitly_and_aliases(self):
        a = self.dataset('first', ('custom can', 'watch'))
        registry = a/'classes.json'
        override = self.root/'mapping.json'
        registry.rename(override)
        with self.assertRaisesRegex(ValueError, 'no class registry'):
            audit([a])
        aliases=self.root/'aliases.json'
        aliases.write_text(json.dumps({'custom can':'cans'}))
        output=self.run_merge([a], extra=['--class-map',f'{a}={override}','--aliases',str(aliases)])
        self.assertEqual(json.loads((output/'classes.json').read_text()), list(CLASSES))

    def test_duplicate_names_and_manifest_owner_mismatch_rejected(self):
        a=self.dataset('first')
        (a/'classes.json').write_text(json.dumps(['cup',' CUP ']))
        with self.assertRaisesRegex(ValueError, 'duplicate class names'):
            audit([a])
        (a/'classes.json').write_text(json.dumps(['drill','cup','clamp']))
        # The source manifest can preserve sparse instance IDs for ownership checks.
        (a/'manifest.json').write_text(json.dumps({'samples':{'1':{'split':'train','object_ids_in_row_order':[9,4]}}}))
        with self.assertRaisesRegex(ValueError, 'owner/class'):
            audit([a])

    def test_real_dataset_loader_on_merged_output(self):
        from datasets import build_dataset
        a = self.dataset('lab', grasps=False)
        b = self.dataset('annotated', ('clamp','drill','cup'))
        output=self.run_merge([a,b])
        args=SimpleNamespace(dataset_root=output, no_grasps=None)
        configure_tasks(args)
        for split in ('train','val'):
            dataset=build_dataset(split,args=args)
            self.assertEqual(len(dataset),2)
            for index in range(len(dataset)):
                image,target=dataset[index]
                self.assertEqual(image.shape[0],3)
                self.assertEqual(target['boxes'].shape,(2,4))
                self.assertEqual(target['adj'].tolist(),[[0,1],[0,0]])
                self.assertEqual(target['grasp_points'].shape,(0,4,2))


if __name__=='__main__':
    unittest.main()
