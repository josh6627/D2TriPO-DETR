"""Manual holdout membership and deterministic three-way export."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from scripts.label_project import Project
from scripts.merge_datasets import audit, write_dataset


class SplitExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root/'images'
        self.images.mkdir()
        for i in range(1, 11):
            Image.new('RGB', (50, 40), (i*20, 30, 50)).save(self.images/f'{i}.png')
        self.project = Project(self.images, self.root/'project.json', require_grasps=False)
        for name in self.project.files:
            self.project.sample(name).update(objects=[{'id':4,'class_id':0,'box':[5,5,40,35]}],
                                              relations_reviewed=True,complete=True)

    def mark_test(self):
        self.project.set_test_image('2.png', True)
        self.project.set_test_image('9.png', True)

    def test_exact_70_10_20_and_manual_membership(self):
        self.mark_test()
        output=self.project.export(self.root/'export')
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['split_counts'],{'train':7,'val':1,'test':2})
        tests={s['image'] for s in manifest['samples'].values() if s['split']=='test'}
        self.assertEqual(tests,{'2.png','9.png'})
        for folder in ('images','labels','adj','grasps'):
            for split, count in manifest['split_counts'].items():
                self.assertEqual(len(list((output/folder/split).iterdir())),count)
        # Identical seed produces identical membership; it never changes the chosen holdout.
        other=self.project.export(self.root/'again')
        self.assertEqual(manifest,json.loads((other/'manifest.json').read_text()))
        another=self.project.export(self.root/'seed2',seed=999)
        samples=json.loads((another/'manifest.json').read_text())['samples']
        self.assertEqual({s['image'] for s in samples.values() if s['split']=='test'},tests)
        self.assertNotEqual({s['image'] for s in samples.values() if s['split']=='val'},
                            {s['image'] for s in manifest['samples'].values() if s['split']=='val'})

    def test_missing_or_excess_test_selections_block_export(self):
        with self.assertRaisesRegex(ValueError,'Choose exactly 2'):
            self.project.export(self.root/'missing')
        self.assertFalse((self.root/'missing').exists())
        self.mark_test()
        self.project.set_test_image('3.png',True)
        with self.assertRaisesRegex(ValueError,'currently selected 3'):
            self.project.export(self.root/'excess')
        self.assertFalse((self.root/'excess').exists())

    def test_partial_completion_target_and_saved_selections(self):
        self.mark_test()
        for name in self.project.files[5:]:
            self.project.sample(name)['complete']=False
        # 9.png remains planned test, but only 2.png participates in the partial export.
        progress=self.project.test_progress()
        self.assertEqual((progress['completed'],progress['selected'],progress['target'],progress['planned']),
                         (5,1,1,2))
        output=self.project.export(self.root/'partial')
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['split_counts'],{'train':3,'val':1,'test':1})
        self.assertNotIn('9.png',{s['image'] for s in manifest['samples'].values()})
        self.project.save()
        reopened=Project(self.images,self.project.path)
        self.assertEqual(reopened.data['test_images'],['2.png','9.png'])
        self.assertEqual(reopened.test_progress(),progress)
        self.assertTrue(self.project.sample('2.png')['complete'])

    def test_custom_ratios_and_old_project_migration(self):
        self.project.set_split_fractions(.2,.3)
        for name in ('2.png','9.png','10.png'):
            self.project.set_test_image(name,True)
        output=self.project.export(self.root/'custom')
        self.assertEqual(json.loads((output/'manifest.json').read_text())['split_counts'],
                         {'train':5,'val':2,'test':3})
        with self.assertRaises(ValueError):
            self.project.set_split_fractions(.8,.3)
        with self.assertRaises(ValueError):
            self.project.set_split_fractions(float('nan'),.2)
        original=self.project.sample('2.png').copy()
        self.project.data.pop('test_images')
        self.project.data.pop('split_fractions')
        self.project.save()
        reopened=Project(self.images,self.project.path)
        self.assertEqual(reopened.data['test_images'],[])
        self.assertEqual(reopened.data['split_fractions']['test'],.2)
        self.assertEqual(reopened.sample('2.png'),original)

    def test_zero_ratios_export_and_persist(self):
        for val, test, expected in (
            (0, 1, (0, 0, 10)), (0, 0, (10, 0, 0)), (1, 0, (0, 10, 0)),
            (.2, 0, (8, 2, 0)), (0, .2, (8, 0, 2)), (.8, .2, (0, 8, 2)),
        ):
            with self.subTest(val=val, test=test):
                self.project.set_split_fractions(val, test)
                self.project.data['test_images'] = self.project.files[:expected[2]]
                self.project.save()
                reopened = Project(self.images, self.project.path)
                self.assertEqual(reopened.test_progress()['target'], expected[2])
                output = reopened.export(self.root/f'export_{val}_{test}')
                counts = dict(zip(('train', 'val', 'test'), expected))
                self.assertEqual(json.loads((output/'manifest.json').read_text())['split_counts'], counts)
                for folder in ('images', 'labels', 'adj', 'grasps'):
                    for split, count in counts.items():
                        self.assertEqual(len(list((output/folder/split).iterdir())), count)

    def test_single_image_exports_for_each_exclusive_split(self):
        for name in self.project.files[1:]:
            self.project.sample(name)['complete'] = False
        for val, test, split in ((0, 0, 'train'), (1, 0, 'val'), (0, 1, 'test')):
            with self.subTest(split=split):
                self.project.set_split_fractions(val, test)
                self.project.data['test_images'] = self.project.files[:test]
                output = self.project.export(self.root/split)
                self.assertEqual(json.loads((output/'manifest.json').read_text())['split_counts'],
                                 {s: int(s == split) for s in ('train', 'val', 'test')})
        self.project.sample(self.project.files[0])['complete'] = False
        with self.assertRaises(ValueError):
            self.project.export(self.root/'empty')

    def test_test_membership_still_requires_manual_selection_at_zero_and_100(self):
        for fraction, selected, target in ((0, ['2.png'], 0), (1, [], 10)):
            with self.subTest(fraction=fraction):
                self.project.set_split_fractions(0, fraction)
                self.project.data['test_images'] = selected
                with self.assertRaisesRegex(ValueError, f'Choose exactly {target}'):
                    self.project.export(self.root/'invalid')
                self.assertFalse((self.root/'invalid').exists())

    def test_ratio_dialog_accepts_zero_and_rejects_invalid_percentages(self):
        from scripts.label_dataset import Labeler
        app = SimpleNamespace(project=self.project, root=None, refresh=Mock())
        for value in ('0, 0, 100', '100, 0, 0', '0, 100, 0', '0, 80, 20'):
            with patch('scripts.label_dataset.simpledialog.askstring', return_value=value):
                Labeler.edit_split_ratios(app)
            self.assertEqual(list(self.project.data['split_fractions'].values()),
                             [float(v)/100 for v in value.split(',')])
        for value in ('-1, 1, 100', '100, -1, 1', '100, 1, -1', '0, 0, 0',
                      'nan, 0, 100', '0, inf, 0'):
            with self.subTest(value=value):
                before = self.project.data['split_fractions'].copy()
                with patch('scripts.label_dataset.simpledialog.askstring', return_value=value):
                    with self.assertRaises(ValueError):
                        Labeler.edit_split_ratios(app)
                self.assertEqual(self.project.data['split_fractions'], before)

    def test_test_only_export_merges_with_train_val_and_loads(self):
        from datasets import build_dataset
        self.project.set_split_fractions(0, 1)
        for name in self.project.files:
            self.project.set_test_image(name, True)
        test_only = self.project.export(self.root/'test_only')
        # A separate dataset supplies train/val images with distinct contents.
        other_images = self.root/'other_images'
        other_images.mkdir()
        for i in range(2):
            Image.new('RGB', (50, 40), (0, i*100, 0)).save(other_images/f'{i}.png')
        other = Project(other_images, self.root/'other.json', require_grasps=False)
        other.set_split_fractions(.5, 0)
        for name in other.files:
            other.sample(name).update(objects=[{'id':4,'class_id':0,'box':[5,5,40,35]}],
                                      relations_reviewed=True, complete=True)
        train_val = other.export(self.root/'train_val')
        samples, report = audit([train_val, test_only])
        self.assertEqual({s: c['images'] for s, c in report['counts'].items()},
                         {'train': 1, 'val': 1, 'test': 10})
        merged = self.root/'merged'
        write_dataset(merged, samples, report)
        for split, count in (('train', 1), ('val', 1), ('test', 10)):
            dataset = build_dataset(split, args=SimpleNamespace(dataset_root=merged, no_grasps=True))
            self.assertEqual(len(dataset), count)
            for i in range(count):
                _, target = dataset[i]
                self.assertEqual(target['labels'].tolist(), [0])

    def test_merge_preserves_test_and_loader_reads_it(self):
        from datasets import build_dataset
        self.mark_test()
        output=self.project.export(self.root/'export')
        samples,report=audit([output])
        self.assertEqual(report['counts']['test']['images'],2)
        merged=self.root/'merged'
        write_dataset(merged,samples,report)
        self.assertEqual(len(list((merged/'images'/'test').iterdir())),2)
        dataset=build_dataset('test',args=SimpleNamespace(dataset_root=merged,no_grasps=True))
        self.assertEqual(len(dataset),2)
        _,target=dataset[0]
        self.assertEqual(target['labels'].tolist(),[0])

    def test_duplicate_test_images_never_leak_back_into_train_or_val(self):
        self.mark_test()
        output=self.project.export(self.root/'export')
        manifest=json.loads((output/'manifest.json').read_text())
        paths={split:next((output/'images'/split).iterdir()) for split in ('train','val','test')}
        paths['train'].write_bytes(paths['test'].read_bytes())
        paths['val'].write_bytes(paths['test'].read_bytes())
        # Add an unrelated validation image so the combined val split remains usable.
        import shutil
        stem=paths['val'].stem
        extra='999999'
        Image.new('RGB',(50,40),(0,0,0)).save(output/'images'/'val'/f'{extra}.png')
        for folder in ('labels','adj','grasps'):
            shutil.copyfile(output/folder/'val'/f'{stem}.txt',output/folder/'val'/f'{extra}.txt')
        with self.assertRaisesRegex(ValueError,'different splits'):
            audit([output])
        retained,report=audit([output],duplicate_policy='prefer-heldout')
        self.assertEqual(len(report['excluded_samples']),2)
        test_hashes={s.digest for s in retained if s.split=='test'}
        self.assertFalse(any(s.digest in test_hashes for s in retained if s.split!='test'))
        self.assertEqual(report['counts']['test']['images'],2)


if __name__=='__main__':
    unittest.main()
