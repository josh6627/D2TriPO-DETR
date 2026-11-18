"""Manual holdout membership and deterministic three-way export."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

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

    def test_imported_splits_remain_fixed_for_full_and_partial_exports(self):
        fixed = {name: ('test' if name in ('2.png', '9.png', '10.png') else
                        'val' if name in ('3.png', '4.png') else 'train')
                 for name in self.project.files}
        self.project.data['imported_splits'] = fixed
        self.project.data['test_images'] = [n for n, split in fixed.items() if split == 'test']
        self.project.save()
        reopened = Project(self.images, self.project.path)
        self.assertEqual(reopened.fixed_splits, fixed)
        for seed in (42, 999):
            output = reopened.export(self.root/f'imported_{seed}', seed=seed)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['split_counts'], dict(train=5, val=2, test=3))
            self.assertEqual(manifest['split_policy'], 'preserve_imported')
            self.assertEqual({s['image']: s['split'] for s in manifest['samples'].values()}, fixed)
        # No target-ratio check may move reviewed images out of their original split.
        for name in reopened.files:
            reopened.sample(name)['complete'] = name in ('1.png', '2.png', '3.png')
        output = reopened.export(self.root/'partial_imported')
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['split_counts'], dict(train=1, val=1, test=1))
        for sample in manifest['samples'].values():
            self.assertEqual(sample['split'], fixed[sample['image']])
        with self.assertRaisesRegex(ValueError, 'fixed'):
            reopened.set_test_image('1.png', True)
        with self.assertRaisesRegex(ValueError, 'fixed'):
            reopened.set_split_fractions(.1, .2)
        with self.assertRaisesRegex(ValueError, 'fixed'):
            reopened.export(self.root/'override', test_fraction=.2)

    def test_imported_split_map_must_cover_every_image(self):
        self.project.data['imported_splits'] = {'1.png': 'train'}
        with self.assertRaisesRegex(ValueError, 'valid original split'):
            self.project.export(self.root/'bad_import')
        self.assertFalse((self.root/'bad_import').exists())

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
