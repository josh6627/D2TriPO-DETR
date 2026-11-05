"""Exercise export → loader/transforms → real losses/backprop without grasp labels."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from scripts.label_project import Project, validate
from util.dataset_config import configure_tasks


class NoGraspTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'source'
        self.images.mkdir()
        for i in (1, 2):
            Image.new('RGB', (100, 80)).save(self.images / f'{i}.png')
        self.project = Project(self.images, self.root / 'project.json', require_grasps=False)
        for name in self.project.files:
            sample = self.project.sample(name)
            sample.update(objects=[{'id': 2, 'class_id': 0, 'box': [5, 5, 35, 35]},
                                   {'id': 8, 'class_id': 1, 'box': [40, 10, 90, 70]}],
                          edges=[[2, 8]], relations_reviewed=True, complete=True)
            validate(sample, self.project.classes, (100, 80), complete=True, require_grasps=False)
        self.project.save()

    def test_persist_export_and_infer_mode(self):
        self.assertFalse(Project(self.images, self.project.path).require_grasps)
        output = self.project.export(self.root / 'export')
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['tasks'], {'grasps': False})
        args = SimpleNamespace(dataset_root=output, no_grasps=None)
        self.assertTrue(configure_tasks(args))
        self.assertFalse(configure_tasks(SimpleNamespace()))
        with self.assertRaisesRegex(ValueError, 'grasp-only'):
            configure_tasks(SimpleNamespace(no_grasps=True, modelname='dab_detr'))
        for file in (output / 'grasps').rglob('*.txt'):
            self.assertEqual(file.read_text().strip(), '')
        self.project.set_require_grasps(True)
        self.assertTrue(all(not self.project.sample(n)['complete'] for n in self.project.files))
        with self.assertRaisesRegex(ValueError, 'grasp'):
            validate(self.project.sample('1.png'), self.project.classes, (100, 80), complete=True)

    def test_real_loader_transforms_without_grasp_directory(self):
        from datasets.vmrd import CustomDataset, make_vmrd_transforms
        from datasets.transforms import hflip, vflip
        output = self.project.export(self.root / 'export')
        for split in ('train', 'val'):
            with self.assertRaisesRegex(ValueError, 'No grasp labels'):
                CustomDataset(*(str(output / d / split) for d in ('images', 'labels', 'adj', 'grasps')))[0]
            dataset = CustomDataset(*(str(output / d / split) for d in ('images', 'labels', 'adj')),
                                    str(output / 'nonexistent'), require_grasps=False,
                                    transform=make_vmrd_transforms(split))
            _, target = dataset[0]
            self.assertEqual(target['grasp_points'].shape, (0, 4, 2))
            self.assertEqual(target['grasp_widths'].shape, (0,))
            self.assertEqual(target['grasp_classes'].shape, (0,))
            for flip in (hflip, vflip):
                _, flipped = flip(Image.new('RGB', (100, 80)), target)
                self.assertEqual(flipped['grasp_points'].shape, (0, 4, 2))

    def criterion(self):
        from models.dab_deformable_detr.dab_deformable_detr import SetCriterion
        from models.dab_deformable_detr.matcher import HungarianMatcher
        return SetCriterion(32, {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_adj': 1},
                            ['labels', 'boxes', 'adj'], matcher=HungarianMatcher(),
                            matcher_grasp=Mock(side_effect=AssertionError('Grasp matching must be skipped')))

    def target(self):
        return {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[.2, .2, .2, .2], [.7, .6, .3, .4]]),
                'adj': torch.tensor([[0, 1], [0, 0]]), 'orig_size': torch.tensor([80, 100]),
                'image_id': torch.tensor(1)}

    def predictions(self):
        return {'pred_logits': torch.randn(1, 4, 32, requires_grad=True),
                'pred_boxes': torch.rand(1, 4, 4, requires_grad=True),
                'pred_adj': torch.randn(1, 4, 4, requires_grad=True)}

    def test_loss_backward_skips_grasp_matching_for_aux_and_encoder(self):
        criterion = self.criterion()
        outputs = self.predictions()
        outputs['aux_outputs'] = [self.predictions()]
        outputs['enc_outputs'] = self.predictions()
        # No grasp keys in targets or outputs: no invented negative supervision.
        losses = criterion(outputs, [self.target()])
        self.assertTrue(all('grasp' not in key for key in losses))
        self.assertIn('loss_adj', losses)
        self.assertIn('loss_bbox_0', losses)
        self.assertIn('loss_ce_enc', losses)
        sum(value for key, value in losses.items() if key.startswith('loss_')).backward()
        for name in ('pred_logits', 'pred_boxes', 'pred_adj'):
            self.assertTrue(torch.isfinite(outputs[name].grad).all())
        criterion.matcher_grasp.assert_not_called()

    def test_builder_selects_only_detection_and_relationship_losses(self):
        from main import get_args_parser, build_model_main
        args = get_args_parser().parse_args(['--no-grasps', '--device', 'cpu', '--no-pred-train'])
        with patch('models.dab_deformable_detr.dab_deformable_detr.build_backbone', return_value=Mock()), \
             patch('models.dab_deformable_detr.dab_deformable_detr.build_deforamble_transformer', return_value=Mock()), \
             patch('models.dab_deformable_detr.dab_deformable_detr.DABDeformableDETR', return_value=torch.nn.Linear(1, 1)):
            _, criterion, _ = build_model_main(args)
        self.assertEqual(criterion.losses, ['labels', 'boxes', 'adj'])
        self.assertFalse(any('grasp' in key for key in criterion.weight_dict))

    def test_training_and_evaluation_step_without_grasp_targets(self):
        from engine import train_one_epoch, evaluate
        from models.dab_deformable_detr.dab_deformable_detr import PostProcess

        class TinyModel(torch.nn.Module):
            def __init__(self, predictions):
                super().__init__()
                self.values = torch.nn.ParameterDict(predictions)
                self.grasp_head = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, samples):
                return dict(self.values.items())

        model = TinyModel(self.predictions())
        before = model.values['pred_logits'].detach().clone()
        criterion = self.criterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = SimpleNamespace(amp=False, debug=False, save_results=False)
        batches = [(torch.zeros(1, 3, 80, 100), [self.target()])]
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_one_epoch(model, criterion, batches, optimizer, torch.device('cpu'), 20, args=args)
            val = evaluate(model, criterion, {'bbox': PostProcess()}, batches, torch.device('cpu'), args=args)
        self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        self.assertTrue(torch.isfinite(torch.tensor(val['loss'])))
        self.assertFalse(torch.equal(before, model.values['pred_logits']))
        self.assertIsNone(model.grasp_head.grad)
        self.assertEqual(model.grasp_head.item(), 1.0)
        criterion.matcher_grasp.assert_not_called()


if __name__ == '__main__':
    unittest.main()
