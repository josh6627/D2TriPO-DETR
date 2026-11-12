"""Resume matches uninterrupted optimization, including scheduler boundaries."""
import copy
import unittest

import torch

from util.training_state import restore_training_state


class TrainingResumeTests(unittest.TestCase):
    def setup_training(self, lr=1e-5, lr_drop=3):
        parameters = [torch.nn.Parameter(torch.tensor([1.])),
                      torch.nn.Parameter(torch.tensor([2.]))]
        optimizer = torch.optim.AdamW([
            {'params': parameters[:1], 'lr': lr},
            {'params': parameters[1:], 'lr': lr / 10},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
        return parameters, optimizer, scheduler

    def train_step(self, parameters, optimizer):
        optimizer.zero_grad()
        sum(p.square().sum() for p in parameters).backward()
        optimizer.step()

    def test_resume_preserves_rates_optimizer_and_decay_schedule(self):
        # Test checkpoints on either side of, and exactly at, a decay boundary.
        for stepped in (False, True):
            for saved_epoch in (1, 2, 3, 5):
                with self.subTest(stepped=stepped, saved_epoch=saved_epoch):
                    parameters, optimizer, scheduler = self.setup_training()
                    for epoch in range(saved_epoch + 1):
                        self.train_step(parameters, optimizer)
                        if epoch != saved_epoch or stepped:
                            scheduler.step()
                    checkpoint = copy.deepcopy({
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(), 'epoch': saved_epoch,
                    })
                    if stepped:
                        checkpoint['lr_scheduler_stepped'] = True
                    else:
                        scheduler.step()  # What uninterrupted legacy training does next.

                    resumed, resumed_optimizer, resumed_scheduler = self.setup_training(lr=.5, lr_drop=99)
                    for new, old in zip(resumed, parameters):
                        new.data.copy_(old.data)
                    self.assertEqual(restore_training_state(resumed_optimizer, resumed_scheduler, checkpoint),
                                     saved_epoch + 1)
                    self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
                    self.assertEqual(resumed_scheduler.step_size, 3)
                    for _ in range(6):
                        self.assertEqual([g['lr'] for g in resumed_optimizer.param_groups],
                                         [g['lr'] for g in optimizer.param_groups])
                        self.train_step(parameters, optimizer)
                        self.train_step(resumed, resumed_optimizer)
                        scheduler.step()
                        resumed_scheduler.step()
                        for actual, expected in zip(resumed, parameters):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reset_rates_preserves_moments_and_restarts_decay_interval(self):
        parameters, optimizer, scheduler = self.setup_training()
        for _ in range(6):
            self.train_step(parameters, optimizer)
            scheduler.step()
        checkpoint = copy.deepcopy({
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(), 'epoch': 49,
            'lr_scheduler_stepped': True,
        })
        resumed, new_optimizer, new_scheduler = self.setup_training(lr=2e-5, lr_drop=4)
        self.assertEqual(restore_training_state(
            new_optimizer, new_scheduler, checkpoint, reset_lr=True), 50)
        for old, new in zip(parameters, resumed):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(new_optimizer.state[new][key], optimizer.state[old][key])
        for epoch in range(4):
            self.assertAlmostEqual(new_optimizer.param_groups[0]['lr'], 2e-5)
            self.assertAlmostEqual(new_optimizer.param_groups[1]['lr'], 2e-6)
            self.train_step(resumed, new_optimizer)
            new_scheduler.step()
        self.assertAlmostEqual(new_optimizer.param_groups[0]['lr'], 2e-6)
        self.assertAlmostEqual(new_optimizer.param_groups[1]['lr'], 2e-7)
        for actual, expected in zip(new_scheduler.base_lrs, [2e-5, 2e-6]):
            self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
