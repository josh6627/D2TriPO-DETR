"""Restore optimizer and epoch scheduler state without changing learning rates."""


def restore_training_state(optimizer, lr_scheduler, checkpoint, reset_lr=False):
    """Optionally restart the configured LR schedule, retaining optimizer history."""
    configured_rates = [group['lr'] for group in optimizer.param_groups]
    configured_schedule = lr_scheduler.state_dict()
    optimizer.load_state_dict(checkpoint['optimizer'])
    if reset_lr:
        for group, rate in zip(optimizer.param_groups, configured_rates):
            group['lr'] = rate
            group['initial_lr'] = rate
        lr_scheduler.load_state_dict(configured_schedule)
        return checkpoint['epoch'] + 1
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if not checkpoint.get('lr_scheduler_stepped', False):
        # Older main.py checkpoints were written before the end-of-epoch step.
        # Complete that pending step, including a decay at a schedule boundary.
        lr_scheduler.step()
    return checkpoint['epoch'] + 1
