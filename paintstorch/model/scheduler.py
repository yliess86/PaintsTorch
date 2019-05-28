import torch

from bisect import bisect_right

class LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')

        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        'param initial_lr is not specified',
                        f'in params_group[{i}] when resuming an optimizer'
                    )

        self.base_lrs  = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        list(map(lambda group: group['lr'], self.optimizer.params_group))

    def step(self, this_iter=None):
        if this_iter is None: this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr

class WarmUpLRScheduler(LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr      = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_lr    = base_lr if self.warmup_steps == 0 else warmup_lr
        super(WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            scale = (
                (self.last_iter / self.warmup_steps) * \
                (self.warmup_lr - self.base_lr) + \
                self.base_lr
            ) / self.base_lr

            return [scale * base_lr for base_lr in self.base_lrs]

        else:
            return None

class StepLRScheduler(WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, lr_mults, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(milestones) == len(lr_mults), f'{milestones} vs {lr_mults}'
        for x in milestones: assert isinstance(x, int)
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of',
                f' increasing integers. Got {milestones}'
            )

        self.milestones = milestones
        self.lr_mults   = [1.0]
        for x in lr_mults: self.lr_mults.append(self.lr_mults[-1] * x)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None: return warmup_lr

        pos   = bisect_right(self.milestones, self.last_iter)
        scale = self.warmup_lr * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]
