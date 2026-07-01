"""Compute-penalty warm-up hook: hold compute_lambda at 0 during a warmup, then
ramp it to a target. Efficiency is a second-phase objective -- the detector must
learn to detect before compute pressure is meaningful (flat lambda from step 0
collapses the gates)."""
import math
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ComputeLambdaWarmup(Hook):
    priority = 'NORMAL'

    def __init__(self, target=0.01, warmup_iters=4000, ramp_iters=4000,
                 shape='cosine'):
        assert shape in ('cosine', 'linear')
        self.target = float(target)
        self.warmup_iters = int(warmup_iters)
        self.ramp_iters = max(int(ramp_iters), 1)
        self.shape = shape

    def _lambda_at(self, it):
        if it < self.warmup_iters:
            return 0.0
        prog = (it - self.warmup_iters) / self.ramp_iters
        if prog >= 1.0:
            return self.target
        frac = prog if self.shape == 'linear' else 0.5 * (1.0 - math.cos(math.pi * prog))
        return self.target * frac

    def _model(self, runner):
        m = runner.model
        return m.module if hasattr(m, 'module') else m

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._model(runner).compute_lambda = self._lambda_at(runner.iter)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter % 50 == 0:
            lam = self._model(runner).compute_lambda
            runner.message_hub.update_scalar('train/compute_lambda', lam)
