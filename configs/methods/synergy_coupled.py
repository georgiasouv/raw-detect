_base_ = ['./real_base.py']
model = dict(freeze_mode='head', frontend=dict(couple_gate=True))
optim_wrapper = dict(optimizer=dict(lr=1e-3))
custom_hooks = [
    dict(type='ComputeLambdaWarmup', target=0.01,
         warmup_iters=30000, ramp_iters=30000, shape='cosine'),
]
