# ROD UNCOUPLED synergy (Syn05 analogue): gate blind to s(x) (control). freeze_mode=head.
_base_ = ['./real_base_rod.py']
model = dict(freeze_mode='head', frontend=dict(couple_gate=False))
optim_wrapper = dict(optimizer=dict(lr=1e-3))
custom_hooks = [
    dict(type='ComputeLambdaWarmup', target=0.01,
         warmup_iters=30000, ramp_iters=30000, shape='cosine'),
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', rule='greater',
         patience=8, min_delta=0.001),
]
