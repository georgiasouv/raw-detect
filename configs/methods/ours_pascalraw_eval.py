# Training + EVALUATION run: front-end learns on PASCALRAW, then we score mAP on val.
# Adds the class filter (num_eval_classes) for the 80-vs-3 mismatch and turns val on.
_base_ = ['../_base_/eval_core.py', '../_base_/pascalraw.py']

model = dict(
    _delete_=True,
    type='TaskDrivenRAWDetector',
    compute_lambda=0.0,
    detector_checkpoint='checkpoints/retinanet_r50_fpn_1x_coco.pth',
    num_eval_classes=3,
    train_head=True,                          # keep person/bicycle/car at eval
    frontend=dict(type='RAWFrontEnd', in_ch=4, feat=16, num_stages=2, upsample=False),
    detector={{_base_.model}},
    data_preprocessor=dict(
        type='DetDataPreprocessor', bgr_to_rgb=False, pad_size_divisor=32),
)

# Modest first run: 1000 iters, validate at 500 and 1000. Raise max_iters for a real
# result; expect a LOW mAP here (front-end barely trained) -- the point is the number prints.
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=1000, val_interval=500)
val_cfg = dict(type='ValLoop')
param_scheduler = [dict(type='ConstantLR', factor=1.0, begin=0, end=1000, by_epoch=False)]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=500, by_epoch=False))
log_processor = dict(by_epoch=False)

# W&B logging -- entity is the team from api.viewer.teams (the writable one)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(entity='georgiasouval-university-of-warwick',
                          project='raw-detect', group='pascalraw',
                          name=None, tags=['frozen-study'],
                          settings=dict(console='auto'))),
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

