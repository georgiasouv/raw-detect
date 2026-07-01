# ROD training + EVALUATION run: front-end learns on ROD, score mAP on val.
# Same frozen-80-class-COCO design as PASCALRAW; num_eval_classes=5 for ROD's
# 5 COCO-aligned classes (person/bicycle/car/train/truck). Dataset base = rod.py.
_base_ = ['../_base_/eval_core.py', '../_base_/rod.py']
model = dict(
    _delete_=True,
    type='TaskDrivenRAWDetector',
    compute_lambda=0.0,
    detector_checkpoint='checkpoints/retinanet_r50_fpn_1x_coco.pth',
    num_eval_classes=5,
    frontend=dict(type='RAWFrontEnd', in_ch=4, feat=16, num_stages=2, upsample=True),
    detector={{_base_.model}},
    data_preprocessor=dict(
        type='DetDataPreprocessor', bgr_to_rgb=False, pad_size_divisor=32),
)
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=1000, val_interval=500)
val_cfg = dict(type='ValLoop')
param_scheduler = [dict(type='ConstantLR', factor=1.0, begin=0, end=1000, by_epoch=False)]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=500, by_epoch=False))
log_processor = dict(by_epoch=False)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(entity='georgiasouval-university-of-warwick',
                          project='raw-detect', group='rod',
                          name=None, tags=['synergy'],
                          settings=dict(console='auto'))),
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
