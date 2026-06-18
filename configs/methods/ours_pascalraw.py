_base_ = ['../_base_/eval_core.py', '../_base_/pascalraw.py']

model = dict(
    _delete_=True,
    type='TaskDrivenRAWDetector',
    compute_lambda=0.01,
    detector_checkpoint='checkpoints/retinanet_r50_fpn_1x_coco.pth',
    frontend=dict(type='RAWFrontEnd', in_ch=4, feat=16, num_stages=2, upsample=False),
    detector={{_base_.model}},
    data_preprocessor=dict(
        type='DetDataPreprocessor', bgr_to_rgb=False, pad_size_divisor=32),
)

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=30, val_interval=10**9)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
param_scheduler = [dict(type='ConstantLR', factor=1.0, begin=0, end=30, by_epoch=False)]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=10**9))
log_processor = dict(by_epoch=False)
