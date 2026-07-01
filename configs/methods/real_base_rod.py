# ROD training substrate: same 100-epoch schedule/optimizer/hooks as real_base.py,
# but inherits the ROD eval config (dataset + 5-class filter) instead of PASCALRAW.
_base_ = ['./ours_rod_eval.py']
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=100, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True,
         begin=0, end=1, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', by_epoch=True,
         begin=1, end=100, eta_min=1e-6, convert_to_iter_based=True),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True,
                    interval=5, max_keep_ckpts=3,
                    save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', rule='greater',
         patience=8, min_delta=0.001),
]
custom_imports = dict(imports=['rawdet'], allow_failed_imports=False)
