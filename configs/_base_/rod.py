# ROD (Raw Object Detection) as COCO-format. Annotations in annotations_packed/
# are aligned to the packed grid; category ids already remapped to COCO names:
#   ('person','bicycle','car','train','truck') -> labels 0..4, all real COCO
#   classes, so GT supervises the right neurons of the frozen 80-class detector.
# ROD packed .npy live in per-split subfolders (packed/train, packed/val,
# packed/test); LoadPackedNPY uses basename, so npy_dir is set per split.
# Normalization: raw uint24 counts, black=8.5, robust white=771608 (p99.9 clip).
data_root = '/scratch/INC1526354/rod/'
metainfo = dict(classes=('person', 'bicycle', 'car', 'train', 'truck'))
train_pipeline = [
    dict(type='LoadPackedNPY', npy_dir='/scratch/INC1526354/rod/packed/train',
         black=8.5, white=771608),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RescaleBoxesToPacked'),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadPackedNPY', npy_dir='/scratch/INC1526354/rod/packed/val',
         black=8.5, white=771608),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True),
    dict(type='SetPackedScaleFactor'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations_packed/rod_packed_train.json', data_prefix=dict(img='packed/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations_packed/rod_packed_val.json', data_prefix=dict(img='packed/val/'),
        test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_packed/rod_packed_val.json',
                     metric='bbox')
test_evaluator = val_evaluator
