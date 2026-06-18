# PASCALRAW as COCO-format. Annotations in json_raw/ are aligned to the .nef RAW.
# Class ORDER maps category ids to COCO-contiguous labels:
#   ('person','bicycle','car') -> labels 0,1,2 == COCO person/bicycle/car,
# so GT supervises the right neurons of the frozen 80-class COCO detector.
# (JSON cat ids are person=1,car=2,bicycle=3; get_cat_ids(cat_names=...) reorders.)
data_root = '/scratch/INC1526354/pascalraw/'
metainfo = dict(classes=('person', 'bicycle', 'car'))

train_pipeline = [
    dict(type='LoadPackedNPY', npy_dir='/scratch/INC1526354/pascalraw/packed'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RescaleBoxesToPacked'),                 # JSON boxes -> packed grid
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadPackedNPY', npy_dir='/scratch/INC1526354/pascalraw/packed'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='SetPackedScaleFactor'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='json_raw/train.json', data_prefix=dict(img='raw/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='json_raw/val.json', data_prefix=dict(img='raw/'),
        test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'json_raw/val.json',
                     metric='bbox')
test_evaluator = val_evaluator
