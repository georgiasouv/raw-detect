#!/usr/bin/env bash
# Applies the evaluation layer: class filter + JSON-space rescale + eval config.
# Run from the project root.  Then: python tools/train.py configs/methods/ours_pascalraw_eval.py
set -euo pipefail
mkdir -p rawdet configs/_base_ configs/methods
cat > rawdet/transforms.py << 'EVAL_EOF_8c5f'
"""Data-pipeline transforms: packed-RAW loading, box rescaling, and test-time scale_factor."""
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

from .packing import pack_rggb, normalize_raw


@TRANSFORMS.register_module()
class LoadPackedRAW(BaseTransform):
    """Read sensor RAW (.nef), normalize to [0,1], pack RGGB -> 4-channel HWC.

    ori_shape is set to the JSON annotation size so predictions can be rescaled
    back to it at eval; img_shape is the packed grid (Resize updates it later).
    """

    def transform(self, results: dict) -> dict:
        import rawpy
        with rawpy.imread(results['img_path']) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
        packed = pack_rggb(normalize_raw(bayer, black, white))    # [4, H/2, W/2]
        img = np.ascontiguousarray(packed.transpose(1, 2, 0))     # HWC, 4 channels
        ph, pw = img.shape[:2]

        jw = int(results.get('width') or pw * 2)
        jh = int(results.get('height') or ph * 2)
        results['json_wh'] = (jw, jh)

        results['img'] = img
        results['img_shape'] = (ph, pw)
        results['ori_shape'] = (jh, jw)        # JSON space (original annotation size)
        return results


@TRANSFORMS.register_module()
class RescaleBoxesToPacked(BaseTransform):
    """Scale GT boxes from JSON annotation space onto the packed image grid.
    Insert AFTER LoadAnnotations and BEFORE any Resize."""

    def transform(self, results: dict) -> dict:
        boxes = results.get('gt_bboxes', None)
        if boxes is None or len(boxes) == 0:
            return results
        ph, pw = results['img_shape']
        jw, jh = results['json_wh']
        sx, sy = pw / jw, ph / jh
        if hasattr(boxes, 'tensor'):
            boxes.tensor[:, 0::2] *= sx
            boxes.tensor[:, 1::2] *= sy
        else:
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
        return results


@TRANSFORMS.register_module()
class SetPackedScaleFactor(BaseTransform):
    """TEST pipeline: set scale_factor so predict(rescale=True) maps predictions
    from the (packed+resized) input back to JSON space. Place AFTER Resize.
    scale_factor = input/ori = resized/json (predict divides boxes by it)."""

    def transform(self, results: dict) -> dict:
        ih, iw = results['img_shape']          # resized (current)
        jw, jh = results['json_wh']            # JSON (original)
        results['scale_factor'] = (iw / jw, ih / jh)
        return results
EVAL_EOF_8c5f
cat > rawdet/detector.py << 'EVAL_EOF_8c5f'
"""mmdetection 3.x model: task-driven RAW front-end + FROZEN off-the-shelf detector."""
import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector

from .frontend import RAWFrontEnd


@MODELS.register_module()
class TaskDrivenRAWDetector(BaseDetector):
    def __init__(self, detector, frontend=dict(type='RAWFrontEnd'),
                 compute_lambda=0.01, detector_checkpoint=None, num_eval_classes=None,
                 pixel_mean=(123.675, 116.28, 103.53),
                 pixel_std=(58.395, 57.12, 57.375),
                 data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.frontend = MODELS.build(frontend)
        self.detector = MODELS.build(detector)
        if detector_checkpoint:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.detector, detector_checkpoint,
                            map_location='cpu', strict=False)
        self.compute_lambda = compute_lambda
        # keep only the first N COCO classes at eval (e.g. 3 = person/bicycle/car);
        # None keeps all 80. The frozen detector still predicts 80; we filter here.
        self.num_eval_classes = num_eval_classes
        for p in self.detector.parameters():
            p.requires_grad_(False)
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.detector.eval()
        return self

    def _frontend_to_input(self, raw):
        rgb01, pis = self.frontend(raw)
        x = (rgb01 * 255.0 - self.pixel_mean) / self.pixel_std
        return x, pis

    def loss(self, batch_inputs, batch_data_samples):
        x, pis = self._frontend_to_input(batch_inputs)
        losses = self.detector.loss(x, batch_data_samples)
        compute = sum(stage.cost * pi for stage, pi in zip(self.frontend.stages, pis))
        losses['loss_compute'] = self.compute_lambda * compute
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        x, _ = self._frontend_to_input(batch_inputs)
        results = self.detector.predict(x, batch_data_samples, rescale=rescale)
        if self.num_eval_classes is not None:
            for ds in results:
                inst = ds.pred_instances
                ds.pred_instances = inst[inst.labels < self.num_eval_classes]
        return results

    def _forward(self, batch_inputs, batch_data_samples=None):
        x, _ = self._frontend_to_input(batch_inputs)
        return self.detector._forward(x, batch_data_samples)

    def extract_feat(self, batch_inputs):
        x, _ = self._frontend_to_input(batch_inputs)
        return self.detector.extract_feat(x)
EVAL_EOF_8c5f
cat > configs/_base_/pascalraw.py << 'EVAL_EOF_8c5f'
# PASCALRAW as COCO-format. Annotations in json_raw/ are aligned to the .nef RAW.
# Class ORDER maps category ids to COCO-contiguous labels:
#   ('person','bicycle','car') -> labels 0,1,2 == COCO person/bicycle/car,
# so GT supervises the right neurons of the frozen 80-class COCO detector.
# (JSON cat ids are person=1,car=2,bicycle=3; get_cat_ids(cat_names=...) reorders.)
data_root = 'data/PASCALRAW/'
metainfo = dict(classes=('person', 'bicycle', 'car'))

train_pipeline = [
    dict(type='LoadPackedRAW'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RescaleBoxesToPacked'),                 # JSON boxes -> packed grid
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadPackedRAW'),
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
        _delete_=True, type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='json_raw/train.json', data_prefix=dict(img='raw/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True, type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='json_raw/val.json', data_prefix=dict(img='raw/'),
        test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'json_raw/val.json',
                     metric='bbox')
test_evaluator = val_evaluator
EVAL_EOF_8c5f
cat > configs/methods/ours_pascalraw_eval.py << 'EVAL_EOF_8c5f'
# Training + EVALUATION run: front-end learns on PASCALRAW, then we score mAP on val.
# Adds the class filter (num_eval_classes) for the 80-vs-3 mismatch and turns val on.
_base_ = ['../_base_/eval_core.py', '../_base_/pascalraw.py']

model = dict(
    _delete_=True,
    type='TaskDrivenRAWDetector',
    compute_lambda=0.01,
    detector_checkpoint='checkpoints/retinanet_r50_fpn_1x_coco.pth',
    num_eval_classes=3,                          # keep person/bicycle/car at eval
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
EVAL_EOF_8c5f
echo "applied eval layer (4 files)"