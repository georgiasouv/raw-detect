"""mmdetection 3.x model: task-driven RAW front-end + off-the-shelf detector.

freeze_mode controls how much of the detector adapts -- this is the Phase-2
gradual-unfreezing study, switchable from the config by one string:

    'all'  : backbone + neck + head all frozen (the strict "frozen detector").
             The front-end alone must bridge the RAW->detector gap.
    'head' : backbone + neck frozen (eval, frozen BN); bbox_head trainable.
             Recalibrates the final decision layer to the front-end's features.
    'full' : whole detector fine-tuned end-to-end (upper bound on accuracy,
             lower bound on "off-the-shelf reuse").

('lora' -- backbone weights frozen + low-rank adapters trainable -- is the
fourth variant; it needs adapter modules injected into the backbone and is
added separately once this three-way sweep is in hand.)
"""
import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector

from .frontend import RAWFrontEnd


@MODELS.register_module()
class TaskDrivenRAWDetector(BaseDetector):
    def __init__(self, detector, frontend=dict(type='RAWFrontEnd'),
                 compute_lambda=0.0, freeze_mode='head',
                 detector_checkpoint=None, num_eval_classes=None,
                 pixel_mean=(123.675, 116.28, 103.53),
                 pixel_std=(58.395, 57.12, 57.375),
                 data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.frontend = MODELS.build(frontend)
        self.detector = MODELS.build(detector)
        # load COCO weights BEFORE freezing so 'all'/'head' start from real features
        if detector_checkpoint:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.detector, detector_checkpoint,
                            map_location='cpu', strict=False)
        self.compute_lambda = compute_lambda
        self.num_eval_classes = num_eval_classes
        assert freeze_mode in ('all', 'head', 'full'), freeze_mode
        self.freeze_mode = freeze_mode
        self._apply_freeze()
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(1, 3, 1, 1))

    # --- freezing policy -----------------------------------------------------
    def _apply_freeze(self):
        det = self.detector
        if self.freeze_mode == 'all':
            for p in det.parameters():
                p.requires_grad_(False)
        elif self.freeze_mode == 'head':
            for p in det.parameters():
                p.requires_grad_(False)
            for p in det.bbox_head.parameters():
                p.requires_grad_(True)
        elif self.freeze_mode == 'full':
            for p in det.parameters():
                p.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen sub-modules must stay in eval so their BN running stats don't drift
        if self.freeze_mode == 'all':
            self.detector.eval()
        elif self.freeze_mode == 'head':
            self.detector.eval()
            self.detector.bbox_head.train(mode)
        # 'full': leave whatever super().train(mode) set
        return self

    # --- forward paths -------------------------------------------------------
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