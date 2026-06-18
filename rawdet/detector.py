"""mmdetection 3.x model: task-driven RAW front-end + FROZEN off-the-shelf detector."""
import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector

from .frontend import RAWFrontEnd


@MODELS.register_module()
class TaskDrivenRAWDetector(BaseDetector):
    def __init__(self, detector, frontend=dict(type='RAWFrontEnd'),
                 compute_lambda=0.01, detector_checkpoint=None, num_eval_classes=None,
                 train_head=False,
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
        # let the detector's HEAD adapt to the front-end's output while the
        # backbone+neck stay frozen -- stabilizes the target, unblocks learning.
        if train_head:
            for p in self.detector.bbox_head.parameters():
                p.requires_grad_(True)
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        # keep frozen sub-modules in eval (fixed BN stats); head follows `mode`
        self.detector.backbone.eval()
        self.detector.neck.eval()
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
