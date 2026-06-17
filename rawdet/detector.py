"""
mmdetection 3.x integration for the task-driven RAW front-end.

Our verified front-end (raw_frontend.RAWFrontEnd) is wrapped around a FROZEN
off-the-shelf mmdet detector. Only the front-end trains; the detector is a fixed
downstream consumer. Provenance of each component is noted inline.

NOTE ON VERIFICATION: the imported front-end is unit-tested (smoke_test.py) and
the gate coupling is tested (synergy.py). The two classes below are framework
glue written to mmdet 3.x / mmengine conventions; they follow the documented
APIs but are NOT executed in our sandbox (mmdet is a heavy install). Check the
exact method signatures against your installed mmdet version.
"""
import numpy as np
import torch

from mmcv.transforms import BaseTransform
from mmdet.registry import MODELS, TRANSFORMS
from mmdet.models.detectors.base import BaseDetector

from .frontend import RAWFrontEnd              # OURS (unit-tested)
from .packing import pack_rggb, normalize_raw  # OURS (verified earlier)


# ---------------------------------------------------------------------------
# Data pipeline transform: read a RAW file and pack it to 4 channels.
# (RAW loading is dataset-specific; this is a template.)
# ---------------------------------------------------------------------------
@TRANSFORMS.register_module()
class LoadPackedRAW(BaseTransform):
    """Read sensor RAW, subtract black level, normalize to [0,1], pack RGGB->4ch."""

    def transform(self, results: dict) -> dict:
        import rawpy
        with rawpy.imread(results['img_path']) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
        packed = pack_rggb(normalize_raw(bayer, black, white))   # [4, H/2, W/2]
        img = np.ascontiguousarray(packed.transpose(1, 2, 0))    # -> HWC for mmdet
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


# ---------------------------------------------------------------------------
# Model: learnable RAW front-end (OURS) + frozen detector (cited, off-the-shelf).
# ---------------------------------------------------------------------------
@MODELS.register_module()
class TaskDrivenRAWDetector(BaseDetector):
    """Front-end-only adaptation of a frozen detector to RAW.

    Provenance:
      - front-end statistics-conditioned global/local processing: adapted from
        histogram-conditioned ISP (RAWild) and FiLM (Perez et al. 2018);
      - frozen-detector-as-loss: AdaptiveISP (Wang et al. 2024);
      - OURS: the gate coupled to sensor statistics (synergy) and the per-sensor
        LoRA deployment paradigm.
    """

    def __init__(self, detector, frontend=dict(type='RAWFrontEnd'),
                 compute_lambda=0.01,
                 pixel_mean=(123.675, 116.28, 103.53),
                 pixel_std=(58.395, 57.12, 57.375),
                 data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.frontend = MODELS.build(frontend)        # OURS
        self.detector = MODELS.build(detector)        # cited; frozen next
        self.compute_lambda = compute_lambda
        for p in self.detector.parameters():
            p.requires_grad_(False)                   # freeze: fixed consumer
        # normalization to match the inner detector's sRGB-pretrained backbone
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Keep the inner detector in eval() so its BatchNorm running stats stay
        # frozen even when the runner puts the wrapper in train mode.
        super().train(mode)
        self.detector.eval()
        return self

    def _frontend_to_input(self, raw):
        rgb01, pis = self.frontend(raw)               # [B,3,H,W] in [0,1], gate probs
        x = (rgb01 * 255.0 - self.pixel_mean) / self.pixel_std
        return x, pis

    def loss(self, batch_inputs, batch_data_samples):
        x, pis = self._frontend_to_input(batch_inputs)
        # gradient flows THROUGH the frozen detector to the front-end:
        losses = self.detector.loss(x, batch_data_samples)
        # OUR compute penalty: lambda * sum_t c_t * pi_t
        compute = sum(stage.cost * pi for stage, pi in zip(self.frontend.stages, pis))
        losses['loss_compute'] = self.compute_lambda * compute
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        x, _ = self._frontend_to_input(batch_inputs)
        return self.detector.predict(x, batch_data_samples, rescale=rescale)

    def _forward(self, batch_inputs, batch_data_samples=None):
        x, _ = self._frontend_to_input(batch_inputs)
        return self.detector._forward(x, batch_data_samples)

    def extract_feat(self, batch_inputs):
        x, _ = self._frontend_to_input(batch_inputs)
        return self.detector.extract_feat(x)
