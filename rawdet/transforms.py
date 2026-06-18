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

    def transform(self, results: dict):
        import rawpy
        try:
            with rawpy.imread(results['img_path']) as raw:
                bayer = raw.raw_image_visible.astype(np.float32)
                black = float(np.mean(raw.black_level_per_channel))
                white = float(raw.white_level)
        except (rawpy.LibRawError, rawpy.LibRawDataError, OSError) as e:
            import warnings
            warnings.warn(f"skipping unreadable RAW {results['img_path']}: {e}")
            return None
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


@TRANSFORMS.register_module()
class LoadPackedNPY(BaseTransform):
    """Load a PRE-PACKED .npy (float16 [4, H/2, W/2]) from tools/prep_dataset.py
    instead of decoding the .nef at train time. A missing .npy returns None."""

    def __init__(self, npy_dir):
        self.npy_dir = npy_dir

    def transform(self, results: dict):
        import os
        stem = os.path.splitext(os.path.basename(results['img_path']))[0]
        npy = os.path.join(self.npy_dir, stem + '.npy')
        if not os.path.exists(npy):
            return None
        packed = np.load(npy).astype(np.float32)
        img = np.ascontiguousarray(packed.transpose(1, 2, 0))
        ph, pw = img.shape[:2]
        jw = int(results.get('width') or pw * 2)
        jh = int(results.get('height') or ph * 2)
        results['json_wh'] = (jw, jh)
        results['img'] = img
        results['img_shape'] = (ph, pw)
        results['ori_shape'] = (jh, jw)
        return results
