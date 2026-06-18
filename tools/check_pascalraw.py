"""Draw GT boxes on one packed PASCALRAW sample to verify alignment before training.
    python tools/check_pascalraw.py --out pascalraw_check.png
"""
import argparse
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
import rawdet  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/_base_/pascalraw.py')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--out', default='pascalraw_check.png')
    args = ap.parse_args()

    init_default_scope('mmdet')
    cfg = Config.fromfile(args.config)
    ds = DATASETS.build(cfg.train_dataloader['dataset'])
    sample = ds[args.index]

    inputs = sample['inputs']
    gt = sample['data_samples'].gt_instances
    boxes = gt.bboxes.tensor.numpy()
    labels = gt.labels.numpy()

    r, g1, g2, b = inputs.float().numpy()
    rgb = np.stack([r, (g1 + g2) / 2, b], axis=-1)
    rgb = (255 * (rgb / (rgb.max() + 1e-6)) ** (1 / 2.2)).clip(0, 255).astype(np.uint8)

    names = ['person', 'bicycle', 'car']
    try:
        import cv2
        rgb = np.ascontiguousarray(rgb)
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(rgb, names[int(lab)], (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(args.out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb).save(args.out)
        print("(no opencv -> saved without box overlay)")

    print(f"sample {args.index}: img={tuple(inputs.shape)}  boxes={len(boxes)}  "
          f"labels={labels.tolist()}  (0=person 1=bicycle 2=car)")
    print(f"saved -> {args.out}   OPEN IT: every box must sit on its object")


if __name__ == '__main__':
    main()
