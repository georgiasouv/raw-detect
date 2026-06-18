import argparse, torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS
import rawdet  # noqa: F401

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config')
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()
    init_default_scope('mmdet')
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).eval()
    x = torch.rand(2, 4, args.size, args.size)
    with torch.no_grad():
        feats = model.extract_feat(x)
    print("FPN feature maps out of the frozen detector:")
    for i, f in enumerate(feats):
        print(f"  level {i}: {tuple(f.shape)}")
    n_train = sum(p.requires_grad for p in model.detector.parameters())
    print(f"trainable params in inner detector : {n_train}   (want 0)")
    print(f"inner detector in training mode    : {model.detector.training}   (want False)")
    print("SMOKE BUILD OK")

if __name__ == '__main__':
    main()
