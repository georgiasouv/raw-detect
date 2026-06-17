"""Training entry for mmdet-installed-as-a-package.

Because mmdet is a dependency (not a cloned source tree), there is no
tools/train.py to copy. The training engine actually lives in mmengine, so a
Runner built from your config is all you need:

    python tools/train.py configs/methods/ours.py
    python tools/train.py configs/methods/ours.py --work-dir work_dirs/ours_seed0

Registration of our custom modules happens two ways (either suffices):
  1. `custom_imports=dict(imports=['rawdet'])` in the config (canonical), or
  2. the `import rawdet` below (explicit, robust even without custom_imports).
"""
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

import rawdet  # noqa: F401  -- side effect: registers all custom modules


def main():
    parser = argparse.ArgumentParser(description="Train via mmengine Runner")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--work-dir", default=None, help="override output dir")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
