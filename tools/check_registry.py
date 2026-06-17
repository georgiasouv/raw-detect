"""Confirm our custom modules registered into mmdet's registries.

Run from the project root, in the mmdet env:
    python tools/check_registry.py

If a name is missing, the usual cause is a missing @register_module decorator
or that rawdet failed to import (run `python -c "import rawdet"` to see the error).
"""
import rawdet  # noqa: F401  -- importing triggers the @register_module decorators

from mmdet.registry import MODELS, TRANSFORMS

EXPECTED_MODELS = ["RAWFrontEnd", "TaskDrivenRAWDetector"]
EXPECTED_TRANSFORMS = ["LoadPackedRAW"]

ok = True
for name in EXPECTED_MODELS:
    found = MODELS.get(name) is not None
    print(f"[{'ok' if found else 'MISSING'}] MODELS.{name}")
    ok &= found
for name in EXPECTED_TRANSFORMS:
    found = TRANSFORMS.get(name) is not None
    print(f"[{'ok' if found else 'MISSING'}] TRANSFORMS.{name}")
    ok &= found

print("REGISTRATION OK" if ok else "REGISTRATION INCOMPLETE -- see MISSING above")
