"""rawdet — task-driven RAW preprocessing front-end for mmdetection.

Importing this package runs the @register_module decorators in each submodule,
which is how our custom front-end, wrapper detector, and transforms become
visible to mmdet's registries. A config only needs:

    custom_imports = dict(imports=['rawdet'], allow_failed_imports=False)

NOTE: when you move the modules we built into this package, fix their imports to
be relative, e.g. in detector.py change
    from raw_frontend import RAWFrontEnd   ->   from .frontend import RAWFrontEnd
    from data.packing  import pack_rggb    ->   from .packing  import pack_rggb
"""
from . import packing        # noqa: F401  pack_rggb, normalize_raw
from . import frontend       # noqa: F401  RAWFrontEnd
from . import synergy        # noqa: F401  CoupledGatedStage + measurement
from . import taxonomy       # noqa: F401  class mapping / evaluable set
from . import compute_meter  # noqa: F401  executed-FLOPs meter
from . import detector       # noqa: F401  TaskDrivenRAWDetector, LoadPackedRAW (registers)
