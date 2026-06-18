# Fixed evaluation core: brings the RetinaNet MODEL + schedule + runtime, but NO
# dataset -- so a dataset config can be a separate base without key clashes.
_base_ = [
    'mmdet::_base_/models/retinanet_r50_fpn.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
]

# Make our custom modules visible to the registries before anything is built.
custom_imports = dict(imports=['rawdet'], allow_failed_imports=False)
randomness = dict(seed=0, deterministic=False)

# Train a small front-end, not the detector -> front-end-appropriate optimizer.
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-3, weight_decay=1e-4))
