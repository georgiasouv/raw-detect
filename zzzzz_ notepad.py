python - <<'EOF'
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

CONFIG = 'configs/methods/real_base_rod.py'          # <-- set to your ROD config

cfg = Config.fromfile(CONFIG)
if cfg.get('custom_imports'):
    from mmengine.utils import import_modules_from_strings
    import_modules_from_strings(**cfg.custom_imports)
init_default_scope(cfg.get('default_scope', 'mmdet'))

from mmengine.runner import Runner
from mmdet.registry import MODELS

for name in ('val_dataloader', 'train_dataloader'):
    cfg[name].batch_size, cfg[name].num_workers, cfg[name].persistent_workers = 1, 0, False

model = MODELS.build(cfg.model)
model.init_weights()

def T(b):
    return b.tensor if hasattr(b, 'tensor') else b

with torch.no_grad():
    vb = next(iter(Runner.build_dataloader(cfg.val_dataloader)))
    pre = tuple(vb['data_samples'][0].metainfo['img_shape'])
    pre_sf = tuple(vb['data_samples'][0].metainfo['scale_factor'])
    vb = model.data_preprocessor(vb, False)
    model.eval()
    r = model.predict(vb['inputs'], vb['data_samples'])[0]
    post = tuple(r.metainfo['img_shape'])
    post_sf = tuple(r.metainfo['scale_factor'])
    ok1 = post == (2 * pre[0], 2 * pre[1])
    ok2 = all(abs(a - 2 * b) < 1e-6 for a, b in zip(post_sf, pre_sf))
    print(f"[INFO] img_shape    pre {pre}  ->  post {post}")
    print(f"[INFO] scale_factor pre ({pre_sf[0]:.4f}, {pre_sf[1]:.4f})  ->  post ({post_sf[0]:.4f}, {post_sf[1]:.4f})")
    print(f"[{'PASS' if ok1 else 'FAIL'}] predict: img_shape doubled")
    print(f"[{'PASS' if ok2 else 'FAIL'}] predict: scale_factor doubled")
    bb = T(r.pred_instances.bboxes)
    if len(bb):
        print(f"[INFO] {len(bb)} predictions, x_max {float(bb[:,2].max()):.1f}, y_max {float(bb[:,3].max()):.1f}  (JSON space)")

    tb = next(iter(Runner.build_dataloader(cfg.train_dataloader)))
    tb = model.data_preprocessor(tb, True)
    g_pre = T(tb['data_samples'][0].gt_instances.bboxes)[0].clone()
    model.train()
    model.loss(tb['inputs'], tb['data_samples'])
    g_post = T(tb['data_samples'][0].gt_instances.bboxes)[0]
    ok3 = bool(torch.allclose(g_post, 2 * g_pre, atol=0.5))
    print(f"[INFO] GT box pre  {[round(v, 1) for v in g_pre.tolist()]}")
    print(f"[INFO] GT box post {[round(v, 1) for v in g_post.tolist()]}")
    print(f"[{'PASS' if ok3 else 'FAIL'}] loss: train GT lifted x2 into the detector's space")
EOF