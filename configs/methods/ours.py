_base_ = ['../_base_/eval_core.py']
model = dict(
    _delete_=True,
    type='TaskDrivenRAWDetector',
    compute_lambda=0.01,
    frontend=dict(type='RAWFrontEnd', in_ch=4, feat=16, num_stages=2),
    detector={{_base_.model}},
    data_preprocessor=dict(
        type='DetDataPreprocessor', bgr_to_rgb=False, pad_size_divisor=32),
)
