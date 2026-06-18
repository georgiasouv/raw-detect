_base_ = ['./real_base.py']
model = dict(freeze_mode='head')
optim_wrapper = dict(optimizer=dict(lr=1e-3))
