_base_ = ['./real_base.py']
model = dict(freeze_mode='full')
optim_wrapper = dict(optimizer=dict(lr=1e-4))
