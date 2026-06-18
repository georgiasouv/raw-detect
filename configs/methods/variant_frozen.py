# VARIANT 1/3 -- strict frozen detector (backbone + neck + head all frozen).
# The front-end alone must bridge RAW -> detector. Expected: loss barely moves
# (you saw this: ~2.13 flat). This is the lower bound / the failure case.
_base_ = ['./ours_pascalraw_eval.py']
model = dict(freeze_mode='all')