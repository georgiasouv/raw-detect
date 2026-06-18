# VARIANT 2/3 -- frozen backbone + neck, TRAINABLE head (current method).
# Recalibrates the final decision layer to the front-end's features.
# Expected: loss_cls collapses fast (~143 -> ~0.7). The working baseline.
_base_ = ['./ours_pascalraw_eval.py']
model = dict(freeze_mode='head')