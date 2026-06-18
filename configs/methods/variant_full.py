# VARIANT 3/3 -- full end-to-end fine-tune of the detector.
# Upper bound on accuracy, but abandons "off-the-shelf reuse". NOTE: full FT from
# COCO init may want a LOWER lr than head-only (1e-3 AdamW can wreck pretrained
# features); for the matched-schedule DIAGNOSTIC keep lr equal across variants,
# then if 'full' wins, re-run it with lr ~1e-4 for the real number.
_base_ = ['./ours_pascalraw_eval.py']
model = dict(freeze_mode='full')