"""Reproducible experiment harness for the CarPrice paper.

Every study consumes the identical train/calibration/test split from
`experiments.data`, so ablation, model-comparison, and conformal results are
directly comparable and reproducible from the committed processed dataset.
"""
