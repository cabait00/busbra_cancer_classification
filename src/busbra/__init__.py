"""Shared implementation of the BUS-BRA project.

config, data, image_processing  paths, fixed design, split manifest, model inputs
models, training, inference     model construction, classification training, inference (TensorFlow is imported inside the functions)
segmentation, segmentation_model, roi, auto_roi   lesion masks, U-Net, expert-mask and automatic ROI
evaluation, thresholds, final_evaluation, test_evaluation, aggregation, robustness_test, adversarial_roi, gradcam_test, model_diagnostics   metrics, operating points, bootstrap, studies
results                         read access to the stored result tables for the plot scripts

Importing a module has no side effects: nothing is read, written, trained or executed.
"""
