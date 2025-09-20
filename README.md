# Car Damage Classification (Multi‑Label)

## Single Required File

- `car-condition-multilabel.ipynb` (Colab‑ready, outputs cleared)

## Problem Addressed

- Clean cars were often misclassified as scratch due to a lack of clean examples.
- Solution: multi‑label binary classification; map dataset category `car` to label `clean` to enrich the clean class.

## How to Run (Google Colab)

1) Upload `car-condition-multilabel.ipynb` to Google Colab.
2) Set your Roboflow API key before running the dataset cell:
```python
%env ROBOFLOW_API_KEY=your_key_here
```
3) Runtime → Run all.

Notes:
- The notebook downloads two Roboflow datasets automatically. Do not change the dataset count.
- All secrets are read from environment variables; the notebook does not hardcode keys.

## Approach

Multi‑label binary classification with five independent labels: `scratch`, `dent`, `rust`, `dirt`, `clean`.

Key points:
- Interpret category `car` as the `clean` label when no damage is present to enrich the clean class.
- Use balanced augmentations and class‑weighted loss to handle class imbalance.
- Provide an interactive testing cell to validate on your own images.

## Results (two‑dataset setup)

- Best macro F1: 0.8352
- scratch: 0.750
- dent:   0.714
- rust:   0.889
- dirt:   0.944
- clean:  0.879

## Repository Structure

- `car-condition-multilabel.ipynb` — main notebook (training and testing)
- `README.md` — this document

## Why this works

1) Enriching the clean class (via `car` → `clean`) addresses the root cause.
2) Multi‑label setup models co‑occurring conditions (e.g., dirt + scratch).
3) Balanced augmentations, class weighting, and thresholding improve robustness.

## Technical Details

Model:
- Backbone: EfficientNetV2‑S (timm)
- Loss: BCEWithLogitsLoss with class weights (per‑class pos_weight)
- Optimizer/Scheduler: AdamW + CosineAnnealingWarmRestarts
- Image size: typically 512; batch size tuned for Colab GPUs
- Augmentations: Resize, HorizontalFlip, Rotate, RandomBrightnessContrast, Blur (Albumentations)

Data:
- Combines multiple Roboflow datasets into one multi‑label table
- Two‑dataset configuration: dirt/clean + mixed damage (car/dunt/rust/scratch)
- Four‑dataset configuration optional (adds more specialized sources)

## Usage (after training)

Within the notebook, call:
```python
probs = test_prediction("path/to/clean_car.jpg")
# Expectation: clean > 0.8, scratch < 0.1
```

## Reproducibility and Safety

- Outputs are cleared for clean diffs.
- Secrets are not stored in code. Use the `ROBOFLOW_API_KEY` environment variable (see How to Run).
- Interactive testing supports common formats (JPEG/PNG).
