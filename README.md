# Left Atrium Segmentation
### Deep Learning Models for MRI and CT Segmentation (CARE Challenge)

<p align="center">
<img src="figures/overview.png" width="100%">
</p>

<p align="center">

MRI Joint Segmentation • CT Multi-label Segmentation • Semi-supervised Learning • 3D Medical Image Segmentation

</p>

---

## Overview

This repository contains my implementation for the **CARE Challenge** on Left Atrium segmentation.

Two independent segmentation tasks are included:

| Task | Network | Dataset |
|------|---------|---------|
| MRI Left Atrium Cavity + Scar | Multi-task 3D U-Net | LGE-MRI |
| CT Multi-label Segmentation | Semi-supervised SwinUNETR | CT |

---

# MRI Segmentation

## Pipeline

<p align="center">
<img src="figures/diagram.png" width="95%">
</p>

The MRI framework jointly learns

- LA cavity segmentation
- LA scar segmentation

using a shared encoder and dual decoders.

Unlike conventional two-stage methods, the proposed framework learns anatomical structures and scar distributions simultaneously.

### Main Components

✅ Multi-task 3D U-Net

✅ Spatial Encoding Loss

✅ Shape Attention Loss

✅ Mean Teacher Consistency Learning

---

## MRI Results

### Training Curves

<p align="center">
<img src="figures/MR_1.png" width="95%">
</p>

### Segmentation Examples

<p align="center">
<img src="figures/MR_2.png" width="95%">
</p>

### Performance

| Model | LA Dice (Train) | LA Dice (Val) | Scar Dice (Train) | Scar Dice (Val) |
|------|------:|------:|------:|------:|
| Baseline | **0.99** | **0.90** | **0.60** | **0.10** |
| + Consistency Learning | 0.97 | 0.90 | 0.40 | 0.03 |

The cavity segmentation achieved stable performance on both the training and validation sets. Scar segmentation remained challenging because scar voxels occupy only a very small portion of the image, leading to severe class imbalance and noticeable overfitting.

---

# CT Segmentation

## Pipeline

<p align="center">
<img src="figures/CT_diagram.png" width="90%">
</p>

The CT model adopts a semi-supervised learning framework based on **SwinUNETR**.

Two parallel networks are optimized using **Cross Pseudo Supervision (CPS)** to leverage both labeled and unlabeled CT volumes.

### Main Components

✅ SwinUNETR

✅ MONAI

✅ Cross Pseudo Supervision

✅ Semi-supervised Learning

---

## CT Results

### Training Curves

<p align="center">
<img src="figures/training_summary_CT.png" width="90%">
</p>

### Segmentation Examples

<p align="center">
<img src="figures/CT_results.png" width="90%">
</p>

### Performance

| Dataset | Dice |
|---------|------:|
| Training | **0.98** |
| Validation | **0.88** |

The semi-supervised framework achieved strong performance and generalized well on the validation dataset.

---

# References

1. Li L., *AtrialJSQnet: A New Framework for Joint Segmentation and Quantification of Left Atrium and Scars Incorporating Spatial and Shape Information*. Medical Image Analysis, 2022.

2. Cao H., *Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation*. ECCV, 2022.
