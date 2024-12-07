# Knowledge Distillation with Vision Transformer and ResNet on CIFAR-10

This repository contains a Python script that demonstrates **knowledge distillation**, a technique where a smaller "student" model learns from the predictions of a larger "teacher" model. In this example:
- The **teacher model** is a Vision Transformer (ViT-B/16).
- The **student model** is a ResNet-50.

> **Disclaimer**: This implementation is primarily for **learning purposes** and is semantically flawed because the teacher and student models are trained on **different datasets and class distributions** (ImageNet for ViT vs. CIFAR-10 for ResNet). Use this example with caution, and ensure your teacher and student are trained on the same dataset for proper distillation.

---

## Overview

### What is Knowledge Distillation?
Knowledge distillation transfers knowledge from a larger, complex model (teacher) to a smaller, simpler model (student) by leveraging:
1. **Soft targets**: Predictions from the teacher model.
2. **Hard targets**: Ground-truth labels.

The student learns from a combination of these two sources of supervision.

### Script Features
- **Dataset**: CIFAR-10, resized to `224x224` to be compatible with the Vision Transformer.
- **Teacher Model**: Vision Transformer (`vit_b_16`) pre-trained on ImageNet.
- **Student Model**: ResNet-50 (`resnet50`) pre-trained on ImageNet, modified for CIFAR-10.
- **Distillation Loss**: Combines Cross-Entropy Loss (hard labels) and KL Divergence (soft labels).

---

## Usage

### Requirements
- Python 3.8+
- PyTorch and TorchVision
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision matplotlib
```

### Run the Script

To train the model:
```bash
python train.py
```


### Script Breakdown

1. **Dataset**:
   - Downloads the CIFAR-10 dataset.
   - Resizes images to `224x224` for compatibility with Vision Transformer (ViT).
   - Splits the training set into two subsets:
     - **Training subset**: 80% of the data.
     - **Validation subset**: 20% of the data.

2. **Teacher and Student Models**:
   - **Teacher Model**: A Vision Transformer (`vit_b_16`) pre-trained on ImageNet, which outputs predictions for 1,000 classes.
   - **Student Model**: A ResNet-50 (`resnet50`) pre-trained on ImageNet, modified to output predictions for 10 classes (CIFAR-10).
   - The teacher model is frozen (not updated during training), while the student model is trained.

3. **Distillation Loss**:
   - A custom loss function combines:
     - **KL Divergence**: Encourages the student to mimic the teacher’s softened probability distributions (soft labels).
     - **Cross-Entropy Loss**: Trains the student to correctly predict the ground-truth labels (hard labels).
   - A temperature parameter smooths the teacher’s logits to make them more informative for the student.

4. **Training**:
   - The training loop updates the student model using the distillation loss.
   - The student learns from both:
     - The teacher’s predictions.
     - The ground-truth labels of CIFAR-10.
   - Periodic validation is performed after each epoch.

5. **Visualization (Optional)**:
   - The script allows for visualizing training inputs and labels.
   - To enable visualization, set `vis = True`. The script will display one image per batch, denormalized to `[0, 1]` for proper viewing.

---

### Limitations and Recommendations

#### Limitations
- The teacher model (`vit_b_16`) is pre-trained on ImageNet (1,000 classes), which has a different class distribution than CIFAR-10 (10 classes). This makes the distillation process semantically flawed.
- KL divergence expects the teacher’s and student’s outputs to have the same size. Directly modifying the student to output 10 classes while the teacher outputs 1,000 classes creates a mismatch.

#### Recommendations
- Unify the datasets Either fine-tune or train the teacher model on the CIFAR-10 dataset before distillation, or use ImageNet-pretrained models for both teacher and student.
- Ensure both the teacher and student models are trained or fine-tuned on the same dataset and class distribution for meaningful knowledge transfer.

This implementation is for **learning purposes** and demonstrates the general process of knowledge distillation, but should not be used as-is for practical applications without addressing these limitations.

