# SimNCSD: Source-Anchored Semantic Deviation Detection in News Comments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of the paper: **"Source-Anchored Semantic Deviation Detection in News Comments via Supervised Cross-Text Contrastive Learning"**.

## 📖 Overview

In the digital news ecosystem, source-anchored semantic deviation hidden in massive user comments seriously interferes with the objective dissemination of information. **SimNCSD** (Supervised Cross-Text Semantic Alignment and News Comment Semantic Deviation) is designed to address the dual bottlenecks of feature space anisotropic collapse and literal overlap-induced misclassification.

### Key Features:
* **Dual-Tower RoBERTa Architecture:** Extracts decoupled deep semantic features for long source posts and short comments independently.
* **Supervised Cross-Text Contrastive Learning:** Introduces an InfoNCE loss function with explicit manually verified hard negative samples (source-deviant comments) to increase the contrastive margin.
* **Multi-Channel Feature Fusion:** Incorporates element-wise absolute difference and Hadamard product to capture fine-grained semantic conflicts.
* **Two-Stage Training Framework:** Separates contrastive representation fine-tuning from classification decision optimization.
* **Validation-Set Calibrated Operating Threshold:** Optimizes the Youden index for robust discrete classification.

## 📂 Project Structure

```text
SimNCSD/
├── configs/
│   └── base.yaml                 # Global hyperparameter configurations
├── data/
│   └── processed/                # Generated datasets (train/val/test)
├── src/
│   ├── data/
│   │   └── dataset.py            # Tokenization and Dataset class
│   ├── evaluation/
│   │   ├── evaluate.py           # Metric calculation (Precision, Recall, F1, AUC)
│   │   └── threshold.py          # Youden index threshold calibration
│   ├── losses/
│   │   └── info_nce.py           # Supervised Cross-Text InfoNCE loss
│   ├── models/
│   │   └── simncsd.py            # Dual-tower encoder & multi-channel fusion MLP
│   ├── training/
│   │   ├── train_contrastive.py  # Stage 1: Contrastive learning
│   │   └── train_classifier.py   # Stage 2: Classifier training
│   ├── utils/
│   │   └── seed.py               # Reproducibility utilities
│   └── visualization/
│       └── plot_results.py       # ROC, Confusion Matrix, Probability distributions
├── outputs/
│   └── figures/                  # Directory for generated evaluation plots
├── generate_mock_data.py         # Script to generate dummy data for quick testing
├── main.py                       # Main execution pipeline
└── requirements.txt              # Environment dependencies
