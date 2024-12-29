# APAR: Modeling Irregular Target Functions in Tabular Regression via Arithmetic-Aware Pre-Training and Adaptive-Regularized Fine-Tuning [AAAI 2025]

## Overview

This repository contains the official implementation of **APAR**, introduced in our AAAI 2025 paper:  
[**Modeling Irregular Target Functions in Tabular Regression**](https://arxiv.org/abs/2412.10941).

### Abstract

Tabular regression plays a crucial role in diverse fields such as finance, genomics, and healthcare. However, deep learning (DL) methods often struggle to outperform traditional machine learning (ML) models due to the challenges of irregular target functions in tabular data—where small changes in input features can result in significant label shifts.

To address these challenges, we propose **APAR (Arithmetic-Aware Pre-training and Adaptive-Regularized Fine-Tuning)**, a novel framework designed to:

1. **Arithmetic-Aware Pre-Training**: Capture intricate sample-wise relationships via a pretext objective based on continuous labels.
2. **Adaptive-Regularized Fine-Tuning**: Utilize a consistency-based regularization approach that self-learns appropriate data augmentation.

Our extensive experiments on 10 datasets demonstrate that **APAR** achieves **9.43%–20.37% improvement in RMSE** compared to leading GBDT, supervised NN, and pretrain-finetune NN models. We also validate the benefits of our pre-training tasks, including arithmetic operation studies.

---

## Repository Structure

```plaintext
/src/
├── config/                  # Configuration scripts for various datasets
│   ├── am/                 
│   ├── bd/
│   ├── bs/
│   └── ...                 
├── rtdl/                    # Official source code for FT-Transformer
├── model.py                 # APAR model architecture
├── self_datautils.py        # Utilities for dataset preprocessing (pre-training)
├── self_pretrain.py         # Pre-training with arithmetic-aware tasks
├── self_pretrain_vime.py    # Pre-training with feature construction or mask prediction
├── semi_datautils.py        # Utilities for dataset preprocessing (fine-tuning)
├── semi_finetune.py         # Fine-tuning with adaptive regularization
├── datautils.py             # Utilities for dataset preprocessing (testing)
├── test.py                  # Model evaluation script
└── test.sh                  # Script for running test.py
```

---

## Getting Started

### Requirements

- Python 3.x
- Required libraries: Install dependencies via `pip install -r requirements.txt`.

### Pre-training

1. Configure dataset-specific settings in `/src/config/`.
2. Run the pre-training script:
   ```bash
   python self_pretrain.py
   ```

### Fine-tuning

1. Configure dataset-specific settings in `/src/config/`.
2. Run the fine-tuning script:
   ```bash
   python semi_finetune.py
   ```

### Testing

To evaluate the trained model:
```bash
bash test.sh
```

---

## Citation

If you use this code in your research, please cite our paper:
```
@article{wu2024apar,
  title={APAR: Modeling Irregular Target Functions in Tabular Regression via Arithmetic-Aware Pre-Training and Adaptive-Regularized Fine-Tuning},
  author={Wu, Hong-Wei and Wang, Wei-Yao and Wang, Kuang-Da and Peng, Wen-Chih},
  journal={arXiv preprint arXiv:2412.10941},
  year={2024}
}
```
