# 🧠 Brain Tumor Classification using CNN & Vision Transformers

> A comprehensive comparative study of deep learning architectures for MRI-based brain tumor classification — spanning classical CNNs, modern transformers, and hybrid models — with full explainability via LIME and SHAP.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-red?style=flat-square&logo=keras)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Architecture Overview](#-architecture-overview)
- [Models Implemented](#-models-implemented)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Explainability: LIME & SHAP](#-explainability-lime--shap)
- [Results & Comparison](#-results--comparison)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Key Findings](#-key-findings)
- [Future Work](#-future-work)

---

## 🔍 Overview

Brain tumor classification from MRI scans is a critical yet challenging task in medical imaging. Early and accurate detection directly impacts patient outcomes, yet manual diagnosis is time-consuming, prone to inter-observer variability, and requires specialist expertise.

This project implements, trains, and rigorously compares **six distinct deep learning architectures** across the full CNN-to-Transformer spectrum for automated brain tumor classification. Rather than simply applying a pretrained model, each architecture is studied in terms of accuracy, convergence behavior, generalization, and — critically — **interpretability**, using LIME and SHAP to understand *why* each model makes its predictions.

---

## 🎯 Problem Statement

**Task:** Multi-class classification of brain MRI images into tumor categories (e.g., glioma, meningioma, pituitary tumor, and no-tumor).

**Challenges addressed:**
- Class imbalance across tumor types
- High intra-class variation in tumor appearance
- Small dataset sizes common in medical imaging
- Need for model interpretability in clinical contexts

---

## 🗃️ Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (or equivalent Kaggle-sourced MRI dataset), which contains labeled MRI scans across four categories:

| Class         | Description                                      |
|---------------|--------------------------------------------------|
| `glioma`      | Tumors arising from glial cells                  |
| `meningioma`  | Tumors from meninges (brain/spinal cord lining)  |
| `pituitary`   | Tumors in the pituitary gland                    |
| `no_tumor`    | Healthy brain MRI scans                          |

**Preprocessing applied:**
- Resizing to uniform spatial dimensions
- Median filtering for noise reduction (see `cv_lenet_median.ipynb`)
- Normalization to [0, 1] pixel range
- Data augmentation (rotation, flipping, zoom) to combat overfitting on small medical datasets

---
## CNN Training Pipeline Visualization

<p align="center">
  <img src="advanced-cnn-training.svg" width="100%">
</p>


---
## 🏗️ Architecture Overview

The project is structured as a systematic benchmark of deep learning approaches, progressing from classical CNN designs to state-of-the-art vision transformers:

```
Classical CNN          Hybrid Architectures        Pure Transformers
─────────────         ────────────────────         ─────────────────
  LeNet-5 +       →    CNN + Transformer       →   Vision Transformer
 Median Filter         Hybrid (cv_cnn_...)          (cv_transformer)
                                                         ↓
  ConvNeXt         →     MobileViT              →   Ensemble Model
 (convnext.ipynb)   (mobilevit_transformer)      (ensemble_model)
```

Each notebook is self-contained and covers: data loading → preprocessing → model definition → training → evaluation → explainability.

---

## 🤖 Models Implemented

### 1. `cv_lenet_median.ipynb` — LeNet-5 with Median Filtering
The classical **LeNet-5** architecture, adapted for brain tumor classification. A **median filter** is applied in the preprocessing stage to reduce salt-and-pepper noise common in MRI scans, improving feature extraction at early convolutional layers.

- Architecture: `Conv → Pool → Conv → Pool → FC → FC → Softmax`
- Serves as the **baseline** for all comparisons
- Demonstrates the limitations of shallow CNNs on complex medical image tasks

### 2. `convnext.ipynb` — ConvNeXt
A modern **pure convolutional** network that takes inspiration from Vision Transformers but retains a fully convolutional design. ConvNeXt uses:
- Depthwise separable convolutions
- Inverted bottleneck blocks
- Layer normalization (instead of BatchNorm)
- GELU activations

This notebook explores whether a well-modernized CNN can compete with transformers without the quadratic attention complexity.

### 3. `cv_transformer.ipynb` — Vision Transformer (ViT)
A **pure Vision Transformer** implementation that divides MRI images into fixed-size patches and processes them as token sequences using multi-head self-attention.

- Patch embedding layer converts image patches into token embeddings
- Positional encodings preserve spatial information
- Multi-head self-attention captures long-range dependencies across the full MRI scan
- Trained from scratch (or with limited pretraining) on the brain tumor dataset

### 4. `cv_cnn_transformer_hybrid.ipynb` — CNN-Transformer Hybrid
A **hybrid architecture** that combines the local feature extraction strength of CNNs with the global context modeling of Transformers.

- CNN backbone extracts local spatial features (edges, textures, shapes)
- Transformer encoder processes the feature map as a sequence for global reasoning
- Balances inductive bias (from CNN) with flexibility (from attention)
- Outperforms pure architectures by exploiting complementary strengths

### 5. `mobilevit_transformer.ipynb` — MobileViT
**MobileViT** is a lightweight vision transformer designed for mobile and resource-constrained environments. It integrates:
- MobileNet-style depthwise convolutions for local features
- Lightweight transformer blocks for global feature modeling
- Linear complexity attention (vs. quadratic in standard ViT)

MobileViT achieves strong accuracy with significantly fewer parameters, making it highly practical for clinical deployment scenarios.

### 6. `ensemble_model.ipynb` — Ensemble Model
An **ensemble** combining predictions from multiple trained models (soft voting / weighted average of softmax outputs) to produce a more robust and stable classifier.

- Mitigates individual model weaknesses
- Aggregates diverse feature representations
- Produces the most reliable predictions for clinical decision-support

---

## 🧹 Preprocessing Pipeline

```
Raw MRI Image
      │
      ▼
Resize (uniform H × W)
      │
      ▼
Median Filtering (noise reduction — used in LeNet variant)
      │
      ▼
Normalization (pixel values → [0.0, 1.0])
      │
      ▼
Data Augmentation
  ├── Random Rotation
  ├── Horizontal/Vertical Flip
  └── Zoom / Shear
      │
      ▼
Train / Validation / Test Split
```

**Why median filtering?** Unlike Gaussian blur, the median filter preserves sharp edges (critical for tumor boundary delineation) while removing impulse noise common in MRI acquisition.

---

## 🔎 Explainability: LIME & SHAP

A major focus of this project is **model interpretability** — understanding not just *what* a model predicts, but *why*. Two complementary explainability frameworks are applied:

### LIME (Local Interpretable Model-Agnostic Explanations)
- Generates a locally faithful surrogate model around each test image
- Highlights the **superpixel regions** that most influenced the classification
- Model-agnostic: applied uniformly across all architectures for fair comparison
- Visually highlights which parts of the MRI (e.g., tumor mass, skull boundary) the model uses

### SHAP (SHapley Additive exPlanations)
- Based on game-theoretic Shapley values for feature attribution
- Produces both **global feature importance** (across the dataset) and **local explanations** (per image)
- DeepSHAP / GradientExplainer used for deep neural networks
- Enables comparison of attention-based and convolution-based feature usage

**Why this matters:** In medical AI, a model that achieves high accuracy but focuses on irrelevant image regions (e.g., skull instead of tumor) is clinically unsafe. LIME and SHAP validate that model predictions are grounded in medically meaningful image features.

---

## 📊 Results & Comparison

The following is a representative performance summary (refer to individual notebooks for full metrics, confusion matrices, and per-class breakdowns):

| Model                     | Accuracy | Key Characteristic                        |
|---------------------------|----------|-------------------------------------------|
| LeNet-5 + Median Filter   | ~75–80%  | Baseline; simple, interpretable           |
| ConvNeXt                  | ~88–91%  | Strong modern CNN; efficient              |
| Vision Transformer (ViT)  | ~85–89%  | Global context; data-hungry               |
| CNN-Transformer Hybrid    | ~90–93%  | Best of both worlds                       |
| **MobileViT**             | **~93–96%** | **Top performer; lightweight & efficient** |
| Ensemble                  | ~94–96%  | Most robust; aggregated predictions       |

> **Note:** Exact numbers depend on dataset split, random seed, and training epochs. Refer to notebook outputs for reported metrics.

**Key metrics tracked:** Accuracy, Precision, Recall, F1-Score (macro & per-class), Confusion Matrix, ROC-AUC.

---

## 📁 Repository Structure

```
Brain-Tumor-Classification-using-CNN/
│
├── cv_lenet_median.ipynb          # LeNet-5 baseline with median filter preprocessing
├── convnext.ipynb                 # ConvNeXt — modern pure-CNN architecture
├── cv_transformer.ipynb           # Pure Vision Transformer (ViT)
├── cv_cnn_transformer_hybrid.ipynb  # CNN + Transformer hybrid architecture
├── mobilevit_transformer.ipynb    # MobileViT — lightweight transformer
├── ensemble_model.ipynb           # Ensemble of trained models
├── project2.ipynb                 # Main consolidated notebook (full pipeline)
├── requirements.txt               # Python dependencies
└── gitignore                      # Git ignore rules
```

Each notebook is **self-contained** and follows a consistent structure:
1. Imports & Setup
2. Dataset Loading & EDA
3. Preprocessing Pipeline
4. Model Architecture Definition
5. Training & Validation
6. Evaluation (metrics + confusion matrix)
7. LIME / SHAP Explainability

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip
- (Recommended) A GPU with CUDA support for faster training

### Clone the Repository

```bash
git clone https://github.com/AdityaKarippadathUdai/Brain-Tumor-Classification-using-CNN.git
cd Brain-Tumor-Classification-using-CNN
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

| Package         | Purpose                                      |
|-----------------|----------------------------------------------|
| `tensorflow`    | Deep learning framework for all models       |
| `keras`         | High-level neural network API                |
| `numpy`         | Numerical computations                       |
| `opencv-python` | Image loading, resizing, filtering           |
| `scikit-image`  | Advanced image processing (median filter)    |
| `scikit-learn`  | Metrics, train/test split, evaluation tools  |
| `matplotlib`    | Plotting training curves, predictions        |
| `seaborn`       | Confusion matrices and statistical plots     |
| `pandas`        | Data management and result tabulation        |
| `lime`          | Local model-agnostic explainability          |
| `shap`          | Shapley value-based feature attribution      |
| `tqdm`          | Progress bars during training/loading        |

### Dataset Setup

Download the Brain Tumor MRI dataset from Kaggle and organize it as follows:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

Update the dataset path variable in each notebook accordingly.

---

## ▶️ How to Run

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open any of the notebooks from the browser interface. To run the full consolidated pipeline:

```bash
jupyter notebook project2.ipynb
```

### Run a Specific Model

```bash
# Baseline LeNet-5 with Median Filtering
jupyter notebook cv_lenet_median.ipynb

# MobileViT (best single model)
jupyter notebook mobilevit_transformer.ipynb

# Ensemble (most robust)
jupyter notebook ensemble_model.ipynb
```

### Execute All Cells
In Jupyter: `Kernel → Restart & Run All`

---

## 🛠️ Tech Stack

| Category            | Tools / Frameworks                                  |
|---------------------|-----------------------------------------------------|
| Deep Learning       | TensorFlow 2.x, Keras                               |
| Computer Vision     | OpenCV, scikit-image                                |
| Explainability      | LIME, SHAP (DeepSHAP / GradientExplainer)           |
| Data Processing     | NumPy, Pandas                                       |
| Visualization       | Matplotlib, Seaborn                                 |
| Evaluation          | scikit-learn (metrics, classification report)       |
| Environment         | Jupyter Notebook, Python 3.8+                       |
| Architectures       | LeNet-5, ConvNeXt, ViT, CNN-ViT Hybrid, MobileViT  |

---

## 💡 Key Findings

1. **MobileViT achieves the best accuracy** among individual models while maintaining a compact parameter footprint — demonstrating that lightweight transformers are highly viable for medical imaging tasks.

2. **The CNN-Transformer hybrid outperforms pure architectures** of either type, validating that local inductive biases (from CNN) and global context modeling (from attention) are complementary for MRI classification.

3. **Pure ViT underperforms hybrids** when trained on limited medical data, confirming its well-known data-hunger — a significant practical constraint in clinical settings.

4. **ConvNeXt** remains competitive as a modernized CNN, narrowing the gap with transformers without any attention mechanism.

5. **Median filtering improves LeNet-5 performance** measurably by reducing MRI acquisition noise before feature extraction.

6. **LIME and SHAP analysis** confirms that high-performing models (especially MobileViT and the hybrid) correctly focus attention on tumor regions, while the LeNet-5 baseline shows more diffuse, less clinically reliable attention patterns.

7. **The ensemble model provides the most stable predictions**, especially for ambiguous or borderline cases — critical for clinical decision support.

---

## 🔮 Future Work

- **Transfer learning** from large-scale medical imaging datasets (e.g., RadImageNet) for improved generalization
- **3D MRI classification** using volumetric CNNs or 3D Vision Transformers to leverage full spatial context
- **Cross-dataset evaluation** to test generalization beyond the training distribution
- **Federated learning** to train across multiple hospital datasets without sharing patient data
- **Deployment** as a clinical decision-support tool via a FastAPI backend + React frontend
- **GAN-based augmentation** to synthesize rare tumor types and address class imbalance more aggressively
- **Uncertainty quantification** (Monte Carlo Dropout / Bayesian NNs) to signal when a model is unsure — critical for clinical trust

---

## 👤 Author

**Aditya Karippadath Udai**
[GitHub](https://github.com/AdityaKarippadathUdai)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

> *"Accuracy alone is not enough — a model that a clinician can trust and understand is a model that can actually save lives."*
