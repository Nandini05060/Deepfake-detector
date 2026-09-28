# LX-DFD: Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Reproducible Research Codebase and Interactive Web Platform for Dual-Branch Spatial-Frequency Deepfake Detection with Cross-Domain Generalization and Explainable AI (XAI).**

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Base Paper & Research Gap](#-base-paper--research-gap)
- [Key AI & Deep Learning Concepts](#-key-ai--deep-learning-concepts)
- [Architecture Anatomy](#-architecture-anatomy)
- [Benchmark Results](#-benchmark-results)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Web Application & Live Dashboard](#-web-application--live-dashboard)
- [Execution Pipeline](#-execution-pipeline)
- [Explainability & Visualizations (XAI)](#-explainability--visualizations-xai)

---

## 🔍 Project Overview

Deepfake generation technologies driven by Generative Adversarial Networks (GANs) and diffusion models have reached near-photorealistic quality. Conventional deepfake detectors rely solely on spatial (pixel-level) RGB features, making them prone to failure when exposed to:
1. **Unseen image generators and novel deepfake synthesis algorithms** (Cross-Domain Shift).
2. **Real-world transmission corruptions** (JPEG compression, Gaussian noise, motion blur).

**LX-DFD** resolves these limitations by introducing a **Dual-Branch Spatial-Frequency Learning Architecture**:
- **Branch A (Spatial RGB Domain)**: Captures semantic facial cues, lighting variations, and blending boundaries.
- **Branch B (Frequency Domain)**: Transforms images via **2D Discrete Cosine Transform (2D DCT)** to expose invisible GAN upsampling grid artifacts and high-frequency spectral discrepancies.
- **Cross-Modal Attention Fusion**: Dynamically computes learned weights ($w_s, w_f$) to fuse evidence from both domains.

---

## 📄 Base Paper & Research Gap

- **Base Paper**: Patel et al., *"An Improved Dense CNN Architecture for Deepfake Image Detection"*, *IEEE Access*, Volume 11, 2023, pp. 22081–22095.  
  - DOI: [`10.1109/ACCESS.2023.3251417`](https://doi.org/10.1109/ACCESS.2023.3251417)
  - IEEE Xplore: [https://ieeexplore.ieee.org/document/10057390/](https://ieeexplore.ieee.org/document/10057390/)
- **Research Gap**:
  > *"Existing CNN-based deepfake detectors achieve high accuracy when trained and tested on known GAN-generated face datasets, but their ability to generalize to unseen generators, datasets, image transformations, and real-world conditions remains insufficiently investigated."*

---

## 🧠 Key AI & Deep Learning Concepts

| AI Concept | Description & Implementation |
|---|---|
| **Dual-Stream CNN** | Two parallel neural networks extracting spatial visual features and frequency spectrum features simultaneously. |
| **2D Discrete Cosine Transform (DCT)** | Converts spatial pixels into frequency domain components to capture GAN upsampling checkerboard artifacts. |
| **Cross-Modal Attention Fusion** | Self-attention mechanism that dynamically balances the influence of spatial vs. frequency embeddings. |
| **Transfer Learning** | Pretrained **EfficientNet** backbone fine-tuned for high-level facial artifact representations. |
| **Dense Connectivity** | Re-implementation of DenseNet-style feature reuse layers (Patel et al. 2023 baseline). |
| **Explainable AI (Grad-CAM)** | Gradient-weighted Class Activation Mapping to generate interpretable visual decision heatmaps. |
| **Domain Generalization** | Cross-domain and unseen generator testing to evaluate out-of-distribution robustness. |
| **Perturbation Robustness** | Stress-testing models against JPEG compression, Gaussian blur, additive noise, and brightness shifts. |

---

## 🏗️ Architecture Anatomy

```
                              ┌────────────────────────────────────────┐
                              │          Input Face Image (RGB)        │
                              └───────────────────┬────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │       Spatial Branch (RGB)       │             │     Frequency Branch (2D DCT)    │
        ├──────────────────────────────────┤             ├──────────────────────────────────┤
        │  EfficientNet-B0 Backbone        │             │  2D DCT Log-Magnitude Spectrum   │
        │  Adaptive Average Pooling        │             │  4-Stage Spectral ConvNet + Norm │
        │  Linear Projection (256-D)       │             │  Linear Projection (256-D)       │
        └────────────────┬─────────────────┘             └────────────────┬─────────────────┘
                         │ Embedding F_s                                  │ Embedding F_f
                         └────────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │  Cross-Modal Attention Fusion   │
                                 │     F_fused = w_s*F_s + w_f*F_f │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │       Binary MLP Classifier     │
                                 │    Linear(256->128) -> GELU     │
                                 │    Dropout(0.3) -> Linear(1)    │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                   REAL [0]  /  FAKE [1] Output
```

---

## 📊 Benchmark Results

Evaluated on **12,890 face images** with leak-free train/val/test splits (70% / 15% / 15%):

| Model Architecture | In-Domain ROC-AUC | Cross-Domain ROC-AUC | F1-Score | Accuracy | Robustness Score |
|---|:---:|:---:|:---:|:---:|:---:|
| Simple CNN Baseline | 0.892 | 0.710 | 0.842 | 84.2% | 0.680 |
| Dense CNN (Patel et al., 2023) | 0.965 | 0.785 | 0.925 | 92.5% | 0.752 |
| EfficientNet Baseline | 0.978 | 0.812 | 0.941 | 94.1% | 0.795 |
| LX-DFD (Spatial-Only) | 0.975 | 0.805 | 0.938 | 93.8% | 0.788 |
| LX-DFD (Frequency-Only) | 0.915 | 0.842 | 0.865 | 86.5% | 0.835 |
| LX-DFD (Simple Concat) | 0.981 | 0.865 | 0.952 | 95.2% | 0.848 |
| LX-DFD (Weighted Fusion) | 0.984 | 0.879 | 0.958 | 95.8% | 0.862 |
| **LX-DFD (Attention Fusion)** | **0.988** | **0.895** | **0.964** | **96.4%** | **0.884** |

---

## 📁 Project Directory Structure

```plaintext
DEEPFAKE DETECTION/
├── app/
│   ├── backend/
│   │   └── main.py              # FastAPI application & inference endpoints
│   └── frontend/
│       ├── index.html           # 10-page AI laboratory research dashboard
│       ├── styles.css           # Premium dark-mode modern laboratory theme
│       └── app.js               # Interactive frontend controller & charts
├── checkpoints/                 # Trained model weights (.pt) and history logs (.json)
│   ├── dense_cnn_best.pt
│   ├── efficientnet_best.pt
│   ├── lxfd_attention_standard_best.pt
│   └── simple_cnn_best.pt
├── configs/
│   └── config.yaml              # Hyperparameters, augmentation & training configs
├── Final Dataset/               # Real and Fake face dataset
├── reports/                     # Audit, evaluation & robustness reports
│   ├── final_results.md
│   ├── model_architecture_anatomy.md
│   └── LX_DFD_Complete_Research_Guide.md
├── scripts/                     # Executable scripts for audit, training, testing
│   ├── audit_dataset.py
│   ├── create_splits.py
│   ├── evaluate.py
│   ├── print_model_summary.py
│   ├── robustness_test.py
│   ├── train_baseline.py
│   └── train_proposed.py
├── src/
│   ├── data/                    # Dataset loaders, augmentations, and splitters
│   ├── evaluation/              # Metrics, cross-domain, robustness & error analysis
│   ├── explainability/          # Grad-CAM heatmap generator
│   ├── frequency/               # 2D DCT transformations & spectral visualizations
│   ├── models/                  # PyTorch model definitions (LX-DFD, CNNs, Fusion)
│   ├── training/                # Losses, schedulers, and training loop engine
│   └── utils/                   # Device manager, logger, and reproducibility seed
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/Nandini05060/Deepfake-detector.git
cd "DEEPFAKE DETECTION"
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌐 Web Application & Live Dashboard

Launch the FastAPI backend server and interactive web laboratory:

```bash
uvicorn app.backend.main:app --reload --port 8000
```

Once running, visit:
- **Interactive UI Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Dashboard Features:
- 📤 **Live Deepfake Scanner**: Drag-and-drop any face image for real-time inference.
- 🔬 **Grad-CAM Heatmap Viewer**: Visual explanation overlay highlighting facial artifacts.
- ⚡ **2D DCT Spectral Inspector**: Real-time frequency spectrum and high-pass residual filter.
- ⚖️ **Attention Weight Breakdown**: Visual display of $w_s$ (spatial) and $w_f$ (frequency) confidence contribution.
- 📈 **Benchmark Visualizer**: Interactive ROC curves, confusion matrices, and ablation comparisons.

---

## ⚙️ Execution Pipeline

### Step 1: Audit Dataset & Prevent Leakage
```bash
python scripts/audit_dataset.py
python scripts/create_splits.py
```

### Step 2: Train Baseline Models
```bash
# Simple CNN
python scripts/train_baseline.py --model simple_cnn

# Dense CNN (Patel et al., 2023)
python scripts/train_baseline.py --model dense_cnn

# EfficientNet Baseline
python scripts/train_baseline.py --model efficientnet
```

### Step 3: Train Proposed LX-DFD Architecture
```bash
# Proposed Attention-Fusion LX-DFD
python scripts/train_proposed.py --fusion attention --aug-mode standard
```

### Step 4: Run Robustness & Cross-Domain Benchmarking
```bash
python scripts/robustness_test.py --checkpoint checkpoints/lxfd_attention_standard_best.pt --model-type lxfd
```

### Step 5: Print Model Anatomy & Parameter Count
```bash
python scripts/print_model_summary.py
```

---

## 👁️ Explainability & Visualizations (XAI)

LX-DFD includes native Explainable AI tools:
1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Computes gradients from the final convolutional layer of the spatial branch to highlight the facial areas (e.g. eyes, mouth, blending seams) that influenced the prediction.
2. **Frequency Domain Visualizer**: Converts the input to a 2D DCT log-magnitude map and high-frequency residual colormap, revealing GAN lattice artifacts that are otherwise invisible to human observers.

---

## 📜 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
