# LX-DFD: Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning

Reproducible research codebase for LX-DFD deepfake detection system investigating cross-domain generalization, frequency-domain learning (2D DCT), and transformation robustness.

---

## 📑 Research Overview

- **Base Paper**: Patel et al., *"An Improved Dense CNN Architecture for Deepfake Image Detection"*, IEEE Access, 2023. DOI: `10.1109/ACCESS.2023.3251417`.
- **Research Gap**: *"Existing CNN-based deepfake detectors achieve high accuracy when trained and tested on known GAN-generated face datasets, but their ability to generalize to unseen generators, datasets, image transformations, and real-world conditions remains insufficiently investigated."*

---

## 🛠️ Quick Start & Execution Commands

### 1. Dataset Audit (Experiment 0)
```bash
python scripts/audit_dataset.py
```

### 2. Leak-Free Data Splitting (70% Train / 15% Val / 15% Test)
```bash
python scripts/create_splits.py
```

### 3. Train Baselines (Simple CNN, Dense CNN, EfficientNet)
```bash
python scripts/train_baseline.py --model simple_cnn
python scripts/train_baseline.py --model dense_cnn
python scripts/train_baseline.py --model efficientnet
```

### 4. Train Proposed LX-DFD Spatial-Frequency Architecture
```bash
python scripts/train_proposed.py --fusion attention --aug-mode standard
```

### 5. Run Robustness Benchmark
```bash
python scripts/robustness_test.py --checkpoint checkpoints/lxfd_attention_standard_best.pt --model-type lxfd
```

### 6. Run FastAPI Backend API & Dashboard
```bash
uvicorn app.backend.main:app --reload --port 8000
```
Open `app/frontend/index.html` in your browser to view the 10-page AI laboratory research dashboard!
