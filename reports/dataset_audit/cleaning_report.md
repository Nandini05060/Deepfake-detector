# Deepfake Dataset Research-Grade Audit & Cleaning Report

**Dataset Directory**: `C:\Users\nandi\Desktop\DEEPFAKE DETECTION\Final Dataset`  
**Audit Execution Timestamp**: `2026-09-06`  
**Face Detection Engine**: `OpenCV YuNet (yunet.onnx)`  
**Perceptual Hashing Algorithm**: `pHash (2D-DCT 64-bit)`  

---  

## 1. Executive Summary & Core Dataset Metrics

| Metric | Value | Description / Note |
|---|---|---|
| **Original Discovered Files** | `12893` | Total files scanned across all subdirectories |
| **Discovered Images** | `12890` | Raw image count (`.jpg`) |
| **Corrupted Images** | `0` | Unreadable/corrupted files (0.00%) |
| **Valid Usable Images** | `12890` | 100% of raw images successfully decoded |
| **Exact Duplicate Files (SHA-256)** | `8` | Identical bitwise images across 4 groups |
| **Near-Duplicate Pairs (pHash d <= 6)** | `217` | Suspiciously similar images requiring grouped splitting |
| **ONE_FACE Images** | `12567` | Single face detected and cropped with 15% margin |
| **MULTIPLE_FACES Images** | `130` | Multiple faces detected (largest face selected for crop) |
| **NO_FACE Images** | `193` | No faces detected by YuNet (flagged for review list) |

---  

## 2. Class Distribution & Imbalance Analysis

| Class | Image Count | Percentage | Class Ratio (FAKE/REAL) |
|---|---|---|---|
| **REAL** | `5890` | `45.69%` | 1.000 |
| **FAKE** | `7000` | `54.31%` | `1.188` |
| **Total** | `12890` | `100.00%` | - |

> [!IMPORTANT]
> The dataset has a mild class imbalance (8.61% difference). In accordance with research standards, **no images were discarded to artificially balance the dataset**. Instead, weighted loss or weighted sampling will be applied during model training.

---  

## 3. Configurable Audit Thresholds Used

| Parameter | Threshold / Value | Rationale |
|---|---|---|
| **Small Image Resolution Cutoff** | `< 128 x 128` | Flags low-resolution images for quality review |
| **Aspect Ratio Anomaly Bounds** | `< 0.7` or `> 1.4` | Flags non-square or stretched face bounding boxes |
| **pHash Near-Duplicate Hamming Distance** | `d <= 6` | Flags images with high perceptual similarity (out of 64 bits) |
| **YuNet Detection Confidence Threshold** | `>= 0.60` | Ensures high-precision facial bounding box detection |
| **YuNet NMS Threshold** | `0.30` | Prevents duplicate bounding box predictions per face |
| **Face Crop Context Expansion Margin** | `15%` | Expands bounding box to include ears, chin, jawline, and hairline |
| **Standardized Resized Resolution** | `224 x 224 RGB` | Standard input dimensions for CNN & Transformer backbones |

---  

## 4. Leakage-Free Stratified Split Breakdown

Splits were constructed using **Group-Stratified Splitting**. Images belonging to the same base identity prefix or duplicate/near-duplicate cluster were strictly assigned to the **same split**, guaranteeing **ZERO data leakage** across train, validation, and test sets.

| Split | Ratio Target | Total Images | REAL Images | FAKE Images | Usable % |
|---|---|---|---|---|---|
| **TRAIN** | 70% | `9020` | `4120` | `4900` | `69.98%` |
| **VALIDATION** | 15% | `1933` | `885` | `1048` | `15.00%` |
| **TEST** | 15% | `1937` | `885` | `1052` | `15.03%` |
| **TOTAL** | 100% | `12890` | `5890` | `7000` | `100.00%` |

---  

## 5. Audit Deliverables Checklist

- [x] `reports/dataset_audit/image_statistics.csv` (12890 rows)
- [x] `reports/dataset_audit/corrupt_images.csv` (0 rows)
- [x] `reports/dataset_audit/exact_duplicates.csv` (8 rows)
- [x] `reports/dataset_audit/near_duplicates.csv` (217 rows)
- [x] `reports/dataset_audit/face_detection.csv` (12890 rows)
- [x] `data/metadata.csv` (12890 rows)
- [x] `data/splits/train.csv` (9020 rows)
- [x] `data/splits/val.csv` (1933 rows)
- [x] `data/splits/test.csv` (1937 rows)
- [x] Visual Contact Sheets generated under `reports/dataset_audit/figures/` (7 PNG files)
- [x] Standardized 224x224 RGB dataset created under `data/processed/` (12890 images)

