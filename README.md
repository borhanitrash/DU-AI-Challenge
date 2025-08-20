# 3rd Place Solution - DU AI Challenge 2025 🥉
## Overview
The DU AI Challenge: DU Arena Season 1 is an intensive 8-hour computer vision competition where participants develop AI models to detect and classify traffic objects in drone-captured aerial images from Bangladesh. Teams will compete to achieve the highest mean Average Precision (mAP) on a dataset featuring 11 object classes including Bangladesh-specific vehicles like CNG auto-rickshaws, legunas, and cycle rickshaws.

## Dataset Overview      
Training Set: 174 high-resolution drone images with YOLO format annotations     
Test Set: 57 images requiring object detection predictions     
Classes: 11 traffic participant categories including cars, buses, trucks, motorcycles, pedestrians, and Bangladesh-specific vehicles (CNG, leguna, rickshaw, manual-van)     
Image Resolution: Primarily 4K (3840×2160) aerial captures     
Average Objects per Image: 33 (dense traffic scenes) 

## Our Solution
**Final Score: 0.62898 mAP (Private Leaderboard)**

## Solution Overview

Our 3rd place solution implements an ensemble of three YOLO models (YOLOv8L, YOLOv8M, YOLOv5L) with enhanced CLAHE preprocessing and custom NMS fusion specifically optimized for aerial traffic detection in Bangladesh.

## Technical Architecture

### Preprocessing Pipeline
We implemented CLAHE (Contrast Limited Adaptive Histogram Equalization) with clip limit 3.0 and 8×8 tile grid in LAB color space. This addresses the primary challenge of aerial imagery where lighting conditions vary across frames and small vehicles like CNG auto-rickshaws suffer from poor contrast.

```python
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)
```

The preprocessing converts RGB→LAB, applies CLAHE to L-channel, adds 10% sharpening blend, and converts back to RGB. This provided +2.3 mAP improvement on validation over raw images.

### Model Configuration
Training used 1280px resolution to preserve fine details crucial for small object detection from aerial perspectives. All models trained for 100 epochs with early stopping patience of 30.

**Optimization Setup:**
- AdamW optimizer with cosine learning rate schedule (0.001→0.0001)
- Batch size 4, mixed precision training enabled
- Loss weights: Box=0.05, Classification=0.5, DFL=1.5

**Augmentation Strategy:**
Conservative rotation (10°) and translation (0.1) to simulate drone variations while maintaining aerial perspective integrity. HSV augmentation (h=0.015, s=0.7, v=0.4) handles Bangladesh weather variations. Mosaic probability 1.0 for dense traffic learning, disabled final 20 epochs for individual image fine-tuning.

### Ensemble Architecture
Each model specializes in different detection aspects:
- **YOLOv8L**: Superior small object detection (pedestrians, motorcycles)  
- **YOLOv8M**: Balanced accuracy-speed with strong generalization
- **YOLOv5L**: Architectural diversity with proven aerial performance

### Custom Ensemble NMS
Standard NMS fails in dense traffic scenarios. Our algorithm:
1. Collects predictions from all models with confidence threshold 0.1
2. Applies individual model NMS at IoU 0.7  
3. Groups predictions by class
4. Applies ensemble NMS at IoU 0.6 (less aggressive than individual)
5. Preserves complementary detections while suppressing redundancy

```python
CONFIDENCE_THRESHOLD = 0.1
MODEL_IOU_THRESH = 0.7
ENSEMBLE_IOU_THRESH = 0.6
```

This fusion strategy provided +1.8 mAP over the best individual model.

## Implementation Details

### Data Split Strategy
Used 75%-25% train-validation split (130-44 images) with fixed random seed 42 for reproducibility. The limited dataset size required careful validation to prevent overfitting while maintaining model diversity.

### Training Efficiency
Parallel model training within 8-hour constraint using mixed precision for 1.5x speedup. Training timeline:
- Hours 1-2: Data preprocessing and pipeline setup
- Hours 2-6: Parallel YOLOv8L/M and YOLOv5L training
- Hours 6-8: Ensemble implementation and final inference

### Loss Function Tuning
Reduced box regression gain (0.05) accounts for inherent localization uncertainty in aerial imagery. Higher DFL gain (1.5) better handles dense overlapping scenarios typical in Bangladesh traffic.

## Results Analysis

**Validation Performance:**
- CLAHE preprocessing: +2.3 mAP improvement
- Ensemble fusion: +1.8 mAP over best single model
- High resolution training: Critical for small object detection

**Final Submission Statistics:**
- Test images with detections: 52/57 (91.2%)
- Average detections per image: ~28 objects
- Processing time: 3.2 minutes for full test set
- **Private leaderboard: 0.62898 mAP**

The ensemble approach proved particularly effective for Bangladesh-specific vehicles (CNG, leguna, rickshaw) that individual models struggled with due to their unique appearance and scale variations.

## Key Technical Insights

**CLAHE Preprocessing:** Essential for aerial imagery where standard enhancement fails. LAB color space processing preserves color while enhancing luminance details critical for small vehicle detection.

**Conservative Ensemble NMS:** Aggressive suppression eliminates valuable detections in dense traffic. Our class-wise approach with relaxed IoU threshold (0.6) maintains difficult detections while preserving precision.

**High Resolution Training:** 1280px input size crucial for detecting small objects like motorcycles and pedestrians from aerial perspectives. Standard 640px resolution insufficient for this use case.

**Domain-Specific Augmentation:** Avoiding vertical flips maintains traffic orientation integrity while horizontal flips simulate bidirectional flow. Mosaic augmentation excellent for dense scenario learning.

## Architecture Limitations

Time constraints prevented test-time augmentation (TTA) implementation, which typically provides +0.5-1.0 mAP improvement. Weighted Box Fusion (WBF) could potentially outperform our custom NMS but requires more tuning time than available in competition setting.
