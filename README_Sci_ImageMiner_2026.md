
# Scientific Figure Panel Classification – Sci-ImageMiner 2026

## Overview
This repository contains the implementation of our solution for **Sci-ImageMiner 2026 Task 1**, part of the **ICDAR 2026 Conference**.  
The objective of this task is to classify individual scientific figure panels extracted from scientific literature into one of **49 predefined figure categories**.

Our approach combines **ConvNeXt-Base** and **Swin Transformer-Base** architectures using an **ensemble learning strategy** to capture both local visual features and global contextual relationships.

---

## Authors

- **K. Hari Prasad (2301AI05)**
- **Vivek Katuri (2301AI36)**
  
Department of Computer Science and Engineering  
Indian Institute of Technology Patna

---

## Problem Statement

Scientific figures contain complex visual information such as:

- Spectra charts  
- Line graphs  
- Phase diagrams  
- Experimental visualizations  

The goal is to:

**Classify each figure panel into one of 49 predefined classes using deep learning models.**

---

## Methodology

Our pipeline consists of three major stages:

### 1. Panel Extraction
- Use bounding box annotations
- Crop individual panels
- Validate bounding box coordinates
- Handle missing or invalid annotations

### 2. Preprocessing
- Square padding
- Resize images to **384 × 384**
- Apply data augmentation:
  - Random rotation
  - Color jitter
- Normalize images

### 3. Model Training
Two complementary backbone models are trained:

#### ConvNeXt-Base
- Captures fine-grained local textures
- Strong at detecting edges and structures

#### Swin Transformer-Base
- Captures global relationships
- Models long-range dependencies

### 4. Ensemble Inference
Final predictions are generated using weighted averaging of both model outputs.

---

## Performance Results

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| ConvNeXt-Base | 0.64 | 0.66 | 0.63 | 0.64 |
| Swin Transformer | 0.65 | 0.68 | 0.64 | 0.66 |
| **Ensemble Model** | **0.67** | **0.70** | **0.67** | **0.68** |

**Final Micro F1 Score: 0.68**

---

## Dataset

Dataset Used:

**ALD-E Scientific Figure Dataset**

Available at:
https://github.com/sciknoworg/ALD-E-ImageMiner

---

## Installation

Clone the repository:

```bash
git clone https://github.com/HARI-45-FAV/capstone_2301AI05.git
cd capstone_2301AI05
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Typical dependencies include:

- Python >= 3.8  
- PyTorch  
- Torchvision  
- TIMM  
- NumPy  
- OpenCV  
- Scikit-learn  
- Matplotlib  
- Pandas  

---

## Training

Train ConvNeXt:

```bash
python train_convnext.py
```

Train Swin Transformer:

```bash
python train_swin.py
```

---

## Inference

Run ensemble inference:

```bash
python ensemble_inference.py
```

Output:

- Predicted class label  
- Confidence score  

---

## Project Structure

```
project_root/
│
├── data/
├── preprocessing/
├── models/
├── training/
├── inference/
├── utils/
│
├── requirements.txt
├── README.md
└── submission_script.py
```

---

## Training Configuration

| Parameter | Value |
|----------|------|
| Input Size | 384 × 384 |
| Optimizer | AdamW |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Batch Size | 8 |
| Epochs | 12 |
| Scheduler | Cosine Annealing |
| Precision | Mixed Precision |

---

## Challenges

Key challenges encountered:

- High similarity between classes  
- Variability in layouts  
- Class imbalance  
- Domain-specific visual complexity  

---

## Future Work

Potential improvements:

- Test-time augmentation  
- Advanced augmentation strategies  
- Domain-specific pretraining  
- Vision-language integration  

---

## License

This project is licensed under:

**Creative Commons Attribution 4.0 International License**

---

## Acknowledgements

We thank:

- Sci-ImageMiner 2026 Organizing Committee  
- ICDAR 2026  
- IIT Patna for providing computational resources  

---

## Citation

If you use this work, please cite:

K. Hari Prasad, Vivek Katuri,  
Scientific Figure Panel Classification,  
Sci-ImageMiner 2026, ICDAR 2026.
