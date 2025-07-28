# STARK+SAM Tracker for the VOT Toolkit

A hybrid visual object tracker that combines STARK's transformer-based bounding box regression with the Segment Anything Model (SAM) for mask-level precision. This tracker is designed to plug directly into the [VOT Toolkit](https://github.com/votchallenge/vot-toolkit) for standardized evaluation.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Project Setup](#-project-setup)
- [Running Experiments](#-running-experiments)
- [Tracker Configuration (trackersini)](#-tracker-configuration-trackersini)
- [Command-Line Arguments](#-command-line-arguments)
- [Troubleshooting](#-troubleshooting)


- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Running Experiments](#running-experiments)
- [Tracker Configuration (trackers.ini)](#tracker-configuration-trackersini)
- [Command-Line Arguments](#command-line-arguments)
- [Troubleshooting](#troubleshooting)

---

## 🧠 Overview

This tracker executes a **two-stage hybrid pipeline** per frame:

1. **Bounding Box Prediction**  
   Uses a high-performance STARK model (`stark_s`, `stark_st`, etc.) for bounding box regression.

2. **Mask Generation**  
   Passes the predicted box as a prompt to Meta AI’s **Segment Anything Model (SAM)** to refine object boundaries.

Designed for seamless integration with the **VOT Toolkit** (TraX protocol), the tracker supports flexible configuration through both command-line arguments and `.ini` files.

---

## 🚀 Features

- 🔄 **Hybrid Tracking**  
  Combines STARK's transformer tracking with SAM's segmentation accuracy.

- ⚙️ **Fully Configurable**  
  Easily switch between STARK/SAM models, VOT datasets, and hardware setups.

- 🎛 **Confidence Filtering**  
  Set thresholds to reject masks with low quality scores.

- 🎯 **Multi-Mask Support**  
  Generate and auto-select the best of multiple SAM predictions.

- 🧪 **VOT Toolkit Integration**  
  Designed for direct compatibility with `vot-toolkit` evaluation workflows.

---

## 📋 Prerequisites

- Linux OS
- Python 3.8+
- NVIDIA GPU + CUDA
- Conda or Miniconda

---

## ⚙️ Project Setup

### 🔹 Step 1: Recommended Directory Structure

```
/your/projects/
├── Stark/                   # Official STARK codebase
├── my_vot_project/          # Your tracker scripts
│   ├── vot_stark_sam_tracker.py
│   ├── vot_data_preprocessing.py
│   ├── vot.py
│   ├── checkpoints/
│   │   ├── sam_vit_b_01ec64.pth
│   │   └── sam_vit_h_4b8939.pth
```

### 🔹 Step 2: Configure STARK Path

Edit `vot_stark_sam_tracker.py`:

```python
# Example (edit this to your actual path)
STARK_PROJECT_PATH = "/absolute/path/to/Stark/"
```

### 🔹 Step 3: Conda Environment

```bash
conda create -n stark_sam python=3.8 -y
conda activate stark_sam
```

### 🔹 Step 4: Install Dependencies

```bash
# Install PyTorch (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies
pip install opencv-python vot-toolkit segment-anything
```

### 🔹 Step 5: Download SAM Checkpoints

```bash
mkdir -p checkpoints/
# ViT-B
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
# ViT-H
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

---

## 🧪 Running Experiments

### 1. Configure `trackers.ini`

Place this file inside your VOT workspace. Example:

```ini
# STARK-S + SAM ViT-B
[stark_s_sam_b_vot20]
label = STARK-S_SAM-B
protocol = trax
command = /home/user/.conda/envs/stark_sam/bin/python /path/to/my_vot_project/vot_stark_sam_tracker.py --tracker_name stark_s --sam_checkpoint /path/to/checkpoints/sam_vit_b_01ec64.pth --sam_model_type vit_b

# STARK-ST + SAM ViT-H with multimask and GPU 1
[stark_st_sam_h_multi_gpu1]
label = STARK-ST_SAM-H_Multi_GPU1
protocol = trax
command = /home/user/.conda/envs/stark_sam/bin/python /path/to/my_vot_project/vot_stark_sam_tracker.py --tracker_name stark_st --sam_checkpoint /path/to/checkpoints/sam_vit_h_4b8939.pth --sam_model_type vit_h --multimask_output --gpu_id 1
```

### 2. Run the Evaluation

```bash
conda activate stark_sam
vot-evaluate --workspace /absolute/path/to/vot_workspace stark_s_sam_b_vot20
```

---

## 🛠️ Command-Line Arguments

| Argument            | Description                                                                 | Default   |
|---------------------|-----------------------------------------------------------------------------|-----------|
| `--tracker_name`    | STARK model name (`stark_s`, `stark_st`, etc.)                              | _Required_ |
| `--tracker_param`   | Tracker parameter variant (`baseline`, etc.)                                 | `baseline` |
| `--sam_checkpoint`  | Path to SAM checkpoint `.pth`                                                | _Required_ |
| `--sam_model_type`  | SAM model type (`vit_b`, `vit_l`, `vit_h`)                                   | `vit_b`   |
| `--vot_version`     | Dataset version (e.g., `vot20`, `vot20lt`)                                   | `vot20`   |
| `--multimask_output`| If set, outputs multiple masks and picks the best                            | `False`   |
| `--gpu_id`          | GPU index to use                                                             | `3`       |
| `--confidence`      | IOU threshold for mask filtering                                             | `0.6`     |

---

## 🧯 Troubleshooting

### ❗ `Unable to connect to tracker`

- Ensure paths in `trackers.ini` are **absolute**
- Run the command manually to debug
- Verify the `python` path points to your `stark_sam` conda env

---

### ❗ `RuntimeError: Error(s) in loading state_dict for Sam`

- Mismatch between `--sam_model_type` and checkpoint
- Make sure `vit_h` → `sam_vit_h_*.pth`, and so on

