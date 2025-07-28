
# STARK+SAM Tracker for VOT Evaluation

A hybrid visual object tracker that fuses **STARK's transformer-based bounding box prediction** with **Segment Anything Model (SAM)** for pixel-accurate segmentation. This tracker is fully compatible with the [VOT Toolkit](https://github.com/votchallenge/toolkit) via the TraX protocol and is designed for easy experimentation and benchmarking.

---

## 🧠 Overview

The tracker employs a **two-stage pipeline** per frame:

1. **Bounding Box Estimation**: A STARK variant (e.g., `stark_s`, `stark_st`) predicts the object’s bounding box using transformer-based regression.
2. **Mask Refinement**: The bounding box is passed to Meta AI’s **SAM** as a prompt to generate a high-fidelity segmentation mask.

This modular design enables **fine-grained tracking**, balancing STARK’s speed with SAM’s precision. It supports **multi-GPU setups**, **multi-mask selection**, and **confidence filtering**, while seamlessly integrating with the VOT ecosystem.

---

## 🚀 Key Features

- 🔄 **Hybrid Tracking Pipeline**: Leverages the strengths of STARK for bounding box tracking and SAM for detailed object segmentation.
- ⚙️ **Fully Configurable**: Easily switch between different STARK variants, SAM backbones, datasets, and hardware settings.
- 🧪 **Out-of-the-Box VOT Support**: Compatible with the [VOT toolkit](https://github.com/votchallenge/toolkit) for automatic benchmarking.
- 🎯 **Multi-Mask Output**: Optionally enables SAM's multi-mask mode and selects the highest-scoring mask.
- 📉 **Confidence-Based Filtering**: Suppress masks with low IoU-based confidence scores to reduce noise.

---

## 📚 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running with the VOT Toolkit](#running-with-the-vot-toolkit)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## 📋 Prerequisites

- Linux OS (Ubuntu recommended)
- Python ≥ 3.8
- NVIDIA GPU with CUDA
- Conda or Miniconda

---

## ⚙️ Installation & Setup

### Step 1: Organize Project Structure

```bash
/your/projects/
├── Stark/                   # Official STARK codebase (clone here)
└── my_vot_project/          # Your tracking integration scripts
    ├── vot_stark_sam_tracker.py
    └── vot_stark_sam_tracker_conf.py
```

### Step 2: Set Up Environment

```bash
conda create -n stark_sam python=3.8 -y
conda activate stark_sam
```

### Step 3: Install Dependencies

```bash
# Adjust PyTorch version based on your CUDA setup
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python vot-toolkit segment-anything
```

### Step 4: Download SAM Checkpoints

```bash
# Download ViT-B or ViT-H checkpoint from Meta
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P my_vot_project/checkpoints/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P my_vot_project/checkpoints/
```

### Step 5: Link to STARK

In `vot_stark_sam_tracker.py`, set the correct absolute path:

```python
# Example line to edit:
STARK_PROJECT_PATH = "/absolute/path/to/your/projects/Stark/"
```

---

## 🧪 Running with the VOT Toolkit

### Step 1: Create `trackers.ini`

Located in your VOT workspace. Define each tracker configuration:

```ini
# STARK-S + SAM ViT-B
[stark_s_sam_b]
label = STARK-S_SAM-B
protocol = trax
command = python /abs/path/my_vot_project/vot_stark_sam_tracker.py --tracker_name stark_s --sam_model_type vit_b --sam_checkpoint /abs/path/my_vot_project/checkpoints/sam_vit_b_01ec64.pth

# STARK-ST + SAM ViT-H + multimask, GPU 1
[stark_st_sam_h_multi]
label = STARK-ST_SAM-H_Multi
protocol = trax
command = python /abs/path/my_vot_project/vot_stark_sam_tracker.py --tracker_name stark_st --sam_model_type vit_h --sam_checkpoint /abs/path/my_vot_project/checkpoints/sam_vit_h_4b8939.pth --multimask_output --gpu_id 1
```

> 💡 You may need to replace `python` with the absolute path to your Conda environment’s `python` binary.

### Step 2: Run Evaluation

```bash
conda activate stark_sam
vot-evaluate --workspace /abs/path/to/vot_workspace stark_s_sam_b
```

---

## 🎛️ Configuration Reference

| Argument             | Description                                                  | Default     |
|----------------------|--------------------------------------------------------------|-------------|
| `--tracker_name`     | STARK variant (`stark_s`, `stark_st`, etc.)                  | *Required*  |
| `--sam_checkpoint`   | Path to SAM `.pth` weights                                    | *Required*  |
| `--sam_model_type`   | SAM variant (`vit_b`, `vit_l`, `vit_h`)                       | `vit_b`     |
| `--tracker_param`    | STARK parameter version (e.g., `baseline`)                    | `baseline`  |
| `--gpu_id`           | GPU index to use                                              | `3`         |
| `--confidence`       | IoU threshold for rejecting low-quality masks                 | `0.6`       |
| `--multimask_output` | Enable multiple masks and select best                         | `False`     |
| `--vot_version`      | VOT dataset version (`vot20`, `vot20lt`, etc.)                | `vot20`     |

---

## 🧯 Troubleshooting

### ❌ Tracker Fails to Launch
- Ensure all paths in `trackers.ini` are absolute and valid.
- Check that the Python executable points to the correct Conda environment.
- Run the command manually in a terminal to get more detailed error logs.

### ⚠️ RuntimeError: Error(s) in loading state_dict for Sam
- Usually caused by mismatched SAM model type and checkpoint.
- Ensure `--sam_model_type` matches the actual `.pth` file:
  - `sam_vit_h_*.pth` → `--sam_model_type vit_h`

---

