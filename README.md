# TVDRNet: Text-driven Viewpoint Optimization via Differentiable Rendering for 3D Reasoning Segmentation

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![PyTorch3D](https://img.shields.io/badge/PyTorch3D-0.7.8-green)
![CUDA](https://img.shields.io/badge/CUDA-12.4-brightgreen)

**An active vision paradigm for 3D reasoning segmentation**
<div align="center">
  <img src="https://github.com/hangjiaqi1/TVDRNet/blob/main/share-1.gif" width="49%" />
  <img src="https://github.com/hangjiaqi1/TVDRNet/blob/main/share-2.gif" width="49%" />
</div>
---

## 📖 What is TVDRNet?

The TVDRNet employs textual instructions as a supervisory signal, utilizing a differentiable renderer (a rendering system that allows gradients to flow through the rendering process, enabling end-to-end optimization of camera parameters) to guide the system to observe the 3D scene from optimal viewpoints dictated by the text's meaning. This work proposes a 3D learning paradigm that computationally determines the informative 2D virtual viewpoints for rendering, in turn building a complete 3D perception that alleviates the challenge of Erroneous localization and Boundary ambiguity in 3D Reasoning Segmentation. 

## 🗓️ Open Source Schedule

### Phase 1: Adaptive Viewpoint Position Learning ✅ (Released)

**Status**: Complete and available now

**Includes**:
- High-quality differentiable rendering (`color_render.py`)
- PLY file loading with vertex colors (`mesh_loader.py`)
- Three rendering modes (single, turntable, grid)
- Example 3D scene
- Tested on PyTorch 2.5.1 + PyTorch3D 0.7.8

**What you can do**:
- Validating the TVDRNet's key design, including the optimization of the rendering camera intrinsic (e.g., focal length) and extrinsic (e.g., azimuth, elevation) parameters.
  
### Phase 2: Complete Evaluating Pipeline 🚧 

**Status**: coming soon



### Phase 3: Complete Training Pipeline 📋

**Status**: coming soon






### 🎯 Phase 1: Adaptive Viewpoint Position Learning Functionality

**Rendering Camera Parameter Optimization** - The key of Adaptive Viewpoint Position Learning (AVPL) enables automatic adjustment of camera parameters:

- **Distance** (`--distance`): Controls how far the camera is from the scene origin
- **Elevation** (`--elevation`): Controls the vertical angle (0° = eye level, 90° = top-down view)
- **Azimuth** (`--azimuth`): Controls the horizontal rotation around the scene (0° to 360°)

**How it works:** The system uses differentiable rendering to compute gradients with respect to camera parameters. Through iterative optimization (typically 50-200 iterations), it automatically finds the optimal camera position by minimizing the difference between rendered and target images. This demonstrates the power of differentiable rendering for inverse graphics problems.

<img src="https://github.com/hangjiaqi1/TVDRNet/blob/main/avpl.jpg">


## 🛠️ Environment Setup

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Python**: 3.12 or higher
- **CUDA**: 12.4


### Installation Steps

#### Step 1: Create Python Environment

```bash
# Using Anaconda/Miniconda (recommended)
conda create -n tvdrnet python=3.12 -y
conda activate tvdrnet
```

#### Step 2: Install PyTorch with CUDA Support

PyTorch 2.5.1 with CUDA 12.4 support:

```bash
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Step 3: Build PyTorch3D from Source



**Install build dependencies:**

```bash
pip install fvcore iopath
```

**Build and install PyTorch3D:**

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
<div align="center">
  <img src="https://github.com/hangjiaqi1/TVDRNet/blob/main/share-1.gif" width="49%" />
  <img src="https://github.com/hangjiaqi1/TVDRNet/blob/main/share-2.gif" width="49%" />
</div>

#### Step 4: Install Additional Dependencies

```bash
pip install numpy matplotlib imageio scikit-image tqdm plyfile
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```



## 🚀 Quick Start

### Optimizing the camera rendering parameters

```bash
# optimizing elevation\azimuth\azimuth
python color_render.py --mode optimize --initial-distance 6.0 --initial-elevation 45.0 --initial-azimuth 90.0 --iterations 120 --image-size 256

python color_render.py --mode optimize --initial-distance 15.0 --initial-elevation 38.0 --initial-azimuth 38.0 --iterations 256 --image-size 512 --mesh "scene0015_00_vh_clean_2.ply"
```
The visualization of rendering viewpoints optimization can be found in the folder /TVDRNet_clean/color_renders.

