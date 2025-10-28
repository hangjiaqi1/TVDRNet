# TVDRNet: Text-driven Viewpoint Optimization via Differentiable Rendering for 3D Reasoning Segmentation

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![PyTorch3D](https://img.shields.io/badge/PyTorch3D-0.7.8-green)
![CUDA](https://img.shields.io/badge/CUDA-12.4-brightgreen)

**An active vision paradigm for 3D reasoning segmentation**

---

## 📖 What is TVDRNet?

The TVDRNet employs textual instructions as a supervisory signal, utilizing a differentiable renderer (a rendering system that allows gradients to flow through the rendering process, enabling end-to-end optimization of camera parameters) to guide the system to observe the 3D scene from optimal viewpoints dictated by the text's meaning. This work proposes a 3D learning paradigm that computationally determines the informative 2D virtual viewpoints for rendering, in turn building a complete 3D perception that alleviates the challenge of Erroneous localization and Boundary ambiguity in 3D Reasoning Segmentation. 

## 🗓️ Open Source Schedule

### Phase 1: Adaptive Viewpoint Position Learning ✅ (Released)

**Status**: Complete and available now

**Includes**:
- High-quality Phong rendering (`color_render.py`)
- PLY file loading with vertex colors (`mesh_loader.py`)
- Three rendering modes (single, turntable, grid)
- Complete documentation
- Example 3D scene
- Tested on PyTorch 2.5.1 + PyTorch3D 0.7.8

**What you can do**:
- Visualize 3D models with original colors
- Generate presentations and documentation
- Experiment with camera angles and lighting
- Create animations and multi-view grids

### Phase 2: Complete Evaluting Pipeline 🚧 

**Status**: coming soon



### Phase 3: Complete Training Pipeline 📋

**Status**: coming soon






### 🎯 Adaptive Viewpoint Position Learning Functionality

**Rendering Camera Parameter Optimization** - The key of Adaptive Viewpoint Position Learning (AVPL) enables automatic adjustment of camera parameters:

- **Distance** (`--distance`): Controls how far the camera is from the scene origin
- **Elevation** (`--elevation`): Controls the vertical angle (0° = eye level, 90° = top-down view)
- **Azimuth** (`--azimuth`): Controls the horizontal rotation around the scene (0° to 360°)

**How it works:** The system uses differentiable rendering to compute gradients with respect to camera parameters. Through iterative optimization (typically 50-200 iterations), it automatically finds the optimal camera position by minimizing the difference between rendered and target images. This demonstrates the power of differentiable rendering for inverse graphics problems.

```
        Z (up)
        ↑
        |    
        |   📷 Camera
        |  /
        | /
        |/________→ X (right)
       /
      /
     ↙
   Y (forward)
```


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


#### Step 4: Install Additional Dependencies

```bash
pip install numpy matplotlib imageio scikit-image tqdm plyfile
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```



## 🚀 Quick Start

### Basic Usage

```bash
# Single view rendering with original colors
python color_render.py --mode single

# 360° turntable animation (36 frames)
python color_render.py --mode turntable

# Multi-view grid (4 elevations × 8 azimuths = 32 views)
python color_render.py --mode grid

# Generate all outputs
python color_render.py --mode all
```

### Advanced Usage

```bash
# High-resolution rendering (1024×1024)
python color_render.py --mode single --image-size 1024

# Custom camera parameters
python color_render.py \
    --mode single \
    --distance 6.0 \
    --elevation 45.0 \
    --azimuth 60.0

# Custom lighting position
python color_render.py --mode single --light-pos 0.0 5.0 5.0

# Override vertex colors (e.g., red model)
python color_render.py --mode single --color 1.0 0.3 0.3

# Smooth animation with more frames
python color_render.py --mode turntable --num-views 72
```

---
