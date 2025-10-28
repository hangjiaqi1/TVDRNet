# TVDRNet: Differentiable Rendering Visualization Tool

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![PyTorch3D](https://img.shields.io/badge/PyTorch3D-0.7.8-green)
![CUDA](https://img.shields.io/badge/CUDA-12.4-brightgreen)

**A high-quality visualization and optimization tool for differentiable rendering of 3D scenes.**

---

## 📖 What is TVDRNet?

TVDRNet is a **differentiable rendering** demonstration project that showcases camera parameter optimization through gradient-based methods. This repository contains the core implementation with two main components:

### 🎯 Core Functionality

**Camera Parameter Optimization** - The heart of this project enables automatic adjustment of three critical camera parameters:

- **Distance** (`--distance`): Controls how far the camera is from the scene origin
- **Elevation** (`--elevation`): Controls the vertical angle (0° = eye level, 90° = top-down view)
- **Azimuth** (`--azimuth`): Controls the horizontal rotation around the scene (0° to 360°)

**How it works:** The system uses differentiable rendering to compute gradients with respect to camera parameters. Through iterative optimization (typically 50-200 iterations), it automatically finds the optimal camera position by minimizing the difference between rendered and target images. This demonstrates the power of differentiable rendering for inverse graphics problems.

### 🎨 Visualization Module (Current Release)

The visualization component (`color_render.py`) provides high-quality rendering capabilities:

- **Original Vertex Colors**: Automatically loads and preserves vertex colors from PLY files
- **Phong Shading**: Realistic lighting with ambient (50%), diffuse (80%), and specular (30%) components
- **Multiple Modes**: Single view, 360° turntable animations, and multi-angle grids
- **GPU Acceleration**: CUDA-enabled for real-time rendering on NVIDIA GPUs
- **Flexible Control**: Full control over camera position, lighting, and output resolution

---

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

## 🛠️ Environment Setup

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Python**: 3.12 or higher
- **CUDA**: 12.4

### Environment Specifications

This project was developed and tested with:

```
Python:      3.12.3 (Anaconda distribution)
PyTorch:     2.5.1+cu124
PyTorch3D:   0.7.8
CUDA:        12.4
GPU:         NVIDIA GeForce RTX 4090
```

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

**Verify PyTorch installation:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.5.1+cu124
CUDA Available: True
```

#### Step 3: Build PyTorch3D from Source

⚠️ **Important**: PyTorch3D must be built from source for compatibility with PyTorch 2.5.1.

**Install build dependencies:**

```bash
pip install fvcore iopath
```

**Build and install PyTorch3D:**

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

⏱️ **Build time**: Approximately 15-20 minutes (compiles CUDA kernels)

**Build process details:**
1. Downloads source code from GitHub
2. Compiles C++ and CUDA extensions
3. Optimizes kernels for your specific GPU architecture
4. Links with PyTorch libraries
5. Installs Python bindings

**Verify PyTorch3D installation:**

```bash
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"
```

Expected output:
```
PyTorch3D: 0.7.8
```

#### Step 4: Install Additional Dependencies

```bash
pip install numpy matplotlib imageio scikit-image tqdm plyfile
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**Complete dependency list:**
- `torch==2.5.1` - Deep learning framework
- `pytorch3d==0.7.8` - 3D deep learning library
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `imageio` - Image/GIF I/O
- `scikit-image` - Image processing
- `tqdm` - Progress bars
- `plyfile` - PLY file reading

#### Step 5: Verify Installation

Run a quick test:

```bash
python color_render.py --mode single --image-size 256
```

**Expected output:**

```
✓ Loaded original vertex colors from PLY file
✓ 彩色渲染器初始化完成
  设备: cuda:0
  图像大小: 256x256
✓ 单视角图像已保存: color_renders/single_view.png
✓ 完成!
```

**Total installation time**: ~20-25 minutes

### Troubleshooting

**Issue 1: CUDA not available**
- Check GPU drivers: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Code will run on CPU if CUDA unavailable (slower)

**Issue 2: PyTorch3D build fails**
- Install build tools: `sudo apt install build-essential`
- Check CUDA version matches
- Try with verbose output: `pip install -v "git+..."`

**Issue 3: Out of memory**
- Reduce image size: `--image-size 256`
- Reduce animation frames: `--num-views 12`
- Close other GPU applications

---

## 📋 Command-Line Reference

### All Arguments

```bash
python color_render.py \
    --mesh MESH_FILE \              # PLY file path (default: scene0526_01_vh_clean_2.ply)
    --mode MODE \                   # Rendering mode: single/turntable/grid/all
    --output-dir DIR \              # Output directory (default: color_renders)
    --image-size SIZE \             # Image resolution (default: 512)
    --distance DIST \               # Camera distance (default: 5.0)
    --elevation ELEV \              # Camera elevation in degrees (default: 30.0)
    --azimuth AZI \                 # Camera azimuth in degrees (default: 45.0)
    --color R G B \                 # Override color RGB 0-1 (optional)
    --light-pos X Y Z \             # Light position (default: 2.0 2.0 -3.0)
    --num-views N                   # Turntable frames (default: 36)
```

### Rendering Modes

| Mode | Output | Description |
|------|--------|-------------|
| `single` | `single_view.png` | One viewpoint (512×512) |
| `turntable` | `turntable.gif` | 360° rotation animation |
| `grid` | `multi_view_grid.png` | 4×8 multi-angle grid |
| `all` | All of above | Complete rendering set |

### Camera Parameters Explained

**Distance** (`--distance`):
- Controls camera distance from scene origin
- Default: 5.0
- Smaller values = closer view, larger values = wider view
- Example: `--distance 3.0` (close-up), `--distance 8.0` (wide)

**Elevation** (`--elevation`):
- Vertical angle in degrees
- 0° = eye level, 90° = top-down view
- Default: 30.0
- Example: `--elevation 45.0` (45° angle), `--elevation 90.0` (bird's eye)

**Azimuth** (`--azimuth`):
- Horizontal rotation in degrees
- 0° = front, 90° = right, 180° = back, 270° = left
- Default: 45.0
- Example: `--azimuth 0.0` (front), `--azimuth 120.0` (120° rotation)

---

## 💡 Usage Examples

### Example 1: Quick Preview

Fast low-resolution render for quick testing:

```bash
python color_render.py --mode single --image-size 256
```

### Example 2: Paper-Quality Figure

High-resolution multi-view grid for publications:

```bash
python color_render.py --mode grid --image-size 1024 --distance 6.0
```

### Example 3: Presentation Animation

Smooth 72-frame turntable for presentations:

```bash
python color_render.py --mode turntable --num-views 72 --elevation 35.0
```

### Example 4: Specific Viewpoint

Render from exact camera position:

```bash
python color_render.py \
    --mode single \
    --distance 5.5 \
    --elevation 45.0 \
    --azimuth 120.0 \
    --image-size 1024
```

### Example 5: Custom Lighting

Top-down lighting for different effect:

```bash
python color_render.py --mode single --light-pos 0.0 5.0 5.0
```

### Example 6: Color Override

Render in custom color (red example):

```bash
python color_render.py --mode single --color 1.0 0.3 0.3
```

---

## 📊 Output Files

### File Structure

After running `--mode all`, you'll have:

```
color_renders/
├── single_view.png         (~60 KB)   - Single viewpoint
├── turntable.gif          (~400 KB)   - 36-frame rotation
└── multi_view_grid.png    (~650 KB)   - 32 different angles
```

### Multi-View Grid Layout

The grid shows the model from 32 different angles:

- **Rows (4)**: Elevations at 0°, 30°, 60°, 90°
- **Columns (8)**: Azimuths at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

Perfect for documentation, papers, or comprehensive visualization.

---

## ⚡ Performance

### Benchmarks (NVIDIA RTX 4090)

| Operation | Resolution | Time |
|-----------|-----------|------|
| Single frame | 512×512 | ~15 ms |
| Single frame | 1024×1024 | ~30 ms |
| Animation (36 frames) | 512×512 | ~0.5 s |
| Multi-view grid (32) | 512×512 | ~0.6 s |
| Complete (all modes) | 512×512 | ~1.5 s |

### Memory Usage

| Resolution | VRAM | RAM |
|-----------|------|-----|
| 256×256 | ~500 MB | ~2 GB |
| 512×512 | ~800 MB | ~3 GB |
| 1024×1024 | ~2 GB | ~5 GB |

### Performance Tips

**For faster rendering:**
- Use smaller resolution: `--image-size 256`
- Reduce animation frames: `--num-views 12`
- Ensure GPU is available

**For higher quality:**
- Increase resolution: `--image-size 1024`
- More animation frames: `--num-views 72`
- Adjust camera for optimal view

---

## 🔬 Technical Details

### Phong Lighting Model

The renderer uses the Phong shading model with three components:

- **Ambient Light** (50%): Provides base illumination uniformly
- **Diffuse Reflection** (80%): Angle-dependent surface lighting
- **Specular Highlights** (30%): Glossy reflections from light source

Light position is configurable via `--light-pos X Y Z`.

### Vertex Color Loading

The system automatically detects and loads RGB vertex colors from PLY files:

1. Reads PLY file structure
2. Extracts red, green, blue channels (0-255)
3. Normalizes to [0, 1] range
4. Creates PyTorch3D texture
5. Applies during rendering

If no vertex colors exist, falls back to white or uses `--color` override.

### Camera Coordinate System

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

- **Distance**: Distance from origin along viewing direction
- **Elevation**: Vertical angle (pitch)
- **Azimuth**: Horizontal angle (yaw)

---

## 📁 Project Structure

```
TVDRNet_clean/
├── README.md                        # This file
├── color_render.py                  # Main rendering script (349 lines)
├── mesh_loader.py                   # PLY/OBJ loader (160 lines)
├── scene0526_01_vh_clean_2.ply      # Example 3D scene (48K vertices)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusions
└── color_renders/                   # Output directory (auto-generated)
    ├── single_view.png
    ├── turntable.gif
    └── multi_view_grid.png
```

### Code Organization

**color_render.py** (Main Script):
- `ColorRenderer` class: Core rendering engine
- `render_single_view()`: Single viewpoint rendering
- `render_turntable()`: 360° animation generation
- `render_multi_view_grid()`: Multi-angle grid rendering
- `main()`: Command-line interface

**mesh_loader.py** (Mesh Loading):
- `MeshLoader` class: PLY/OBJ file handling
- Vertex color extraction from PLY files
- PyTorch3D mesh creation
- Texture management

---

## 🗓️ Open Source Schedule

### Phase 1: Visualization Module ✅ (Released)

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

### Phase 2: Camera Optimization Module 🚧 (Q1 2025)

**Status**: In development, to be open sourced

**Will include**:
- **Differentiable Rendering Pipeline**: End-to-end gradient computation
- **Camera Parameter Optimization**: Automatic `distance`, `elevation`, `azimuth` tuning
- **Loss Functions**: Silhouette matching, photometric loss
- **Training Scripts**: Complete optimization loop with Adam optimizer
- **Convergence Monitoring**: Real-time loss visualization
- **Configuration System**: Flexible parameter management

**Key Features**:
```python
# Optimize camera to match target image
from camera_optimizer import CameraOptimizer

optimizer = CameraOptimizer(
    mesh=scene_mesh,
    target_image=reference_image,
    learning_rate=0.15
)

result = optimizer.optimize(
    num_iterations=50,
    init_distance=5.0,
    init_elevation=30.0,
    init_azimuth=0.0
)

print(f"Optimized position: {result.camera_position}")
print(f"Final loss: {result.final_loss}")
```

**How it works**:
1. Starts from initial camera parameters
2. Renders scene using differentiable renderer
3. Computes loss vs. target image
4. Backpropagates gradients through rendering pipeline
5. Updates camera parameters using Adam optimizer
6. Iterates until convergence or max iterations

**Example use case**:
Given a target image, automatically find the best camera angle that reproduces that view. Useful for:
- Inverse graphics problems
- Camera pose estimation
- 3D reconstruction validation
- Automatic viewpoint selection

### Phase 3: Complete Training Pipeline 📋 (Q2 2025)

**Status**: Planned

**Will include**:
- Tutorial notebooks
- Multi-view consistency loss
- Batch processing capabilities
- Advanced optimization strategies
- Video demonstrations
- Extended format support (STL, OFF, OBJ)

---

## ❓ FAQ

**Q: Why is the output all white?**  
A: Your PLY file may not contain vertex colors. Try: `--color 0.3 0.7 1.0`

**Q: Can I use my own PLY files?**  
A: Yes! Just specify: `--mesh /path/to/your/model.ply`

**Q: How do I change the view angle?**  
A: Use: `--distance 6.0 --elevation 45.0 --azimuth 60.0`

**Q: The rendering is slow. How to speed up?**  
A: Reduce resolution: `--image-size 256` or frames: `--num-views 12`

**Q: Out of GPU memory?**  
A: Use smaller images: `--image-size 256` or run on CPU (automatic fallback)

**Q: How to change lighting?**  
A: Use: `--light-pos 0.0 5.0 5.0` for top lighting

**Q: Can I batch process multiple files?**  
A: Yes, use a shell script:
```bash
for file in *.ply; do
    python color_render.py --mesh "$file" --mode all --output-dir "renders_${file%.ply}"
done
```

**Q: What's the difference between modes?**  
A: 
- `single`: One image from one angle
- `turntable`: Animated 360° rotation
- `grid`: 32 images from different angles
- `all`: All of the above

---

## 🤝 Contributing

We welcome contributions! The camera optimization module will be released as open source in Q1 2025. Stay tuned for:

- Camera parameter optimization code
- Training examples
- Additional tutorials
- Community contributions

---

## 📄 License

To be announced upon full release of the camera optimization module.

---

## 🙏 Acknowledgments

This project builds upon excellent open-source work:

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch3D](https://pytorch3d.org/) - 3D computer vision library  
- The differentiable rendering research community

---

## 📧 Contact

For questions and feedback, please open an issue in the repository (coming soon with Phase 2 release).

---

**Version**: 1.0 (Visualization Module)  
**Status**: ✅ Production Ready  
**Last Updated**: October 28, 2025  
**Coming Soon**: Camera optimization module (Q1 2025)

---

**Quick Start**: `python color_render.py --mode all`  
**Documentation**: This README  
**Examples**: See usage section above  
**Installation**: See environment setup section  
**Source Code**: `color_render.py`, `mesh_loader.py`
