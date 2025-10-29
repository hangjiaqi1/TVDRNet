"""
Color Rendering for TVDRNet
High-quality colored rendering with Phong shading
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import imageio
from tqdm import tqdm

from mesh_loader import MeshLoader
from differentiable_optimizer import DifferentiableOptimizer
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)


class ColorRenderer:
    """High-quality color renderer"""
    
    def __init__(self, image_size=512, device=None):
        """
        Initialize color renderer
        
        Args:
            image_size: Output image size (default 512 for high quality)
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.image_size = image_size
        self.cameras = FoVPerspectiveCameras(device=self.device)
        
        print(f"✓ Color renderer initialized")
        print(f"  Device: {self.device}")
        print(f"  Image size: {image_size}x{image_size}")
    
    def create_renderer(self, light_location=None, ambient_color=None, 
                       diffuse_color=None, specular_color=None):
        """
        Create Phong renderer
        
        Args:
            light_location: Light position (x, y, z)
            ambient_color: Ambient light color RGB
            diffuse_color: Diffuse light color RGB
            specular_color: Specular light color RGB
        """
        # Default lighting parameters
        if light_location is None:
            light_location = ((2.0, 2.0, -3.0),)
        
        if ambient_color is None:
            ambient_color = ((0.5, 0.5, 0.5),)
        
        if diffuse_color is None:
            diffuse_color = ((0.8, 0.8, 0.8),)
        
        if specular_color is None:
            specular_color = ((0.3, 0.3, 0.3),)
        
        # Create lights
        lights = PointLights(
            device=self.device,
            location=light_location,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color
        )
        
        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # Create renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights
            )
        )
        
        return renderer
    
    def render_single_view(self, mesh, distance=5.0, elevation=30.0, azimuth=0.0,
                          light_location=None, renderer=None):
        """
        Render single view
        
        Args:
            mesh: PyTorch3D Meshes object
            distance: Camera distance
            elevation: Elevation angle (degrees)
            azimuth: Azimuth angle (degrees)
            light_location: Light position
            renderer: Custom renderer (optional)
        
        Returns:
            rendered_image: Rendered RGB image (H, W, 3)
        """
        # Get camera transform
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
        
        # Create or use renderer
        if renderer is None:
            renderer = self.create_renderer(light_location=light_location)
        
        # Render
        with torch.no_grad():
            images = renderer(meshes_world=mesh, R=R, T=T)
        
        # Convert to numpy array and ensure in 0-1 range
        image = images[0, ..., :3].cpu().numpy()
        image = np.clip(image, 0, 1)
        
        return image
    
    def render_turntable(self, mesh, output_path, num_views=36, 
                        distance=5.0, elevation=30.0,
                        light_location=None, duration=0.1):
        """
        Render 360-degree turntable animation
        
        Args:
            mesh: PyTorch3D Meshes object
            output_path: Output GIF path
            num_views: Number of views
            distance: Camera distance
            elevation: Elevation angle
            light_location: Light position
            duration: Duration per frame (seconds)
        """
        print(f"\nGenerating 360-degree turntable animation...")
        print(f"  Number of views: {num_views}")
        print(f"  Camera distance: {distance}")
        print(f"  Camera elevation: {elevation}°")
        
        # Create renderer
        renderer = self.create_renderer(light_location=light_location)
        
        # Generate images with different azimuth angles
        images = []
        azimuths = np.linspace(0, 360, num_views, endpoint=False)
        
        for azimuth in tqdm(azimuths, desc="Rendering"):
            image = self.render_single_view(
                mesh, distance=distance, elevation=elevation, 
                azimuth=azimuth, renderer=renderer
            )
            # Convert to uint8
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            images.append(image_uint8)
        
        # Save GIF
        imageio.mimsave(output_path, images, duration=duration)
        print(f"✓ Animation saved: {output_path}")
        
        return images
    
    def render_multi_view_grid(self, mesh, output_path, 
                              elevations=[0, 30, 60, 90],
                              azimuths=[0, 45, 90, 135, 180, 225, 270, 315],
                              distance=5.0, light_location=None):
        """
        Render multi-view grid
        
        Args:
            mesh: PyTorch3D Meshes object
            output_path: Output image path
            elevations: List of elevations
            azimuths: List of azimuths
            distance: Camera distance
            light_location: Light position
        """
        print(f"\nGenerating multi-view grid...")
        print(f"  Elevations: {elevations}")
        print(f"  Number of azimuths: {len(azimuths)}")
        
        # Create renderer
        renderer = self.create_renderer(light_location=light_location)
        
        # Create image grid
        n_rows = len(elevations)
        n_cols = len(azimuths)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        
        for i, elevation in enumerate(tqdm(elevations, desc="Rendering elevations")):
            for j, azimuth in enumerate(azimuths):
                # Render
                image = self.render_single_view(
                    mesh, distance=distance, elevation=elevation,
                    azimuth=azimuth, renderer=renderer
                )
                
                # Display
                ax = axes[i, j] if n_rows > 1 else axes[j]
                ax.imshow(image)
                ax.set_title(f"E:{elevation}° A:{azimuth}°", fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Multi-view grid saved: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TVDRNet Color Rendering Tool")
    
    # Basic parameters
    parser.add_argument("--mesh", type=str, default="scene0526_01_vh_clean_2.ply",
                       help="Mesh file path")
    parser.add_argument("--output-dir", type=str, default="color_renders",
                       help="Output directory")
    parser.add_argument("--image-size", type=int, default=512,
                       help="Image size")
    
    # Rendering mode
    parser.add_argument("--mode", type=str, default="all",
                       choices=["single", "turntable", "grid", "all", "optimize"],
                       help="Rendering mode: single, turntable, grid, all, optimize (differentiable optimization)")
    
    # Camera parameters
    parser.add_argument("--distance", type=float, default=5.0,
                       help="Camera distance")
    parser.add_argument("--elevation", type=float, default=30.0,
                       help="Camera elevation (degrees)")
    parser.add_argument("--azimuth", type=float, default=45.0,
                       help="Camera azimuth (degrees)")
    
    # Color parameters (optional, uses original PLY colors by default)
    parser.add_argument("--color", type=float, nargs=3, default=None,
                       help="Optional: Override model color RGB (0-1), defaults to PLY file original colors")
    
    # Lighting parameters
    parser.add_argument("--light-pos", type=float, nargs=3, default=[2.0, 2.0, -3.0],
                       help="Light position [x, y, z]")
    
    # Turntable animation parameters
    parser.add_argument("--num-views", type=int, default=36,
                       help="Number of views for turntable animation")
    
    # Differentiable optimization parameters
    parser.add_argument("--initial-distance", type=float, default=5.0,
                       help="Optimize mode: Initial camera distance")
    parser.add_argument("--initial-elevation", type=float, default=30.0,
                       help="Optimize mode: Initial camera elevation (degrees)")
    parser.add_argument("--initial-azimuth", type=float, default=45.0,
                       help="Optimize mode: Initial camera azimuth (degrees)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Optimize mode: Number of iterations")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Optimize mode: Learning rate")
    parser.add_argument("--target-image", type=str, default=None,
                       help="Optimize mode: Target image path (optional, for supervised optimization)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TVDRNet Color Rendering Tool")
    print("="*70)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load mesh
    print("\n" + "-"*70)
    print("Loading 3D model...")
    print("-"*70)
    mesh = MeshLoader.load_mesh(args.mesh, device, color=args.color)
    mesh_info = MeshLoader.get_mesh_info(mesh)
    print(f"\nModel info:")
    print(f"  Vertices: {mesh_info['num_vertices']}")
    print(f"  Faces: {mesh_info['num_faces']}")
    print(f"  Center: {mesh_info['center']}")
    
    # Create renderer
    print("\n" + "-"*70)
    print("Initializing color renderer...")
    print("-"*70)
    renderer = ColorRenderer(image_size=args.image_size, device=device)
    light_location = (tuple(args.light_pos),)
    
    # Perform rendering
    print("\n" + "="*70)
    print("Starting rendering...")
    print("="*70)
    
    if args.mode in ["single", "all"]:
        print("\n[1] Single View Rendering")
        image = renderer.render_single_view(
            mesh, distance=args.distance, 
            elevation=args.elevation, azimuth=args.azimuth,
            light_location=light_location
        )
        save_path = output_dir / "single_view.png"
        plt.imsave(save_path, image)
        print(f"✓ Single view image saved: {save_path}")
    
    if args.mode in ["turntable", "all"]:
        print("\n[2] 360-degree Turntable Animation")
        gif_path = output_dir / "turntable.gif"
        renderer.render_turntable(
            mesh, gif_path, num_views=args.num_views,
            distance=args.distance, elevation=args.elevation,
            light_location=light_location
        )
    
    if args.mode in ["grid", "all"]:
        print("\n[3] Multi-View Grid")
        grid_path = output_dir / "multi_view_grid.png"
        renderer.render_multi_view_grid(
            mesh, grid_path, distance=args.distance,
            light_location=light_location
        )
    
    if args.mode == "optimize":
        print("\n[4] Differentiable Rendering Optimization")
        
        # Create timestamped subfolder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimize_output_dir = output_dir / f"optimize_{timestamp}"
        optimize_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Optimization results will be saved to: {optimize_output_dir}")
        
        # Load target image (if provided)
        target_image = None
        if args.target_image:
            import matplotlib.image as mpimg
            target_image = mpimg.imread(args.target_image)[..., :3]
            print(f"Target image: {args.target_image}")
        
        # Create optimizer
        optimizer = DifferentiableOptimizer(
            renderer=renderer,
            mesh=mesh,
            device=device,
            target_image=target_image
        )
        
        # Run optimization (save image every N iterations)
        save_every = max(1, args.iterations // 20)  # Save about 20 frames
        optimized_params, history = optimizer.optimize(
            initial_distance=args.initial_distance,
            initial_elevation=args.initial_elevation,
            initial_azimuth=args.initial_azimuth,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            save_every=save_every,
            verbose=True
        )
        
        # Save optimization results
        import json
        result_path = optimize_output_dir / "optimized_params.json"
        with open(result_path, 'w') as f:
            json.dump(optimized_params, f, indent=2)
        print(f"\n✓ Optimized parameters saved: {result_path}")
        
        # Render with optimized parameters
        print("\nRendering optimized view...")
        optimized_image = renderer.render_single_view(
            mesh,
            distance=optimized_params['distance'],
            elevation=optimized_params['elevation'],
            azimuth=optimized_params['azimuth'],
            light_location=light_location
        )
        opt_image_path = optimize_output_dir / "optimized_view.png"
        plt.imsave(opt_image_path, optimized_image)
        print(f"✓ Optimized image saved: {opt_image_path}")
        
        # Save comparison image
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Initial view
        initial_image = renderer.render_single_view(
            mesh,
            distance=optimized_params['initial_distance'],
            elevation=optimized_params['initial_elevation'],
            azimuth=optimized_params['initial_azimuth'],
            light_location=light_location
        )
        axes[0].imshow(initial_image)
        axes[0].set_title(f"Initial View\nD:{optimized_params['initial_distance']:.1f} "
                         f"E:{optimized_params['initial_elevation']:.1f}° "
                         f"A:{optimized_params['initial_azimuth']:.1f}°")
        axes[0].axis('off')
        
        # Optimized view
        axes[1].imshow(optimized_image)
        axes[1].set_title(f"Optimized View\nD:{optimized_params['distance']:.1f} "
                         f"E:{optimized_params['elevation']:.1f}° "
                         f"A:{optimized_params['azimuth']:.1f}°")
        axes[1].axis('off')
        
        plt.tight_layout()
        comparison_path = optimize_output_dir / "optimization_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Comparison image saved: {comparison_path}")
        
        # Generate optimization process GIF animation
        print("\nGenerating optimization process animation...")
        gif_images = []
        for i, (img, iter_num) in enumerate(zip(history['images'], history['iterations'])):
            # Convert to uint8
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            # Add text annotations (show iteration number and parameters)
            from PIL import Image, ImageDraw, ImageFont
            img_pil = Image.fromarray(img_uint8)
            draw = ImageDraw.Draw(img_pil)
            
            # Get current iteration parameters
            dist = history['distance'][iter_num]
            elev = history['elevation'][iter_num]
            azim = history['azimuth'][iter_num]
            loss_val = history['loss'][iter_num]
            
            # Add text (smaller font, at the bottom)
            text = f"Iter {iter_num}/{args.iterations-1} | Loss: {loss_val:.4f}\nD:{dist:.2f} E:{elev:.1f}° A:{azim:.1f}°"
            try:
                # Use smaller font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                # If font is not available, use default font
                font = ImageFont.load_default()
            
            # Get image size and text size
            img_width, img_height = img_pil.size
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Calculate bottom position (5px margin)
            x_pos = 5
            y_pos = img_height - text_height - 10
            
            # Draw text with black background
            draw.rectangle([x_pos, y_pos, x_pos + text_width + 10, img_height - 5], fill='black')
            draw.text((x_pos + 5, y_pos + 2), text, fill='white', font=font)
            
            gif_images.append(np.array(img_pil))
        
        # Save GIF
        gif_path = optimize_output_dir / "optimization_process.gif"
        imageio.mimsave(gif_path, gif_images, duration=0.3)  # 0.3 seconds per frame
        print(f"✓ Optimization process animation saved: {gif_path}")
        print(f"  - Total frames: {len(gif_images)}")
        print(f"  - Sampling interval: every {save_every} iterations")
        
        print(f"\n{'='*70}")
        print("Command line output (use optimized parameters directly):")
        print(f"{'='*70}")
        print(f"python color_render.py --mode single \\")
        print(f"    --distance {optimized_params['distance']:.2f} \\")
        print(f"    --elevation {optimized_params['elevation']:.2f} \\")
        print(f"    --azimuth {optimized_params['azimuth']:.2f}")
        print(f"{'='*70}")
        
        print(f"\nAll optimization results saved to: {optimize_output_dir}")
        print(f"Folder name contains timestamp: optimize_{timestamp}")
    
    # Summary
    print("\n" + "="*70)
    print("Rendering complete!")
    print("="*70)
    print(f"All outputs saved to: {output_dir}")
    print(f"\nGenerated files:")
    for file in sorted(output_dir.rglob("*")):
        if file.is_file():
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.relative_to(output_dir)} ({size:.1f} KB)")
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()

