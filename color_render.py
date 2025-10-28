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
    """高质量彩色渲染器"""
    
    def __init__(self, image_size=512, device=None):
        """
        初始化彩色渲染器
        
        Args:
            image_size: 输出图像大小 (默认512以获得高质量)
            device: PyTorch设备
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.image_size = image_size
        self.cameras = FoVPerspectiveCameras(device=self.device)
        
        print(f"✓ 彩色渲染器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  图像大小: {image_size}x{image_size}")
    
    def create_renderer(self, light_location=None, ambient_color=None, 
                       diffuse_color=None, specular_color=None):
        """
        创建Phong渲染器
        
        Args:
            light_location: 光源位置 (x, y, z)
            ambient_color: 环境光颜色 RGB
            diffuse_color: 漫反射颜色 RGB
            specular_color: 镜面反射颜色 RGB
        """
        # 默认光照参数
        if light_location is None:
            light_location = ((2.0, 2.0, -3.0),)
        
        if ambient_color is None:
            ambient_color = ((0.5, 0.5, 0.5),)
        
        if diffuse_color is None:
            diffuse_color = ((0.8, 0.8, 0.8),)
        
        if specular_color is None:
            specular_color = ((0.3, 0.3, 0.3),)
        
        # 创建光源
        lights = PointLights(
            device=self.device,
            location=light_location,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color
        )
        
        # 光栅化设置
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # 创建渲染器
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
        渲染单个视角
        
        Args:
            mesh: PyTorch3D Meshes对象
            distance: 相机距离
            elevation: 仰角 (度)
            azimuth: 方位角 (度)
            light_location: 光源位置
            renderer: 自定义渲染器 (可选)
        
        Returns:
            rendered_image: 渲染的RGB图像 (H, W, 3)
        """
        # 获取相机变换
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
        
        # 创建或使用渲染器
        if renderer is None:
            renderer = self.create_renderer(light_location=light_location)
        
        # 渲染
        with torch.no_grad():
            images = renderer(meshes_world=mesh, R=R, T=T)
        
        # 转换为numpy数组并确保在0-1范围内
        image = images[0, ..., :3].cpu().numpy()
        image = np.clip(image, 0, 1)
        
        return image
    
    def render_turntable(self, mesh, output_path, num_views=36, 
                        distance=5.0, elevation=30.0,
                        light_location=None, duration=0.1):
        """
        渲染360度旋转动画
        
        Args:
            mesh: PyTorch3D Meshes对象
            output_path: 输出GIF路径
            num_views: 视角数量
            distance: 相机距离
            elevation: 仰角
            light_location: 光源位置
            duration: 每帧持续时间(秒)
        """
        print(f"\n生成360度旋转动画...")
        print(f"  视角数量: {num_views}")
        print(f"  相机距离: {distance}")
        print(f"  相机仰角: {elevation}°")
        
        # 创建渲染器
        renderer = self.create_renderer(light_location=light_location)
        
        # 生成不同方位角的图像
        images = []
        azimuths = np.linspace(0, 360, num_views, endpoint=False)
        
        for azimuth in tqdm(azimuths, desc="渲染中"):
            image = self.render_single_view(
                mesh, distance=distance, elevation=elevation, 
                azimuth=azimuth, renderer=renderer
            )
            # 转换为uint8
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            images.append(image_uint8)
        
        # 保存GIF
        imageio.mimsave(output_path, images, duration=duration)
        print(f"✓ 动画已保存: {output_path}")
        
        return images
    
    def render_multi_view_grid(self, mesh, output_path, 
                              elevations=[0, 30, 60, 90],
                              azimuths=[0, 45, 90, 135, 180, 225, 270, 315],
                              distance=5.0, light_location=None):
        """
        渲染多视角网格图
        
        Args:
            mesh: PyTorch3D Meshes对象
            output_path: 输出图像路径
            elevations: 仰角列表
            azimuths: 方位角列表
            distance: 相机距离
            light_location: 光源位置
        """
        print(f"\n生成多视角网格图...")
        print(f"  仰角: {elevations}")
        print(f"  方位角数量: {len(azimuths)}")
        
        # 创建渲染器
        renderer = self.create_renderer(light_location=light_location)
        
        # 创建图像网格
        n_rows = len(elevations)
        n_cols = len(azimuths)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        
        for i, elevation in enumerate(tqdm(elevations, desc="渲染仰角")):
            for j, azimuth in enumerate(azimuths):
                # 渲染
                image = self.render_single_view(
                    mesh, distance=distance, elevation=elevation,
                    azimuth=azimuth, renderer=renderer
                )
                
                # 显示
                ax = axes[i, j] if n_rows > 1 else axes[j]
                ax.imshow(image)
                ax.set_title(f"E:{elevation}° A:{azimuth}°", fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 多视角网格图已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TVDRNet 彩色渲染工具")
    
    # 基础参数
    parser.add_argument("--mesh", type=str, default="scene0526_01_vh_clean_2.ply",
                       help="网格文件路径")
    parser.add_argument("--output-dir", type=str, default="color_renders",
                       help="输出目录")
    parser.add_argument("--image-size", type=int, default=512,
                       help="图像大小")
    
    # 渲染模式
    parser.add_argument("--mode", type=str, default="all",
                       choices=["single", "turntable", "grid", "all"],
                       help="渲染模式: single(单视角), turntable(旋转), grid(网格), all(全部)")
    
    # 相机参数
    parser.add_argument("--distance", type=float, default=5.0,
                       help="相机距离")
    parser.add_argument("--elevation", type=float, default=30.0,
                       help="相机仰角(度)")
    parser.add_argument("--azimuth", type=float, default=45.0,
                       help="相机方位角(度)")
    
    # 颜色参数（可选，默认使用PLY文件原始颜色）
    parser.add_argument("--color", type=float, nargs=3, default=None,
                       help="可选：覆盖模型颜色 RGB (0-1)，默认使用PLY文件原始颜色")
    
    # 光照参数
    parser.add_argument("--light-pos", type=float, nargs=3, default=[2.0, 2.0, -3.0],
                       help="光源位置 [x, y, z]")
    
    # 旋转动画参数
    parser.add_argument("--num-views", type=int, default=36,
                       help="旋转动画的视角数量")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TVDRNet 彩色渲染工具")
    print("="*70)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载网格
    print("\n" + "-"*70)
    print("加载3D模型...")
    print("-"*70)
    mesh = MeshLoader.load_mesh(args.mesh, device, color=args.color)
    mesh_info = MeshLoader.get_mesh_info(mesh)
    print(f"\n模型信息:")
    print(f"  顶点数: {mesh_info['num_vertices']}")
    print(f"  面数: {mesh_info['num_faces']}")
    print(f"  中心: {mesh_info['center']}")
    
    # 创建渲染器
    print("\n" + "-"*70)
    print("初始化彩色渲染器...")
    print("-"*70)
    renderer = ColorRenderer(image_size=args.image_size, device=device)
    light_location = (tuple(args.light_pos),)
    
    # 执行渲染
    print("\n" + "="*70)
    print("开始渲染...")
    print("="*70)
    
    if args.mode in ["single", "all"]:
        print("\n[1] 单视角渲染")
        image = renderer.render_single_view(
            mesh, distance=args.distance, 
            elevation=args.elevation, azimuth=args.azimuth,
            light_location=light_location
        )
        save_path = output_dir / "single_view.png"
        plt.imsave(save_path, image)
        print(f"✓ 单视角图像已保存: {save_path}")
    
    if args.mode in ["turntable", "all"]:
        print("\n[2] 360度旋转动画")
        gif_path = output_dir / "turntable.gif"
        renderer.render_turntable(
            mesh, gif_path, num_views=args.num_views,
            distance=args.distance, elevation=args.elevation,
            light_location=light_location
        )
    
    if args.mode in ["grid", "all"]:
        print("\n[3] 多视角网格")
        grid_path = output_dir / "multi_view_grid.png"
        renderer.render_multi_view_grid(
            mesh, grid_path, distance=args.distance,
            light_location=light_location
        )
    
    # 总结
    print("\n" + "="*70)
    print("渲染完成!")
    print("="*70)
    print(f"所有输出已保存到: {output_dir}")
    print(f"\n生成的文件:")
    for file in sorted(output_dir.rglob("*")):
        if file.is_file():
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.relative_to(output_dir)} ({size:.1f} KB)")
    print("\n✓ 完成!\n")


if __name__ == "__main__":
    main()

