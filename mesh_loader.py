"""
Mesh loader module for TVDRNet
Supports PLY and OBJ file formats
"""

import torch
import numpy as np
from pathlib import Path
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


class MeshLoader:
    """Mesh loader for various 3D file formats"""
    
    @staticmethod
    def load_mesh(file_path, device, color=None):
        """
        Load mesh from file
        
        Args:
            file_path: Path to mesh file (PLY or OBJ)
            device: PyTorch device
            color: Optional color for vertices (default: white)
        
        Returns:
            PyTorch3D Meshes object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.ply':
            return MeshLoader._load_ply(file_path, device, color)
        elif suffix == '.obj':
            return MeshLoader._load_obj(file_path, device, color)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _load_ply(file_path, device, color=None):
        """Load PLY file with original vertex colors if available"""
        print(f"Loading PLY file: {file_path}")
        
        # Load PLY file geometry
        verts, faces = load_ply(str(file_path))
        
        print(f"  Loaded {verts.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Check if we need to normalize/scale the mesh
        verts_center = verts.mean(dim=0)
        verts_scale = (verts - verts_center).abs().max()
        
        print(f"  Mesh center: {verts_center.numpy()}")
        print(f"  Mesh scale: {verts_scale.item():.4f}")
        
        # Try to load vertex colors from PLY file
        verts_rgb = None
        if color is None:  # Only try to load original colors if no color specified
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(file_path))
                vertex_data = plydata['vertex']
                
                # Check if color information exists (red, green, blue channels)
                if 'red' in vertex_data.data.dtype.names:
                    r = torch.tensor(vertex_data['red'], dtype=torch.float32) / 255.0
                    g = torch.tensor(vertex_data['green'], dtype=torch.float32) / 255.0
                    b = torch.tensor(vertex_data['blue'], dtype=torch.float32) / 255.0
                    verts_rgb = torch.stack([r, g, b], dim=1)[None]  # Shape: (1, N, 3)
                    print(f"  ✓ Loaded original vertex colors from PLY file")
            except Exception as e:
                print(f"  ℹ Could not load vertex colors: {e}")
        
        # Set vertex colors
        if verts_rgb is None:
            if color is None:
                verts_rgb = torch.ones_like(verts)[None]  # White
                print(f"  Using default white color")
            else:
                verts_rgb = torch.ones_like(verts)[None] * torch.tensor(color)
                print(f"  Using specified color: {color}")
        
        # Create textures
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        
        # Create mesh
        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )
        
        print(f"  Mesh created successfully")
        
        return mesh
    
    @staticmethod
    def _load_obj(file_path, device, color=None):
        """Load OBJ file"""
        print(f"Loading OBJ file: {file_path}")
        
        # Load OBJ file
        verts, faces_idx, _ = load_obj(str(file_path))
        faces = faces_idx.verts_idx
        
        print(f"  Loaded {verts.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Set vertex colors
        if color is None:
            verts_rgb = torch.ones_like(verts)[None]  # White
        else:
            verts_rgb = torch.ones_like(verts)[None] * torch.tensor(color)
        
        # Create textures
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        
        # Create mesh
        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )
        
        print(f"  Mesh created successfully")
        
        return mesh
    
    @staticmethod
    def get_mesh_info(mesh):
        """
        Get information about mesh
        
        Args:
            mesh: PyTorch3D Meshes object
        
        Returns:
            Dictionary with mesh information
        """
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        
        verts_center = verts.mean(dim=0)
        verts_min = verts.min(dim=0)[0]
        verts_max = verts.max(dim=0)[0]
        verts_scale = (verts_max - verts_min).max()
        
        return {
            "num_vertices": verts.shape[0],
            "num_faces": faces.shape[0],
            "center": verts_center.cpu().numpy(),
            "min": verts_min.cpu().numpy(),
            "max": verts_max.cpu().numpy(),
            "scale": verts_scale.item(),
        }

