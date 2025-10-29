"""
Differentiable Rendering Optimizer with MLP
Optimizes camera parameters through differentiable rendering using MLP models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch3d.renderer import look_at_view_transform
from tqdm import tqdm


class CameraParamMLP(nn.Module):
    """
    Simple MLP model for predicting camera parameter adjustments
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        """
        Args:
            input_dim: Input dimension (initial camera parameters)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (parameter adjustment values)
        """
        super(CameraParamMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output adjustment values in [-1, 1] range
        )
        
    def forward(self, x):
        """
        Args:
            x: [distance, elevation, azimuth] normalized parameters
        Returns:
            Adjustment values [delta_distance, delta_elevation, delta_azimuth]
        """
        return self.network(x)


class DifferentiableOptimizer:
    """
    Differentiable rendering optimizer
    """
    def __init__(self, renderer, mesh, device, target_image=None):
        """
        Args:
            renderer: ColorRenderer instance
            mesh: PyTorch3D Meshes object
            device: PyTorch device
            target_image: Target image (optional, for supervised optimization)
        """
        self.renderer = renderer
        self.mesh = mesh
        self.device = device
        self.target_image = target_image
        
        # Create MLP model
        self.mlp = CameraParamMLP(input_dim=3, hidden_dim=64, output_dim=3).to(device)
        
        # Normalization parameters (for training stability)
        self.distance_scale = 10.0
        self.elevation_scale = 90.0
        self.azimuth_scale = 180.0
        
    def normalize_params(self, distance, elevation, azimuth):
        """Normalize parameters to [-1, 1] range"""
        norm_distance = distance / self.distance_scale
        norm_elevation = elevation / self.elevation_scale
        norm_azimuth = azimuth / self.azimuth_scale
        # Ensure return shape is [3]
        params = torch.stack([norm_distance, norm_elevation, norm_azimuth])
        return params.squeeze()
    
    def denormalize_params(self, norm_params):
        """Denormalize parameters"""
        distance = norm_params[0] * self.distance_scale
        elevation = norm_params[1] * self.elevation_scale
        azimuth = norm_params[2] * self.azimuth_scale
        return distance, elevation, azimuth
    
    def compute_loss(self, rendered_image, iteration):
        """
        Compute loss function
        
        Args:
            rendered_image: Rendered image [B, H, W, 4]
            iteration: Current iteration number
        
        Returns:
            loss: Total loss
        """
        # Extract RGB channels
        rgb = rendered_image[..., :3]
        
        if self.target_image is not None:
            # Supervised: MSE loss with target image
            target = torch.from_numpy(self.target_image).to(self.device)
            if target.dim() == 3:
                target = target.unsqueeze(0)
            loss = nn.functional.mse_loss(rgb, target)
        else:
            # Unsupervised: Use heuristic loss
            # 1. Encourage image not to be too dark or too bright
            brightness_loss = torch.abs(rgb.mean() - 0.5)
            
            # 2. Encourage image to have certain contrast
            contrast_loss = -rgb.std()
            
            # 3. Encourage rendering coverage (non-background pixels)
            alpha = rendered_image[..., 3]
            coverage_loss = -alpha.mean()
            
            # Combined loss
            loss = brightness_loss + 0.1 * contrast_loss + 0.5 * coverage_loss
        
        return loss
    
    def optimize(self, initial_distance, initial_elevation, initial_azimuth,
                iterations=100, learning_rate=0.001, save_every=5, verbose=True):
        """
        Optimize camera parameters through differentiable rendering and MLP
        
        Args:
            initial_distance: Initial camera distance
            initial_elevation: Initial camera elevation
            initial_azimuth: Initial camera azimuth
            iterations: Number of iterations
            learning_rate: Learning rate
            save_every: Save image every N iterations (for GIF generation)
            verbose: Whether to print detailed information
        
        Returns:
            optimized_params: Dictionary of optimized parameters
            history: Optimization history (including rendered images)
        """
        # Initialize learnable parameters
        distance = torch.tensor([initial_distance], device=self.device, requires_grad=True)
        elevation = torch.tensor([initial_elevation], device=self.device, requires_grad=True)
        azimuth = torch.tensor([initial_azimuth], device=self.device, requires_grad=True)
        
        # Optimizer: Optimize both MLP and camera parameters
        optimizer = optim.Adam([
            {'params': self.mlp.parameters(), 'lr': learning_rate},
            {'params': [distance, elevation, azimuth], 'lr': learning_rate * 0.1}
        ])
        
        # Create renderer
        phong_renderer = self.renderer.create_renderer()
        
        # Optimization history
        history = {
            'loss': [],
            'distance': [],
            'elevation': [],
            'azimuth': [],
            'images': [],  # Save images for each iteration to generate GIF
            'iterations': []  # Corresponding iteration numbers
        }
        
        if verbose:
            print("\n" + "="*70)
            print("Starting differentiable rendering optimization...")
            print("="*70)
            print(f"Initial parameters:")
            print(f"  Distance: {initial_distance:.2f}")
            print(f"  Elevation: {initial_elevation:.2f}°")
            print(f"  Azimuth: {initial_azimuth:.2f}°")
            print(f"\nIterations: {iterations}")
            print(f"Learning rate: {learning_rate}")
            print("="*70 + "\n")
        
        # Iterative optimization
        pbar = tqdm(range(iterations), desc="Optimizing") if verbose else range(iterations)
        
        for i in pbar:
            optimizer.zero_grad()
            
            # MLP predicts parameter adjustments (optional: can optimize parameters directly)
            # Here we use both MLP and direct optimization
            norm_params = self.normalize_params(distance, elevation, azimuth)
            norm_params = norm_params.unsqueeze(0)  # Add batch dimension [3] -> [1, 3]
            mlp_adjustment = self.mlp(norm_params).squeeze(0) * 0.1  # Scale adjustment and remove batch dimension
            
            # Apply adjustments
            adjusted_distance = distance + mlp_adjustment[0] * self.distance_scale
            adjusted_elevation = elevation + mlp_adjustment[1] * self.elevation_scale
            adjusted_azimuth = azimuth + mlp_adjustment[2] * self.azimuth_scale
            
            # Constrain parameter ranges
            adjusted_distance = torch.clamp(adjusted_distance, 1.0, 20.0)
            adjusted_elevation = torch.clamp(adjusted_elevation, -90.0, 90.0)
            adjusted_azimuth = torch.fmod(adjusted_azimuth, 360.0)
            
            # Differentiable rendering
            R, T = look_at_view_transform(
                adjusted_distance, adjusted_elevation, adjusted_azimuth, 
                device=self.device
            )
            rendered = phong_renderer(meshes_world=self.mesh, R=R, T=T)
            
            # Compute loss
            loss = self.compute_loss(rendered, i)
            
            # Backpropagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Record history
            with torch.no_grad():
                history['loss'].append(loss.item())
                history['distance'].append(adjusted_distance.item())
                history['elevation'].append(adjusted_elevation.item())
                history['azimuth'].append(adjusted_azimuth.item())
                
                # Save images for GIF (every save_every iterations or last iteration)
                if i % save_every == 0 or i == iterations - 1:
                    # Extract RGB image and convert to numpy
                    img_rgb = rendered[0, ..., :3].cpu().numpy()
                    img_rgb = np.clip(img_rgb, 0, 1)
                    history['images'].append(img_rgb)
                    history['iterations'].append(i)
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dist': f'{adjusted_distance.item():.2f}',
                    'elev': f'{adjusted_elevation.item():.1f}°',
                    'azim': f'{adjusted_azimuth.item():.1f}°'
                })
        
        # Get final optimized parameters
        with torch.no_grad():
            final_norm_params = self.normalize_params(distance, elevation, azimuth)
            final_norm_params = final_norm_params.unsqueeze(0)  # Add batch dimension
            final_mlp_adjustment = self.mlp(final_norm_params).squeeze(0) * 0.1
            
            final_distance = (distance + final_mlp_adjustment[0] * self.distance_scale).item()
            final_elevation = (elevation + final_mlp_adjustment[1] * self.elevation_scale).item()
            final_azimuth = (azimuth + final_mlp_adjustment[2] * self.azimuth_scale).item()
            
            # Constrain ranges
            final_distance = max(1.0, min(20.0, final_distance))
            final_elevation = max(-90.0, min(90.0, final_elevation))
            final_azimuth = final_azimuth % 360.0
        
        optimized_params = {
            'distance': final_distance,
            'elevation': final_elevation,
            'azimuth': final_azimuth,
            'initial_distance': initial_distance,
            'initial_elevation': initial_elevation,
            'initial_azimuth': initial_azimuth
        }
        
        if verbose:
            print("\n" + "="*70)
            print("Optimization complete!")
            print("="*70)
            print(f"Optimized parameters:")
            print(f"  Distance: {final_distance:.2f} (initial: {initial_distance:.2f})")
            print(f"  Elevation: {final_elevation:.2f}° (initial: {initial_elevation:.2f}°)")
            print(f"  Azimuth: {final_azimuth:.2f}° (initial: {initial_azimuth:.2f}°)")
            print(f"\nFinal loss: {history['loss'][-1]:.6f}")
            print("="*70 + "\n")
        
        return optimized_params, history

