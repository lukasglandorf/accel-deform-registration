# -*- coding: utf-8 -*-
"""
Diffeomorphic registration utilities.

This module provides tools for ensuring diffeomorphic (topology-preserving)
deformations in FFD registration:

1. Jacobian Determinant Computation & Penalty
   - Efficient GPU computation of Jacobian determinant
   - Differentiable penalty to discourage folding during optimization

2. Fold Detection & Warning
   - Post-hoc detection of negative Jacobian determinant regions
   - Statistics and diagnostics for deformation quality

3. Stationary Velocity Field (SVF) Exponentiation
   - Scaling-and-squaring for guaranteed diffeomorphic transforms
   - Mathematically ensures invertibility and topology preservation

Theory
------
For a displacement field u(x), the Jacobian of the transformation is:
    J = I + ∇u

where I is the identity matrix. The determinant det(J) indicates:
    - det(J) > 0: locally invertible (no folding)
    - det(J) = 0: singular (degenerate)
    - det(J) < 0: folding (topology violation)

For diffeomorphic transforms, we want det(J) > 0 everywhere.

The SVF approach parameterizes the transform as ϕ = exp(v) where v is a 
stationary velocity field. The exponential is computed via scaling-and-squaring:
    ϕ = exp(v) ≈ (exp(v/2^n))^(2^n)
This guarantees ϕ is a diffeomorphism if v is smooth.

References
----------
- Arsigny et al., "A Log-Euclidean Framework for Statistics on Diffeomorphisms", MICCAI 2006
- Dalca et al., "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration", MICCAI 2018
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


# =============================================================================
# Jacobian Determinant Computation
# =============================================================================

def compute_jacobian_determinant_2d(
    displacement: torch.Tensor,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """
    Compute the Jacobian determinant of a 2D displacement field.
    
    For displacement u = (u_x, u_y), the Jacobian of the transformation
    T(x) = x + u(x) is:
        J = I + ∇u = [[1 + ∂u_x/∂x, ∂u_x/∂y],
                      [∂u_y/∂x, 1 + ∂u_y/∂y]]
    
    det(J) = (1 + ∂u_x/∂x)(1 + ∂u_y/∂y) - (∂u_x/∂y)(∂u_y/∂x)
    
    Parameters
    ----------
    displacement : torch.Tensor
        Displacement field of shape (N, 2, H, W) or (2, H, W) in voxel units.
        Channel 0 is displacement in x, channel 1 is displacement in y.
    voxel_spacing : tuple of float
        Physical spacing (dy, dx) for gradient computation. Default (1, 1).
    
    Returns
    -------
    det_J : torch.Tensor
        Jacobian determinant of shape (N, 1, H, W) or (1, H, W).
    """
    # Ensure 4D input
    squeeze_batch = False
    if displacement.dim() == 3:
        displacement = displacement.unsqueeze(0)
        squeeze_batch = True
    
    N, C, H, W = displacement.shape
    assert C == 2, f"Expected 2 channels for 2D displacement, got {C}"
    
    dy, dx = voxel_spacing
    
    # Extract displacement components
    u_x = displacement[:, 0:1, :, :]  # (N, 1, H, W)
    u_y = displacement[:, 1:2, :, :]  # (N, 1, H, W)
    
    # Compute spatial gradients using central differences
    # ∂u/∂x using [-1, 0, 1] / (2*dx) kernel
    # Pad to maintain size, then use unfold or conv
    
    # Gradient of u_x with respect to x (along W dimension)
    du_x_dx = (
        F.pad(u_x, (0, 2, 0, 0), mode='replicate')[:, :, :, 2:] -
        F.pad(u_x, (2, 0, 0, 0), mode='replicate')[:, :, :, :-2]
    ) / (2.0 * dx)
    
    # Gradient of u_x with respect to y (along H dimension)
    du_x_dy = (
        F.pad(u_x, (0, 0, 0, 2), mode='replicate')[:, :, 2:, :] -
        F.pad(u_x, (0, 0, 2, 0), mode='replicate')[:, :, :-2, :]
    ) / (2.0 * dy)
    
    # Gradient of u_y with respect to x
    du_y_dx = (
        F.pad(u_y, (0, 2, 0, 0), mode='replicate')[:, :, :, 2:] -
        F.pad(u_y, (2, 0, 0, 0), mode='replicate')[:, :, :, :-2]
    ) / (2.0 * dx)
    
    # Gradient of u_y with respect to y
    du_y_dy = (
        F.pad(u_y, (0, 0, 0, 2), mode='replicate')[:, :, 2:, :] -
        F.pad(u_y, (0, 0, 2, 0), mode='replicate')[:, :, :-2, :]
    ) / (2.0 * dy)
    
    # Jacobian determinant: det(I + ∇u) = (1 + ∂u_x/∂x)(1 + ∂u_y/∂y) - (∂u_x/∂y)(∂u_y/∂x)
    det_J = (1.0 + du_x_dx) * (1.0 + du_y_dy) - du_x_dy * du_y_dx
    
    if squeeze_batch:
        det_J = det_J.squeeze(0)
    
    return det_J


def compute_jacobian_determinant_3d(
    displacement: torch.Tensor,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Compute the Jacobian determinant of a 3D displacement field.
    
    For displacement u = (u_x, u_y, u_z), the Jacobian of T(x) = x + u(x) is:
        J = I + ∇u (3x3 matrix at each voxel)
    
    det(J) is computed using the standard 3x3 determinant formula.
    
    Parameters
    ----------
    displacement : torch.Tensor
        Displacement field of shape (N, 3, D, H, W) or (3, D, H, W) in voxel units.
        Channels: 0=dx, 1=dy, 2=dz.
    voxel_spacing : tuple of float
        Physical spacing (dz, dy, dx) for gradient computation. Default (1, 1, 1).
    
    Returns
    -------
    det_J : torch.Tensor
        Jacobian determinant of shape (N, 1, D, H, W) or (1, D, H, W).
    """
    squeeze_batch = False
    if displacement.dim() == 4:
        displacement = displacement.unsqueeze(0)
        squeeze_batch = True
    
    N, C, D, H, W = displacement.shape
    assert C == 3, f"Expected 3 channels for 3D displacement, got {C}"
    
    dz, dy, dx = voxel_spacing
    
    # Extract displacement components
    u_x = displacement[:, 0:1, :, :, :]  # (N, 1, D, H, W)
    u_y = displacement[:, 1:2, :, :, :]
    u_z = displacement[:, 2:3, :, :, :]
    
    # Helper function for 3D central difference gradient
    def gradient_3d(u, dim, spacing):
        """Compute gradient along specified dimension using central differences."""
        if dim == 2:  # D dimension (z)
            grad = (
                F.pad(u, (0, 0, 0, 0, 0, 2), mode='replicate')[:, :, 2:, :, :] -
                F.pad(u, (0, 0, 0, 0, 2, 0), mode='replicate')[:, :, :-2, :, :]
            ) / (2.0 * spacing)
        elif dim == 3:  # H dimension (y)
            grad = (
                F.pad(u, (0, 0, 0, 2, 0, 0), mode='replicate')[:, :, :, 2:, :] -
                F.pad(u, (0, 0, 2, 0, 0, 0), mode='replicate')[:, :, :, :-2, :]
            ) / (2.0 * spacing)
        elif dim == 4:  # W dimension (x)
            grad = (
                F.pad(u, (0, 2, 0, 0, 0, 0), mode='replicate')[:, :, :, :, 2:] -
                F.pad(u, (2, 0, 0, 0, 0, 0), mode='replicate')[:, :, :, :, :-2]
            ) / (2.0 * spacing)
        else:
            raise ValueError(f"Invalid dimension {dim}")
        return grad
    
    # Compute all partial derivatives (9 components)
    du_x_dx = gradient_3d(u_x, 4, dx)
    du_x_dy = gradient_3d(u_x, 3, dy)
    du_x_dz = gradient_3d(u_x, 2, dz)
    
    du_y_dx = gradient_3d(u_y, 4, dx)
    du_y_dy = gradient_3d(u_y, 3, dy)
    du_y_dz = gradient_3d(u_y, 2, dz)
    
    du_z_dx = gradient_3d(u_z, 4, dx)
    du_z_dy = gradient_3d(u_z, 3, dy)
    du_z_dz = gradient_3d(u_z, 2, dz)
    
    # Jacobian matrix J = I + ∇u:
    # J = [[1+∂u_x/∂x, ∂u_x/∂y, ∂u_x/∂z],
    #      [∂u_y/∂x, 1+∂u_y/∂y, ∂u_y/∂z],
    #      [∂u_z/∂x, ∂u_z/∂y, 1+∂u_z/∂z]]
    
    J00 = 1.0 + du_x_dx
    J01 = du_x_dy
    J02 = du_x_dz
    J10 = du_y_dx
    J11 = 1.0 + du_y_dy
    J12 = du_y_dz
    J20 = du_z_dx
    J21 = du_z_dy
    J22 = 1.0 + du_z_dz
    
    # 3x3 determinant using Sarrus' rule / cofactor expansion
    det_J = (
        J00 * (J11 * J22 - J12 * J21) -
        J01 * (J10 * J22 - J12 * J20) +
        J02 * (J10 * J21 - J11 * J20)
    )
    
    if squeeze_batch:
        det_J = det_J.squeeze(0)
    
    return det_J


# =============================================================================
# Jacobian Penalty (Differentiable, for optimization)
# =============================================================================

def jacobian_penalty(
    displacement: torch.Tensor,
    ndim: int,
    eps: float = 0.01,
    mode: str = 'relu',
    voxel_spacing: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """
    Compute a differentiable penalty for negative/small Jacobian determinants.
    
    This penalty encourages the displacement field to be diffeomorphic
    (locally invertible, no folding) by penalizing regions where det(J) < eps.
    
    Parameters
    ----------
    displacement : torch.Tensor
        Displacement field of shape (N, C, ...) where C=2 for 2D, C=3 for 3D.
    ndim : int
        Number of spatial dimensions (2 or 3).
    eps : float
        Threshold below which det(J) is penalized. Default 0.01.
        - eps=0: only penalize actual folds (det(J) < 0)
        - eps>0: also discourage near-singular regions
    mode : str
        Penalty mode:
        - 'relu': ReLU(eps - det(J))^2, penalizes det(J) < eps
        - 'log': (log(det(J) + delta))^2, soft penalty encouraging det(J) ≈ 1
    voxel_spacing : tuple of float, optional
        Physical voxel spacing. Default (1, ..., 1).
    
    Returns
    -------
    penalty : torch.Tensor
        Scalar penalty value (mean over all voxels).
    """
    if ndim == 2:
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0)
        det_J = compute_jacobian_determinant_2d(displacement, voxel_spacing)
    elif ndim == 3:
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)
        det_J = compute_jacobian_determinant_3d(displacement, voxel_spacing)
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    
    if mode == 'relu':
        # Penalize det(J) < eps: ReLU(eps - det(J))^2
        penalty = (F.relu(eps - det_J) ** 2).mean()
    elif mode == 'log':
        # Soft log penalty: encourages det(J) ≈ 1
        # Use small delta to avoid log(0)
        delta = 1e-6
        log_det = torch.log(det_J.clamp(min=delta))
        penalty = (log_det ** 2).mean()
    else:
        raise ValueError(f"Unknown penalty mode: {mode}. Use 'relu' or 'log'.")
    
    return penalty


# =============================================================================
# Fold Detection & Statistics
# =============================================================================

@dataclass
class JacobianStats:
    """Statistics about the Jacobian determinant of a displacement field."""
    min_det: float
    max_det: float
    mean_det: float
    std_det: float
    num_folds: int
    fold_fraction: float
    num_near_singular: int  # det < 0.1
    near_singular_fraction: float
    total_voxels: int
    has_folds: bool


def detect_folds(
    displacement: Union[torch.Tensor, NDArray],
    ndim: int,
    voxel_spacing: Optional[Tuple[float, ...]] = None,
    fold_threshold: float = 0.0,
    warn: bool = True,
) -> JacobianStats:
    """
    Detect folding (negative Jacobian determinant) in a displacement field.
    
    Parameters
    ----------
    displacement : torch.Tensor or ndarray
        Displacement field of shape (C, ...) or (N, C, ...) where C=2 (2D) or C=3 (3D).
    ndim : int
        Number of spatial dimensions (2 or 3).
    voxel_spacing : tuple of float, optional
        Physical voxel spacing. Default (1, ..., 1).
    fold_threshold : float
        Threshold for detecting folds. det(J) < threshold is considered folded.
        Default 0.0 (only true folds).
    warn : bool
        If True and folds detected, emit a warning. Default True.
    
    Returns
    -------
    stats : JacobianStats
        Statistics about the Jacobian determinant.
    """
    # Convert to tensor if needed
    if isinstance(displacement, np.ndarray):
        displacement = torch.from_numpy(displacement.astype(np.float32))
    
    # Handle permutation: if last dim is channels, permute
    # Expected: (N, C, ...) or (C, ...)
    if ndim == 2 and displacement.shape[-1] == 2:
        # (Y, X, 2) -> (2, Y, X) or (N, Y, X, 2) -> (N, 2, Y, X)
        if displacement.dim() == 3:
            displacement = displacement.permute(2, 0, 1)
        else:
            displacement = displacement.permute(0, 3, 1, 2)
    elif ndim == 3 and displacement.shape[-1] == 3:
        # (Z, Y, X, 3) -> (3, Z, Y, X) or (N, Z, Y, X, 3) -> (N, 3, Z, Y, X)
        if displacement.dim() == 4:
            displacement = displacement.permute(3, 0, 1, 2)
        else:
            displacement = displacement.permute(0, 4, 1, 2, 3)
    
    # Compute Jacobian determinant
    if ndim == 2:
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0)
        det_J = compute_jacobian_determinant_2d(displacement, voxel_spacing)
    elif ndim == 3:
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)
        det_J = compute_jacobian_determinant_3d(displacement, voxel_spacing)
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    
    # Compute statistics
    det_np = det_J.detach().cpu().numpy().flatten()
    
    min_det = float(det_np.min())
    max_det = float(det_np.max())
    mean_det = float(det_np.mean())
    std_det = float(det_np.std())
    
    num_folds = int((det_np < fold_threshold).sum())
    total_voxels = det_np.size
    fold_fraction = num_folds / total_voxels
    
    num_near_singular = int((det_np < 0.1).sum())
    near_singular_fraction = num_near_singular / total_voxels
    
    has_folds = num_folds > 0
    
    stats = JacobianStats(
        min_det=min_det,
        max_det=max_det,
        mean_det=mean_det,
        std_det=std_det,
        num_folds=num_folds,
        fold_fraction=fold_fraction,
        num_near_singular=num_near_singular,
        near_singular_fraction=near_singular_fraction,
        total_voxels=total_voxels,
        has_folds=has_folds,
    )
    
    if warn and has_folds:
        warnings.warn(
            f"Detected {num_folds} voxels ({fold_fraction*100:.2f}%) with det(J) < {fold_threshold}. "
            f"Min det(J) = {min_det:.4f}. The deformation field contains folding "
            f"(topology violations). Consider increasing regularization weights or "
            f"using jacobian_penalty_weight > 0.",
            UserWarning,
        )
    
    return stats


def get_jacobian_determinant_map(
    displacement: Union[torch.Tensor, NDArray],
    ndim: int,
    voxel_spacing: Optional[Tuple[float, ...]] = None,
) -> NDArray:
    """
    Compute the Jacobian determinant map for visualization/analysis.
    
    Parameters
    ----------
    displacement : torch.Tensor or ndarray
        Displacement field of shape (..., C) where C=2 (2D) or C=3 (3D).
    ndim : int
        Number of spatial dimensions (2 or 3).
    voxel_spacing : tuple of float, optional
        Physical voxel spacing. Default (1, ..., 1).
    
    Returns
    -------
    det_J_map : ndarray
        Jacobian determinant map of shape (H, W) for 2D or (D, H, W) for 3D.
    """
    # Convert to tensor if needed
    if isinstance(displacement, np.ndarray):
        displacement = torch.from_numpy(displacement.astype(np.float32))
    
    # Handle permutation: if last dim is channels, permute
    if ndim == 2 and displacement.shape[-1] == 2:
        displacement = displacement.permute(2, 0, 1)
    elif ndim == 3 and displacement.shape[-1] == 3:
        displacement = displacement.permute(3, 0, 1, 2)
    
    # Compute Jacobian determinant
    if ndim == 2:
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0)
        det_J = compute_jacobian_determinant_2d(displacement, voxel_spacing)
    elif ndim == 3:
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)
        det_J = compute_jacobian_determinant_3d(displacement, voxel_spacing)
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    
    return det_J.squeeze().cpu().numpy()


# =============================================================================
# Stationary Velocity Field (SVF) Exponentiation
# =============================================================================

def compose_displacement_2d(
    disp1: torch.Tensor,
    disp2: torch.Tensor,
) -> torch.Tensor:
    """
    Compose two 2D displacement fields: disp1 ∘ disp2.
    
    The composition (disp1 ∘ disp2)(x) = disp1(x + disp2(x)) + disp2(x)
    represents applying disp2 first, then disp1.
    
    Parameters
    ----------
    disp1 : torch.Tensor
        First displacement field of shape (N, 2, H, W).
    disp2 : torch.Tensor
        Second displacement field of shape (N, 2, H, W).
    
    Returns
    -------
    composed : torch.Tensor
        Composed displacement field of shape (N, 2, H, W).
    """
    N, C, H, W = disp1.shape
    device = disp1.device
    
    # Create identity grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    identity_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)
    
    # Convert disp2 to normalized grid coordinates
    disp2_norm = torch.zeros_like(disp2)
    disp2_norm[:, 0] = disp2[:, 0] / (W / 2)  # dx
    disp2_norm[:, 1] = disp2[:, 1] / (H / 2)  # dy
    
    # Sample disp1 at locations x + disp2(x)
    sample_grid = identity_grid + disp2_norm.permute(0, 2, 3, 1)
    disp1_at_deformed = F.grid_sample(
        disp1, sample_grid, mode='bilinear',
        padding_mode='border', align_corners=True
    )
    
    # Composition: disp1(x + disp2(x)) + disp2(x)
    composed = disp1_at_deformed + disp2
    
    return composed


def compose_displacement_3d(
    disp1: torch.Tensor,
    disp2: torch.Tensor,
) -> torch.Tensor:
    """
    Compose two 3D displacement fields: disp1 ∘ disp2.
    
    Parameters
    ----------
    disp1 : torch.Tensor
        First displacement field of shape (N, 3, D, H, W).
    disp2 : torch.Tensor
        Second displacement field of shape (N, 3, D, H, W).
    
    Returns
    -------
    composed : torch.Tensor
        Composed displacement field of shape (N, 3, D, H, W).
    """
    N, C, D, H, W = disp1.shape
    device = disp1.device
    
    # Create identity grid
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    identity_grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).expand(N, -1, -1, -1, -1)
    
    # Convert disp2 to normalized grid coordinates
    disp2_norm = torch.zeros_like(disp2)
    disp2_norm[:, 0] = disp2[:, 0] / (W / 2)  # dx
    disp2_norm[:, 1] = disp2[:, 1] / (H / 2)  # dy
    disp2_norm[:, 2] = disp2[:, 2] / (D / 2)  # dz
    
    # Sample disp1 at locations x + disp2(x)
    sample_grid = identity_grid + disp2_norm.permute(0, 2, 3, 4, 1)
    disp1_at_deformed = F.grid_sample(
        disp1, sample_grid, mode='bilinear',
        padding_mode='border', align_corners=True
    )
    
    # Composition
    composed = disp1_at_deformed + disp2
    
    return composed


def scaling_squaring_2d(
    velocity: torch.Tensor,
    n_steps: int = 7,
) -> torch.Tensor:
    """
    Exponentiate a 2D velocity field to get a diffeomorphic displacement.
    
    Implements scaling-and-squaring: ϕ = exp(v) ≈ (exp(v/2^n))^(2^n)
    
    For small v/2^n, we approximate exp(v/2^n) ≈ id + v/2^n (i.e., the
    displacement equals the scaled velocity). Then we compose n times.
    
    Parameters
    ----------
    velocity : torch.Tensor
        Stationary velocity field of shape (N, 2, H, W) in voxel units.
    n_steps : int
        Number of scaling-squaring steps. More steps = more accurate.
        Default 7 (divides velocity by 128).
    
    Returns
    -------
    displacement : torch.Tensor
        Diffeomorphic displacement field of shape (N, 2, H, W).
    
    Notes
    -----
    The resulting displacement field is guaranteed to have det(J) > 0
    everywhere if the input velocity is smooth (which it is after B-spline
    upsampling).
    """
    # Scale velocity
    displacement = velocity / (2 ** n_steps)
    
    # Square n_steps times
    for _ in range(n_steps):
        displacement = compose_displacement_2d(displacement, displacement)
    
    return displacement


def scaling_squaring_3d(
    velocity: torch.Tensor,
    n_steps: int = 7,
) -> torch.Tensor:
    """
    Exponentiate a 3D velocity field to get a diffeomorphic displacement.
    
    Implements scaling-and-squaring: ϕ = exp(v) ≈ (exp(v/2^n))^(2^n)
    
    Parameters
    ----------
    velocity : torch.Tensor
        Stationary velocity field of shape (N, 3, D, H, W) in voxel units.
    n_steps : int
        Number of scaling-squaring steps. Default 7.
    
    Returns
    -------
    displacement : torch.Tensor
        Diffeomorphic displacement field of shape (N, 3, D, H, W).
    """
    # Scale velocity
    displacement = velocity / (2 ** n_steps)
    
    # Square n_steps times
    for _ in range(n_steps):
        displacement = compose_displacement_3d(displacement, displacement)
    
    return displacement


def integrate_svf(
    velocity: torch.Tensor,
    ndim: int,
    n_steps: int = 7,
) -> torch.Tensor:
    """
    Integrate a stationary velocity field to get diffeomorphic displacement.
    
    Convenience wrapper that dispatches to 2D or 3D implementation.
    
    Parameters
    ----------
    velocity : torch.Tensor
        Stationary velocity field. Shape (N, 2, H, W) or (N, 3, D, H, W).
    ndim : int
        Number of spatial dimensions (2 or 3).
    n_steps : int
        Number of scaling-squaring steps. Default 7.
    
    Returns
    -------
    displacement : torch.Tensor
        Diffeomorphic displacement field.
    """
    if ndim == 2:
        return scaling_squaring_2d(velocity, n_steps)
    elif ndim == 3:
        return scaling_squaring_3d(velocity, n_steps)
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
