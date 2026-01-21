# -*- coding: utf-8 -*-
"""
Common utilities for Free-Form Deformation (FFD) registration.

This module provides shared functions used by both 2D and 3D FFD implementations.

Functions
---------
1. get_default_device: Get PyTorch device with meaningful error handling
2. normalize_image: Normalize image to [0, 1] range
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional
from numpy.typing import NDArray


def get_default_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Get the default PyTorch device for FFD computations.
    
    If no device is specified, attempts to use CUDA if available,
    otherwise falls back to CPU.
    
    Parameters
    ----------
    device : torch.device, optional
        Specific device to use. If None, auto-detect.
    
    Returns
    -------
    torch.device
        The device to use for computations.
    
    Raises
    ------
    RuntimeError
        If no suitable device is found (should not happen with CPU fallback).
    
    Examples
    --------
    >>> device = get_default_device()
    >>> print(device)
    cuda:0  # or cpu
    """
    if device is not None:
        return device
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            # Use first GPU by default
            device = torch.device('cuda:0')
        return device
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    
    # Fallback to CPU
    return torch.device('cpu')


def normalize_image(
    image: NDArray[np.floating],
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """
    Normalize image to [0, 1] range.
    
    Parameters
    ----------
    image : ndarray
        Input image of any shape.
    eps : float
        Small value to prevent division by zero.
    
    Returns
    -------
    normalized : ndarray
        Image normalized to [0, 1] range as float32.
    """
    img = image.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + eps)


def compute_validity_mask_2d(
    displacement_field: NDArray[np.floating],
    image_shape: tuple[int, int],
    margin: float = 1.0,
) -> NDArray[np.bool_]:
    """
    Compute a validity mask for a 2D displacement field.
    
    The mask indicates which pixels in the warped image came from valid
    source locations (within the original image boundaries).
    
    Parameters
    ----------
    displacement_field : ndarray
        2D displacement field of shape (Y, X, 2) with dx, dy in pixel units.
    image_shape : tuple
        Shape of the image (Y, X).
    margin : float
        Pixels within this margin of the boundary are considered invalid.
        Default 1.0 pixel.
    
    Returns
    -------
    mask : ndarray
        Boolean mask of shape (Y, X). True where pixels are valid.
    """
    Y, X = image_shape
    
    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    
    # Compute source positions (where each pixel comes from)
    source_x = xx - displacement_field[:, :, 0]
    source_y = yy - displacement_field[:, :, 1]
    
    # Check if source positions are within valid bounds
    valid_x = (source_x >= margin) & (source_x <= X - 1 - margin)
    valid_y = (source_y >= margin) & (source_y <= Y - 1 - margin)
    
    return valid_x & valid_y


def compute_validity_mask_3d(
    displacement_field: NDArray[np.floating],
    volume_shape: tuple[int, int, int],
    margin: float = 1.0,
) -> NDArray[np.bool_]:
    """
    Compute a validity mask for a 3D displacement field.
    
    The mask indicates which voxels in the warped volume came from valid
    source locations (within the original volume boundaries).
    
    Parameters
    ----------
    displacement_field : ndarray
        3D displacement field of shape (Z, Y, X, 3) with dx, dy, dz in voxel units.
    volume_shape : tuple
        Shape of the volume (Z, Y, X).
    margin : float
        Voxels within this margin of the boundary are considered invalid.
        Default 1.0 voxel.
    
    Returns
    -------
    mask : ndarray
        Boolean mask of shape (Z, Y, X). True where voxels are valid.
    """
    Z, Y, X = volume_shape
    
    # Create coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )
    
    # Compute source positions (where each voxel comes from)
    source_x = xx - displacement_field[:, :, :, 0]
    source_y = yy - displacement_field[:, :, :, 1]
    source_z = zz - displacement_field[:, :, :, 2]
    
    # Check if source positions are within valid bounds
    valid_x = (source_x >= margin) & (source_x <= X - 1 - margin)
    valid_y = (source_y >= margin) & (source_y <= Y - 1 - margin)
    valid_z = (source_z >= margin) & (source_z <= Z - 1 - margin)
    
    return valid_x & valid_y & valid_z
