# -*- coding: utf-8 -*-
"""
Loss functions for FFD registration.

This module provides GPU-accelerated similarity metrics for image registration.

Classes
-------
- BaseLoss: Abstract base class for loss functions
- CorrelationLoss: Negative Pearson correlation (default, best for same-modality)
- MAELoss: Mean Absolute Error (L1 loss)
- MutualInformationLoss: Mutual Information with soft histogram (multi-modal)
- MonaiGlobalMILoss: MONAI's optimized Global Mutual Information (requires MONAI)

Usage Examples
--------------
>>> from accel_deform_registration.losses import CorrelationLoss, get_loss_function
>>> 
>>> # Direct instantiation
>>> loss_fn = CorrelationLoss()
>>> 
>>> # Using convenience function
>>> loss_fn = get_loss_function('correlation')
>>> loss_fn = get_loss_function('mi', num_bins=64)  # Mutual Information
>>> loss_fn = get_loss_function('monai_mi')  # MONAI's optimized MI (requires MONAI)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import warnings

import torch
import torch.nn.functional as F

# Check for MONAI availability
try:
    from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


class BaseLoss(ABC):
    """
    Abstract base class for registration loss functions.
    
    All loss functions should inherit from this class and implement
    the __call__ method.
    
    The loss function receives the warped moving image and fixed image
    as PyTorch tensors and returns a scalar loss value to minimize.
    """
    
    @abstractmethod
    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the similarity loss.
        
        Parameters
        ----------
        warped : torch.Tensor
            Warped moving image, shape (1, 1, ...) where ... is spatial dims.
        fixed : torch.Tensor
            Fixed target image, same shape as warped.
        mask : torch.Tensor, optional
            Weight mask, same spatial shape. Values in [0, 1].
        
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value to minimize.
        """
        pass
    
    @property
    def name(self) -> str:
        """Return the name of the loss function."""
        return self.__class__.__name__


class CorrelationLoss(BaseLoss):
    """
    Negative Pearson correlation coefficient loss.
    
    This is the default and recommended loss function for same-modality 
    registration (e.g., registering two OCT images). Maximizes correlation 
    by minimizing negative correlation.
    
    Parameters
    ----------
    eps : float
        Small value for numerical stability. Default 1e-8.
    
    Examples
    --------
    >>> loss_fn = CorrelationLoss()
    >>> loss = loss_fn(warped_tensor, fixed_tensor)
    
    Notes
    -----
    - Returns -1 for perfectly correlated images (identical)
    - Returns +1 for perfectly anti-correlated images
    - Returns ~0 for uncorrelated images
    """
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
    
    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute negative Pearson correlation."""
        if mask is not None:
            # Weighted correlation
            mask_flat = mask.reshape(-1)
            mask_sum = mask_flat.sum() + self.eps
            
            warped_flat = warped.reshape(-1)
            fixed_flat = fixed.reshape(-1)
            
            # Weighted means
            warped_mean = (warped_flat * mask_flat).sum() / mask_sum
            fixed_mean = (fixed_flat * mask_flat).sum() / mask_sum
            
            warped_centered = (warped_flat - warped_mean) * mask_flat
            fixed_centered = (fixed_flat - fixed_mean) * mask_flat
        else:
            # Standard correlation
            warped_flat = warped.reshape(-1)
            fixed_flat = fixed.reshape(-1)
            
            warped_mean = warped_flat.mean()
            fixed_mean = fixed_flat.mean()
            
            warped_centered = warped_flat - warped_mean
            fixed_centered = fixed_flat - fixed_mean
        
        correlation = (warped_centered * fixed_centered).sum() / (
            torch.sqrt(
                (warped_centered ** 2).sum() * (fixed_centered ** 2).sum()
            ) + self.eps
        )
        
        return -correlation


class MAELoss(BaseLoss):
    """
    Mean Absolute Error (L1) loss.
    
    Computes the mean of absolute differences between images.
    Good for general-purpose registration when correlation assumptions
    may not hold.
    
    Parameters
    ----------
    normalize : bool
        If True, normalize images to [0, 1] before computing loss.
        Default True.
    
    Examples
    --------
    >>> loss_fn = MAELoss()
    >>> loss = loss_fn(warped_tensor, fixed_tensor)
    
    Notes
    -----
    - Returns 0 for identical images
    - More robust to outliers than MSE
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean absolute error."""
        if self.normalize:
            # Normalize to [0, 1]
            warped_norm = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
            fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
        else:
            warped_norm = warped
            fixed_norm = fixed
        
        abs_diff = torch.abs(warped_norm - fixed_norm)
        
        if mask is not None:
            # Weighted MAE
            mask_sum = mask.sum() + 1e-8
            return (abs_diff * mask).sum() / mask_sum
        else:
            return abs_diff.mean()


class MutualInformationLoss(BaseLoss):
    """
    Mutual Information loss for multi-modal registration.
    
    Uses soft histogram binning for GPU-accelerated, differentiable MI computation.
    Suitable for registering images from different imaging modalities where 
    intensity relationships are non-linear.
    
    Parameters
    ----------
    num_bins : int
        Number of histogram bins. More bins capture finer intensity relationships
        but require more samples. Default 64.
    normalize : bool
        If True, return Normalized MI (NMI) which is scale-invariant. Default False.
    eps : float
        Small value for numerical stability. Default 1e-8.
    
    Examples
    --------
    >>> # Standard MI
    >>> loss_fn = MutualInformationLoss(num_bins=64)
    >>> loss = loss_fn(warped_tensor, fixed_tensor)
    >>> 
    >>> # Normalized MI (scale-invariant)
    >>> loss_fn = MutualInformationLoss(num_bins=64, normalize=True)
    
    Notes
    -----
    - Returns negative MI (we minimize loss, so minimize -MI to maximize MI)
    - For larger images or faster computation, consider MonaiGlobalMILoss
    - Typical num_bins values: 32-128
    """
    
    def __init__(
        self,
        num_bins: int = 64,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        self.num_bins = num_bins
        self.normalize = normalize
        self.eps = eps
        
        # Pre-compute bin centers
        self._bin_centers = None
    
    def _get_bin_centers(self, device: torch.device) -> torch.Tensor:
        """Get or create bin centers tensor."""
        if self._bin_centers is None or self._bin_centers.device != device:
            self._bin_centers = torch.linspace(0, 1, self.num_bins, device=device)
        return self._bin_centers
    
    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute negative Mutual Information using soft histogram binning.
        
        Uses bilinear interpolation for soft assignment to bins, making
        the computation differentiable for gradient-based optimization.
        """
        device = warped.device
        
        # Normalize to [0, 1]
        warped_norm = (warped - warped.min()) / (warped.max() - warped.min() + self.eps)
        fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min() + self.eps)
        
        # Flatten (use reshape for non-contiguous tensors)
        w_flat = warped_norm.reshape(-1)
        f_flat = fixed_norm.reshape(-1)
        n_samples = w_flat.shape[0]
        
        if mask is not None:
            m_flat = mask.reshape(-1)
            total_weight = m_flat.sum() + self.eps
        else:
            m_flat = None
            total_weight = float(n_samples)
        
        # Soft binning using bilinear interpolation
        # Each sample contributes to its two nearest bins
        
        # Compute bin indices and weights
        w_scaled = w_flat * (self.num_bins - 1)
        f_scaled = f_flat * (self.num_bins - 1)
        
        w_lower = torch.floor(w_scaled).long().clamp(0, self.num_bins - 2)
        f_lower = torch.floor(f_scaled).long().clamp(0, self.num_bins - 2)
        
        w_frac = w_scaled - w_lower.float()
        f_frac = f_scaled - f_lower.float()
        
        # Build soft 2D histogram (joint distribution)
        joint_hist = torch.zeros(self.num_bins, self.num_bins, device=device)
        
        # Four contributions per sample (bilinear)
        for dw in [0, 1]:
            for df in [0, 1]:
                w_idx = (w_lower + dw).clamp(0, self.num_bins - 1)
                f_idx = (f_lower + df).clamp(0, self.num_bins - 1)
                
                w_weight = (1 - w_frac) if dw == 0 else w_frac
                f_weight = (1 - f_frac) if df == 0 else f_frac
                
                contrib = w_weight * f_weight
                if m_flat is not None:
                    contrib = contrib * m_flat
                
                # Accumulate (use scatter_add for efficiency)
                flat_idx = w_idx * self.num_bins + f_idx
                joint_hist.view(-1).scatter_add_(0, flat_idx, contrib)
        
        # Normalize to probability
        joint_hist = joint_hist / total_weight
        
        # Marginals
        p_w = joint_hist.sum(dim=1)
        p_f = joint_hist.sum(dim=0)
        
        # Compute entropies
        H_w = -torch.sum(p_w * torch.log(p_w + self.eps))
        H_f = -torch.sum(p_f * torch.log(p_f + self.eps))
        H_wf = -torch.sum(joint_hist * torch.log(joint_hist + self.eps))
        
        # Mutual Information: I(W, F) = H(W) + H(F) - H(W, F)
        mi = H_w + H_f - H_wf
        
        if self.normalize:
            # Normalized MI: NMI = 2 * I(W, F) / (H(W) + H(F))
            mi = 2 * mi / (H_w + H_f + self.eps)
        
        return -mi


class MonaiGlobalMILoss(BaseLoss):
    """
    MONAI's optimized Global Mutual Information loss.
    
    This uses MONAI's GPU-optimized implementation which is significantly
    faster than custom implementations, especially for large images.
    Recommended for production use when MONAI is available.
    
    Parameters
    ----------
    num_bins : int
        Number of histogram bins. Default 64.
    sigma_ratio : float
        Ratio of sigma to bin width for Parzen window smoothing. Default 0.5.
    reduction : str
        Reduction mode: 'mean', 'sum', or 'none'. Default 'mean'.
    smooth_nr : float
        Smoothing constant for numerator. Default 1e-7.
    smooth_dr : float
        Smoothing constant for denominator. Default 1e-7.
    
    Raises
    ------
    ImportError
        If MONAI is not installed.
    
    Examples
    --------
    >>> loss_fn = MonaiGlobalMILoss(num_bins=64)
    >>> loss = loss_fn(warped_tensor, fixed_tensor)
    
    Notes
    -----
    Install MONAI with: pip install monai
    
    MONAI's implementation uses Parzen window estimation and is highly
    optimized for GPU computation.
    """
    
    def __init__(
        self,
        num_bins: int = 64,
        sigma_ratio: float = 0.5,
        reduction: str = 'mean',
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ):
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for MonaiGlobalMILoss. "
                "Install with: pip install monai"
            )
        
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.reduction = reduction
        
        # MONAI's GlobalMutualInformationLoss
        self._loss_fn = GlobalMutualInformationLoss(
            num_bins=num_bins,
            sigma_ratio=sigma_ratio,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
    
    @property
    def name(self) -> str:
        return "MonaiGlobalMILoss"
    
    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MONAI Global Mutual Information loss.
        
        Note: MONAI's GMI loss doesn't support masks directly.
        If a mask is provided, a warning will be issued.
        """
        if mask is not None:
            warnings.warn(
                "MonaiGlobalMILoss does not support masks. Mask will be ignored.",
                UserWarning
            )
        
        # MONAI expects (B, C, ...) format, which we already have
        # Returns negative MI (for minimization)
        return self._loss_fn(warped, fixed)


class MonaiLocalNCCLoss(BaseLoss):
    """
    MONAI's Local Normalized Cross-Correlation (LNCC) loss for image registration.
    
    This loss computes local normalized cross-correlation using a sliding window
    approach, making it robust to local intensity variations. This is particularly
    useful for:
    
    - Same-modality registration with local intensity variations
    - Images with bias field artifacts (e.g., MRI)
    - Cases where global correlation may fail due to local intensity differences
    
    The loss computes NCC within local windows and averages across the image,
    capturing local structural similarity rather than global intensity relationships.
    
    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions (2 for 2D images, 3 for 3D volumes).
    kernel_size : int
        Size of the local window for computing NCC. Must be odd.
        Larger windows capture more context but are less local.
        Typical values: 7-15 for fine features, 21-31 for coarser features.
        Default: 9.
    kernel_type : str
        Type of kernel weighting:
        - 'rectangular': Uniform weighting (fastest)
        - 'triangular': Linear weighting toward center
        - 'gaussian': Gaussian weighting (smoothest)
        Default: 'rectangular'.
    reduction : str
        How to reduce the loss across the image:
        - 'mean': Average loss across all positions (default)
        - 'sum': Sum of losses
        - 'none': Return loss map
        Default: 'mean'.
    smooth_nr : float
        Small constant added to numerator for numerical stability. Default: 0.0.
    smooth_dr : float
        Small constant added to denominator for numerical stability. Default: 1e-5.
    
    Attributes
    ----------
    name : str
        Returns "MonaiLocalNCCLoss".
    
    Raises
    ------
    ImportError
        If MONAI is not installed.
    ValueError
        If kernel_size is even.
    
    Examples
    --------
    >>> # Basic 2D usage
    >>> loss_fn = MonaiLocalNCCLoss(spatial_dims=2, kernel_size=9)
    >>> loss = loss_fn(warped_tensor, fixed_tensor)
    >>> 
    >>> # 3D with Gaussian weighting for smoother gradients
    >>> loss_fn = MonaiLocalNCCLoss(
    ...     spatial_dims=3, 
    ...     kernel_size=7, 
    ...     kernel_type='gaussian'
    ... )
    >>> loss = loss_fn(warped_volume, fixed_volume)
    
    Notes
    -----
    - Install MONAI with: pip install monai
    - Returns values in range approximately [-1, 0]
    - Values closer to -1 indicate better alignment
    - Smaller kernel_size makes the loss more sensitive to local details
    - Larger kernel_size provides more stable gradients
    
    The LNCC is computed as:
    
    $$\\text{LNCC} = \\frac{\\sum_i w_i (I_1(i) - \\mu_1)(I_2(i) - \\mu_2)}{\\sqrt{\\sum_i w_i (I_1(i) - \\mu_1)^2} \\sqrt{\\sum_i w_i (I_2(i) - \\mu_2)^2}}$$
    
    where $w_i$ are the kernel weights, and $\\mu_1, \\mu_2$ are local means.
    
    References
    ----------
    - Avants et al., "A reproducible evaluation of ANTs similarity metric 
      performance in brain image registration", NeuroImage, 2011.
    - MONAI Project: https://monai.io
    """
    
    def __init__(
        self,
        spatial_dims: int,
        kernel_size: int = 9,
        kernel_type: str = 'rectangular',
        reduction: str = 'mean',
        smooth_nr: float = 0.0,
        smooth_dr: float = 1e-5,
    ):
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for MonaiLocalNCCLoss. "
                "Install with: pip install monai"
            )
        
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        
        # Import MONAI's LocalNormalizedCrossCorrelationLoss
        from monai.losses import LocalNormalizedCrossCorrelationLoss
        
        self._loss_fn = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
    
    @property
    def name(self) -> str:
        return "MonaiLocalNCCLoss"
    
    def __call__(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Local Normalized Cross-Correlation loss.
        
        Parameters
        ----------
        warped : torch.Tensor
            Warped/moving image tensor of shape (B, C, H, W) for 2D or
            (B, C, D, H, W) for 3D.
        fixed : torch.Tensor
            Fixed/target image tensor of same shape as warped.
        mask : torch.Tensor, optional
            Not supported by MONAI's implementation. If provided, a warning
            will be issued and the mask will be ignored.
        
        Returns
        -------
        torch.Tensor
            The computed loss value. Range approximately [-1, 0], where
            -1 indicates perfect correlation (best alignment).
        
        Notes
        -----
        - The loss is negated internally to make it suitable for minimization
        - Lower values indicate better alignment
        """
        if mask is not None:
            warnings.warn(
                "MonaiLocalNCCLoss does not support masks. Mask will be ignored.",
                UserWarning
            )
        
        # MONAI expects (B, C, ...) format, which we already have
        # Returns negative NCC (for minimization)
        return self._loss_fn(warped, fixed)


def check_monai_available() -> bool:
    """
    Check if MONAI is available for optimized loss functions.
    
    Returns
    -------
    bool
        True if MONAI is installed and importable.
    """
    return MONAI_AVAILABLE


def get_loss_function(name: str, **kwargs) -> BaseLoss:
    """
    Get a loss function by name.
    
    This is a convenience function for creating loss function instances
    by string name, useful for configuration files or command-line arguments.
    
    Parameters
    ----------
    name : str
        Loss function name. Available options:
        
        Same-modality registration (recommended):
        - 'correlation', 'corr', 'pearson': CorrelationLoss
        - 'mae', 'l1': MAELoss
        - 'lncc', 'local_ncc', 'monai_lncc': MonaiLocalNCCLoss (requires MONAI)
        
        Multi-modal registration:
        - 'mi', 'mutual_information': MutualInformationLoss (soft histogram)
        - 'monai_mi', 'monai_global_mi': MonaiGlobalMILoss (requires MONAI, faster)
        
    **kwargs
        Additional arguments passed to the loss function constructor.
        Common kwargs:
        - num_bins (int): Number of bins for MI losses. Default 64.
        - normalize (bool): For MAELoss, normalize images first. Default True.
                          For MutualInformationLoss, use NMI. Default False.
        - spatial_dims (int): For MonaiLocalNCCLoss. Required.
        - kernel_size (int): For MonaiLocalNCCLoss. Default 9.
    
    Returns
    -------
    loss_fn : BaseLoss
        The loss function instance.
    
    Examples
    --------
    >>> # Correlation loss (default)
    >>> loss_fn = get_loss_function('correlation')
    >>> 
    >>> # Mutual Information with custom bins
    >>> loss_fn = get_loss_function('mi', num_bins=64)
    >>> 
    >>> # MONAI's optimized MI (faster, requires MONAI)
    >>> loss_fn = get_loss_function('monai_mi', num_bins=64)
    >>> 
    >>> # Local NCC for same-modality with local variations
    >>> loss_fn = get_loss_function('lncc', spatial_dims=2, kernel_size=9)
    
    Raises
    ------
    ValueError
        If the loss function name is not recognized.
    ImportError
        If MONAI losses are requested but MONAI is not installed.
    """
    name = name.lower()
    
    if name in ('correlation', 'corr', 'pearson'):
        return CorrelationLoss(**kwargs)
    elif name in ('mae', 'l1', 'mean_absolute_error'):
        return MAELoss(**kwargs)
    elif name in ('mi', 'mutual_information', 'mi_histogram'):
        return MutualInformationLoss(**kwargs)
    elif name in ('monai_mi', 'monai_global_mi', 'global_mi'):
        if not MONAI_AVAILABLE:
            raise ImportError(
                f"MONAI is required for '{name}'. Install with: pip install monai"
            )
        return MonaiGlobalMILoss(**kwargs)
    elif name in ('lncc', 'local_ncc', 'monai_lncc', 'local_normalized_cross_correlation'):
        if not MONAI_AVAILABLE:
            raise ImportError(
                f"MONAI is required for '{name}'. Install with: pip install monai"
            )
        return MonaiLocalNCCLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss function: '{name}'. "
            f"Available: correlation, mae, mi, monai_mi, lncc"
        )
