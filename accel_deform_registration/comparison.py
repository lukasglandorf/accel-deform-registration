# -*- coding: utf-8 -*-
"""
Loss function comparison utilities.

This module provides convenience functions for comparing different loss functions
and registration approaches across 2D and 3D data.

Functions
---------
1. compare_loss_functions: Run all loss functions with multiple approaches
2. compare_approaches_2d: Compare single vs pyramid vs multiscale for 2D
3. compare_approaches_3d: Compare single vs pyramid vs multiscale for 3D
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from numpy.typing import NDArray

from .ffd_2d import register_ffd_2d, apply_ffd_2d
from .ffd_3d import register_ffd_3d, apply_ffd_3d
from .pyramid import (
    multiscale_register_2d, multiscale_register_3d,
    apply_transforms_2d, apply_transforms_3d,
)
from .losses import (
    BaseLoss, CorrelationLoss, MAELoss, MutualInformationLoss,
    MonaiGlobalMILoss, MonaiLocalNCCLoss, MONAI_AVAILABLE,
)


@dataclass
class RegistrationResult:
    """Container for registration results."""
    loss_name: str
    approach: str  # 'single' or 'multiscale'
    registered: NDArray[np.float32]
    initial_correlation: float
    final_correlation: float
    correlation_improvement: float
    time_seconds: float
    info: Dict[str, Any]


def get_available_loss_functions(
    include_monai: bool = True,
    ndim: int = 2,
) -> Dict[str, BaseLoss]:
    """
    Get dictionary of available loss functions.
    
    Parameters
    ----------
    include_monai : bool
        If True and MONAI is available, include MONAI loss functions.
    ndim : int
        Number of spatial dimensions (2 or 3). Used for MONAI NCC.
    
    Returns
    -------
    dict
        Dictionary mapping loss names to loss function instances.
    """
    losses = {
        'correlation': CorrelationLoss(),
        'mae': MAELoss(),
        'mi': MutualInformationLoss(num_bins=32),
    }
    
    if include_monai and MONAI_AVAILABLE:
        losses['monai_mi'] = MonaiGlobalMILoss()
        losses['monai_ncc'] = MonaiLocalNCCLoss(spatial_dims=ndim)
    
    return losses


def compare_loss_functions(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    loss_names: Optional[List[str]] = None,
    approaches: Optional[List[str]] = None,
    grid_spacings_single: Union[int, Tuple[int, ...]] = None,
    grid_spacings_multiscale: List[Union[int, Tuple[int, ...]]] = None,
    iterations_single: int = None,
    iterations_multiscale: List[int] = None,
    device=None,
    verbose: bool = True,
) -> Dict[str, RegistrationResult]:
    """
    Compare all loss functions with different registration approaches.
    
    This convenience function runs registration with multiple loss functions
    and approaches (single-level vs multiscale), returning all results for
    easy comparison.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) image/volume. Shape (Y, X) for 2D or (Z, Y, X) for 3D.
    fixed : ndarray
        Fixed (target) image/volume. Same shape as moving.
    loss_names : list of str, optional
        Loss functions to compare. Default: all available.
        Options: 'correlation', 'mae', 'mi', 'monai_mi', 'monai_ncc'
    approaches : list of str, optional
        Approaches to compare. Default: ['single', 'multiscale'].
    grid_spacings_single : int or tuple, optional
        Grid spacing for single-level registration.
        Default: 50 for 2D, 40 for 3D.
    grid_spacings_multiscale : list, optional
        Grid spacings for multiscale registration.
        Default: [100, 50, 25] for 2D, [60, 35, 20] for 3D.
    iterations_single : int, optional
        Iterations for single-level. Default: 500 for 2D, 200 for 3D.
    iterations_multiscale : list of int, optional
        Iterations per level for multiscale. Default: [400, 250, 150].
    device : torch.device, optional
        PyTorch device.
    verbose : bool
        Print progress information. Default True.
    
    Returns
    -------
    results : dict
        Dictionary mapping '{loss_name}_{approach}' to RegistrationResult objects.
        Each result contains the registered image, correlation metrics, and timing.
    
    Examples
    --------
    >>> # Compare loss functions on 2D MIP
    >>> results = compare_loss_functions(moving_mip, fixed_mip)
    >>> for name, result in results.items():
    ...     print(f"{name}: {result.initial_correlation:.4f} -> "
    ...           f"{result.final_correlation:.4f} ({result.time_seconds:.1f}s)")
    
    >>> # Compare only specific losses and approaches
    >>> results = compare_loss_functions(
    ...     moving, fixed,
    ...     loss_names=['correlation', 'monai_ncc'],
    ...     approaches=['multiscale'],
    ... )
    """
    import time
    
    ndim = moving.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Input must be 2D or 3D, got {ndim}D")
    
    # Set defaults based on dimensionality
    if ndim == 2:
        default_spacing_single = 100
        default_spacings_ms = [120, 60]
        default_iters_single = 500
        default_iters_ms = [400, 250]
    else:  # 3D
        default_spacing_single = 100
        default_spacings_ms = [120, 60]
        default_iters_single = 200
        default_iters_ms = [150, 100]
    
    # Apply defaults
    if grid_spacings_single is None:
        grid_spacings_single = default_spacing_single
    if grid_spacings_multiscale is None:
        grid_spacings_multiscale = default_spacings_ms
    if iterations_single is None:
        iterations_single = default_iters_single
    if iterations_multiscale is None:
        iterations_multiscale = default_iters_ms
    
    # Get loss functions
    available_losses = get_available_loss_functions(ndim=ndim)
    if loss_names is None:
        loss_names = list(available_losses.keys())
    
    # Validate loss names
    invalid_losses = set(loss_names) - set(available_losses.keys())
    if invalid_losses:
        raise ValueError(f"Unknown loss functions: {invalid_losses}. "
                        f"Available: {list(available_losses.keys())}")
    
    # Default approaches
    if approaches is None:
        approaches = ['single', 'multiscale']
    
    results = {}
    total_runs = len(loss_names) * len(approaches)
    run_count = 0
    
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    
    for loss_name in loss_names:
        loss_fn = available_losses[loss_name]
        
        for approach in approaches:
            run_count += 1
            key = f"{loss_name}_{approach}"
            
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"[{run_count}/{total_runs}] {loss_name.upper()} + {approach.upper()}")
                print(f"{'=' * 60}")
            
            try:
                t0 = time.time()
                
                if ndim == 2:
                    registered, info = _run_2d_registration(
                        moving, fixed, loss_fn, approach,
                        grid_spacings_single, grid_spacings_multiscale,
                        iterations_single, iterations_multiscale,
                        device, verbose,
                    )
                else:
                    registered, info = _run_3d_registration(
                        moving, fixed, loss_fn, approach,
                        grid_spacings_single, grid_spacings_multiscale,
                        iterations_single, iterations_multiscale,
                        device, verbose,
                    )
                
                elapsed = time.time() - t0
                final_corr = np.corrcoef(registered.flatten(), fixed.flatten())[0, 1]
                
                results[key] = RegistrationResult(
                    loss_name=loss_name,
                    approach=approach,
                    registered=registered,
                    initial_correlation=initial_corr,
                    final_correlation=final_corr,
                    correlation_improvement=final_corr - initial_corr,
                    time_seconds=elapsed,
                    info=info,
                )
                
                if verbose:
                    print(f"\nResult: corr {initial_corr:.4f} -> {final_corr:.4f} "
                          f"(+{final_corr - initial_corr:.4f}) in {elapsed:.1f}s")
                    
            except Exception as e:
                if verbose:
                    print(f"\nERROR: {e}")
                import traceback
                traceback.print_exc()
    
    if verbose:
        _print_summary(results)
    
    return results


def _run_2d_registration(
    moving, fixed, loss_fn, approach,
    grid_spacing_single, grid_spacings_multiscale,
    iterations_single, iterations_multiscale,
    device, verbose,
):
    """Run 2D registration with specified approach."""
    if approach == 'single':
        displacement, _, _, _, info = register_ffd_2d(
            moving, fixed,
            grid_spacing=grid_spacing_single,
            n_iterations=iterations_single,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        registered = apply_ffd_2d(moving, displacement, device=device)
        
    elif approach == 'multiscale':
        displacements, info = multiscale_register_2d(
            moving, fixed,
            grid_spacings=grid_spacings_multiscale,
            iterations_per_level=iterations_multiscale,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        registered = apply_transforms_2d(moving, displacements, device=device)
        
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    return registered, info


def _run_3d_registration(
    moving, fixed, loss_fn, approach,
    grid_spacing_single, grid_spacings_multiscale,
    iterations_single, iterations_multiscale,
    device, verbose,
):
    """Run 3D registration with specified approach."""
    if approach == 'single':
        displacement, _, _, _, info = register_ffd_3d(
            moving, fixed,
            grid_spacing=grid_spacing_single,
            n_iterations=iterations_single,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        registered = apply_ffd_3d(moving, displacement, device=device)
        
    elif approach == 'multiscale':
        displacements, info = multiscale_register_3d(
            moving, fixed,
            grid_spacings=grid_spacings_multiscale,
            iterations_per_level=iterations_multiscale,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        registered = apply_transforms_3d(moving, displacements, device=device)
        
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    return registered, info


def _print_summary(results: Dict[str, RegistrationResult]):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Loss + Approach':<25} {'Initial':>10} {'Final':>10} {'Improve':>10} {'Time':>10}")
    print("-" * 70)
    
    # Sort by final correlation (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].final_correlation,
        reverse=True,
    )
    
    for key, result in sorted_results:
        print(f"{key:<25} {result.initial_correlation:>10.4f} "
              f"{result.final_correlation:>10.4f} "
              f"{result.correlation_improvement:>10.4f} "
              f"{result.time_seconds:>9.1f}s")
    
    print("-" * 70)
    
    # Find best result
    best_key = sorted_results[0][0]
    best_result = sorted_results[0][1]
    print(f"\nBest: {best_key} (corr={best_result.final_correlation:.4f})")
