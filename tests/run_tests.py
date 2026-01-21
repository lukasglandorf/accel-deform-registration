# -*- coding: utf-8 -*-
"""
Comprehensive test runner for accel-deform-registration.

This script runs all tests and outputs visualizations to tests/outputs/:
- 2D images as JPEG
- 3D volumes as NIfTI (.nii.gz)
- Deformable checkerboard patterns (gray checkerboard with deformation applied)
- Control point grids with round nodes (identity, deformed, registered)
- Control point position error metrics

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --losses           # Run only loss tests
    python tests/run_tests.py --2d               # Run only 2D registration tests
    python tests/run_tests.py --3d               # Run only 3D registration tests
    python tests/run_tests.py --pyramid          # Run pyramid registration tests
    python tests/run_tests.py --octa             # Run OCTA data tests (if available)
    python tests/run_tests.py --no-output        # Skip output file generation
    python tests/run_tests.py --repeats 5        # Run with 5 random deformations
    python tests/run_tests.py --size 256         # Use 256x256 phantom size
    python tests/run_tests.py --boundary-layers 2  # Use 2 boundary layers
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

# Ensure package is importable
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Output directory
OUTPUT_DIR = script_dir / "outputs"


def get_output_dir(subdir: str = "") -> Path:
    """Get output directory path, creating if needed."""
    out = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================================
# Loss Function Tests
# ============================================================================

def run_loss_tests() -> Tuple[bool, Dict[str, Any]]:
    """Run all loss function tests with metrics reporting."""
    import torch
    from accel_deform_registration.losses import (
        CorrelationLoss, MAELoss, MutualInformationLoss,
        get_loss_function, check_monai_available
    )
    
    print("\n" + "=" * 70)
    print(" LOSS FUNCTION TESTS")
    print("=" * 70)
    
    results = {}
    all_passed = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    torch.manual_seed(42)
    img1 = torch.rand(1, 1, 128, 128, device=device)
    img2 = torch.rand(1, 1, 128, 128, device=device)
    img1_identical = img1.clone()
    img1_negated = 1.0 - img1
    
    # Test 1: Correlation Loss
    print("\n" + "-" * 50)
    print("1. CorrelationLoss")
    print("-" * 50)
    try:
        t_start = time.time()
        loss_fn = CorrelationLoss()
        
        loss_identical = loss_fn(img1, img1_identical).item()
        loss_different = loss_fn(img1, img2).item()
        loss_negated = loss_fn(img1, img1_negated).item()
        
        img_grad = img1.clone().requires_grad_(True)
        loss = loss_fn(img_grad, img2)
        loss.backward()
        grad_norm = img_grad.grad.norm().item()
        t_elapsed = time.time() - t_start
        
        print(f"   Identical images: {loss_identical:.6f} (expected: -1.0)")
        print(f"   Different images: {loss_different:.6f}")
        print(f"   Negated images:   {loss_negated:.6f} (expected: ~1.0)")
        print(f"   Gradient norm:    {grad_norm:.6f}")
        print(f"   Time:             {t_elapsed*1000:.2f} ms")
        
        passed = abs(loss_identical + 1.0) < 1e-4 and loss_negated > 0.9
        results['correlation'] = {
            'passed': passed,
            'loss_identical': loss_identical,
            'loss_different': loss_different,
            'time_ms': t_elapsed * 1000,
        }
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"   ERROR: {e}")
        results['correlation'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 2: MAE Loss
    print("\n" + "-" * 50)
    print("2. MAELoss")
    print("-" * 50)
    try:
        t_start = time.time()
        loss_fn = MAELoss(normalize=False)
        
        loss_identical = loss_fn(img1, img1_identical).item()
        img_a = torch.ones(1, 1, 64, 64, device=device) * 0.5
        img_b = torch.ones(1, 1, 64, 64, device=device) * 0.7
        loss_known = loss_fn(img_a, img_b).item()
        loss_different = loss_fn(img1, img2).item()
        t_elapsed = time.time() - t_start
        
        print(f"   Identical images: {loss_identical:.6f} (expected: 0.0)")
        print(f"   0.2 difference:   {loss_known:.6f} (expected: 0.2)")
        print(f"   Random images:    {loss_different:.6f}")
        print(f"   Time:             {t_elapsed*1000:.2f} ms")
        
        passed = loss_identical < 1e-6 and abs(loss_known - 0.2) < 1e-4
        results['mae'] = {
            'passed': passed,
            'loss_identical': loss_identical,
            'time_ms': t_elapsed * 1000,
        }
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mae'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 3: MI Histogram
    print("\n" + "-" * 50)
    print("3. MutualInformationLoss (Histogram)")
    print("-" * 50)
    try:
        t_start = time.time()
        loss_fn = MutualInformationLoss(num_bins=32)
        
        loss_identical = loss_fn(img1, img1_identical).item()
        loss_different = loss_fn(img1, img2).item()
        t_elapsed = time.time() - t_start
        
        print(f"   Identical images: {loss_identical:.6f}")
        print(f"   Different images: {loss_different:.6f}")
        print(f"   Time:             {t_elapsed*1000:.2f} ms")
        
        passed = loss_identical < loss_different
        results['mi_histogram'] = {
            'passed': passed,
            'loss_identical': loss_identical,
            'loss_different': loss_different,
            'time_ms': t_elapsed * 1000,
        }
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mi_histogram'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 4: MONAI losses (if available)
    if check_monai_available():
        from accel_deform_registration.losses import MonaiGlobalMILoss
        
        print("\n" + "-" * 50)
        print("4. MonaiGlobalMILoss")
        print("-" * 50)
        try:
            t_start = time.time()
            loss_fn = MonaiGlobalMILoss(num_bins=32)
            
            loss_identical = loss_fn(img1, img1_identical).item()
            loss_different = loss_fn(img1, img2).item()
            t_elapsed = time.time() - t_start
            
            print(f"   Identical images: {loss_identical:.6f}")
            print(f"   Different images: {loss_different:.6f}")
            print(f"   Time:             {t_elapsed*1000:.2f} ms")
            
            results['monai_global_mi'] = {
                'passed': True,
                'loss_identical': loss_identical,
                'loss_different': loss_different,
                'time_ms': t_elapsed * 1000,
            }
            print(f"   STATUS: PASSED")
        except Exception as e:
            print(f"   ERROR: {e}")
            results['monai_global_mi'] = {'passed': False, 'error': str(e)}
            all_passed = False
    else:
        print("\n   MONAI not available - skipping MONAI loss tests")
        results['monai_global_mi'] = {'passed': True, 'skipped': True}
    
    # Test: get_loss_function factory
    print("\n" + "-" * 50)
    print("5. get_loss_function factory")
    print("-" * 50)
    try:
        loss = get_loss_function('correlation')
        assert isinstance(loss, CorrelationLoss)
        print("   'correlation' -> CorrelationLoss ✓")
        
        loss = get_loss_function('mae')
        assert isinstance(loss, MAELoss)
        print("   'mae' -> MAELoss ✓")
        
        loss = get_loss_function('mi')
        assert isinstance(loss, MutualInformationLoss)
        print("   'mi' -> MutualInformationLoss ✓")
        
        results['get_loss_function'] = {'passed': True}
        print(f"   STATUS: PASSED")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['get_loss_function'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    return all_passed, results


# ============================================================================
# 2D Registration Tests
# ============================================================================

def run_2d_single_test(
    target: np.ndarray,
    deformed: np.ndarray,
    true_displacement: np.ndarray,
    loss_name: str,
    loss_fn: Any,
    grid_spacing: int,
    n_iterations: int,
    boundary_layers: int,
    output_dir: Optional[Path],
    save_outputs: bool,
    test_idx: int = 0,
) -> Dict[str, Any]:
    """Run a single 2D registration test with given loss function."""
    from accel_deform_registration import register_ffd_2d, apply_ffd_2d
    from test_output_utils import (
        save_2d_image, create_2d_grid_image_with_circles,
        compute_centered_grid_positions_2d, create_checkerboard_pattern_2d,
        apply_displacement_to_image_2d, compute_control_point_error,
        compute_displacement_field_error,
    )
    
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    def compute_mae(img1, img2):
        return np.mean(np.abs(img1 - img2))
    
    initial_corr = compute_correlation(deformed, target)
    initial_mae = compute_mae(deformed, target)
    
    # Run registration
    t_start = time.time()
    displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_2d(
        deformed, target,
        grid_spacing=grid_spacing,
        n_iterations=n_iterations,
        loss_fn=loss_fn,
        use_boundary_layer=(boundary_layers > 0),
        verbose=(test_idx == 0),  # Only verbose on first run
    )
    t_elapsed = time.time() - t_start
    
    # Apply registration
    registered = apply_ffd_2d(deformed, displacement)
    
    final_corr = compute_correlation(registered, target)
    final_mae = compute_mae(registered, target)
    
    # Compute control point errors
    # The true deformed positions vs estimated positions
    true_deformed_pos = ctrl_pos + true_displacement[
        tuple(np.clip(ctrl_pos[..., 1].astype(int), 0, target.shape[0]-1)),
        tuple(np.clip(ctrl_pos[..., 0].astype(int), 0, target.shape[1]-1)),
    ] if true_displacement is not None else None
    
    # Displacement field error
    disp_error = compute_displacement_field_error(true_displacement, displacement) if true_displacement is not None else {}
    
    result = {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'initial_mae': initial_mae,
        'final_mae': final_mae,
        'time_s': t_elapsed,
        'iterations': info['iterations'],
        'max_displacement': info['max_displacement'],
        'disp_rmse': disp_error.get('rmse', None),
        'disp_mae': disp_error.get('mae', None),
        'passed': final_corr > initial_corr + 0.005,
    }
    
    # Save outputs only on first run
    if save_outputs and output_dir and test_idx == 0:
        suffix = f"_{loss_name}" if loss_name else ""
        
        # Registered image
        save_2d_image(registered, str(output_dir / f"registered{suffix}.jpg"))
        
        # Identity grid
        identity_pos = compute_centered_grid_positions_2d(target.shape, grid_spacing, boundary_layers)
        zero_disps = np.zeros_like(identity_pos)
        grid_identity = create_2d_grid_image_with_circles(
            zero_disps, identity_pos, target.shape,
            point_radius=4, point_value=255, line_value=128
        )
        save_2d_image(grid_identity, str(output_dir / f"grid_identity{suffix}.jpg"), normalize=False)
        
        # Deformed grid (showing the true deformation applied to checkerboard)
        # We need to create a grid showing what the deformation looks like
        # Create grid with the estimated displacements
        grid_deformed = create_2d_grid_image_with_circles(
            ctrl_disps, ctrl_pos, target.shape,
            point_radius=4, point_value=255, line_value=128
        )
        save_2d_image(grid_deformed, str(output_dir / f"grid_registered{suffix}.jpg"), normalize=False)
        
        # Checkerboard with deformation applied
        checker = create_checkerboard_pattern_2d(target.shape, grid_spacing, boundary_layers)
        checker_deformed = apply_displacement_to_image_2d(checker.astype(np.float32), true_displacement)
        save_2d_image(checker_deformed, str(output_dir / f"checkerboard_deformed{suffix}.jpg"))
        
        # Checkerboard with registration (inverse) applied
        checker_registered = apply_displacement_to_image_2d(checker.astype(np.float32), displacement)
        save_2d_image(checker_registered, str(output_dir / f"checkerboard_registered{suffix}.jpg"))
    
    return result


def run_2d_tests(
    save_outputs: bool = True,
    n_repeats: int = 1,
    phantom_size: int = 256,
    boundary_layers: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """Run 2D registration tests with multiple loss functions."""
    from accel_deform_registration import shepp_logan_2d, apply_random_deformation
    from accel_deform_registration.losses import (
        CorrelationLoss, MAELoss, MutualInformationLoss, check_monai_available
    )
    from test_output_utils import (
        save_2d_image, create_checkerboard_pattern_2d, compute_stats,
    )
    
    print("\n" + "=" * 70)
    print(" 2D REGISTRATION TESTS")
    print("=" * 70)
    
    results = {}
    all_passed = True
    
    output_dir = get_output_dir("2d")
    grid_spacing = 50
    n_iterations = 600
    max_displacement = 15.0
    
    # Create phantom
    print(f"\n1. Creating 2D Shepp-Logan phantom ({phantom_size}x{phantom_size})...")
    target = shepp_logan_2d(size=phantom_size)
    
    if save_outputs:
        save_2d_image(target, str(output_dir / "shepp_logan_target.jpg"))
        print(f"   Saved: {output_dir / 'shepp_logan_target.jpg'}")
        
        # Save undeformed checkerboard
        checker = create_checkerboard_pattern_2d(target.shape, grid_spacing, boundary_layers)
        save_2d_image(checker, str(output_dir / "checkerboard_identity.jpg"), normalize=False)
        print(f"   Saved: {output_dir / 'checkerboard_identity.jpg'}")
    
    # Define loss functions to test
    loss_functions = [
        ('correlation', CorrelationLoss()),
        ('mae', MAELoss()),
        ('mi_histogram', MutualInformationLoss(num_bins=32)),
    ]
    
    if check_monai_available():
        from accel_deform_registration.losses import MonaiGlobalMILoss, MonaiLocalNCCLoss
        loss_functions.extend([
            ('monai_global_mi', MonaiGlobalMILoss(num_bins=32)),
            ('monai_lncc', MonaiLocalNCCLoss(spatial_dims=2, kernel_size=9)),
        ])
    
    # Run tests for each loss function
    for loss_name, loss_fn in loss_functions:
        print("\n" + "-" * 50)
        print(f"2D Registration: {loss_name}")
        print("-" * 50)
        
        trial_results = []
        
        for i in range(n_repeats):
            seed = 42 + i
            
            # Apply deformation
            deformed, true_displacement = apply_random_deformation(
                target,
                max_displacement=max_displacement,
                n_control_points=6,
                seed=seed,
            )
            
            if i == 0 and save_outputs:
                save_2d_image(deformed, str(output_dir / f"deformed_{loss_name}.jpg"))
            
            result = run_2d_single_test(
                target, deformed, true_displacement,
                loss_name, loss_fn,
                grid_spacing, n_iterations, boundary_layers,
                output_dir, save_outputs, i
            )
            trial_results.append(result)
            
            if n_repeats == 1:
                print(f"   Initial corr: {result['initial_corr']:.4f}")
                print(f"   Final corr:   {result['final_corr']:.4f} (+{result['final_corr'] - result['initial_corr']:.4f})")
                print(f"   Final MAE:    {result['final_mae']:.4f}")
                print(f"   Time:         {result['time_s']:.2f}s")
                if result['disp_rmse'] is not None:
                    print(f"   Disp RMSE:    {result['disp_rmse']:.4f} px")
        
        # Aggregate results
        if n_repeats > 1:
            corr_improvements = [r['final_corr'] - r['initial_corr'] for r in trial_results]
            times = [r['time_s'] for r in trial_results]
            disp_rmses = [r['disp_rmse'] for r in trial_results if r['disp_rmse'] is not None]
            
            stats = {
                'corr_improvement': compute_stats(corr_improvements),
                'time_s': compute_stats(times),
            }
            if disp_rmses:
                stats['disp_rmse'] = compute_stats(disp_rmses)
            
            print(f"   Corr improvement: {stats['corr_improvement']['mean']:.4f} ± {stats['corr_improvement']['std']:.4f}")
            print(f"   Time:             {stats['time_s']['mean']:.2f}s ± {stats['time_s']['std']:.2f}s")
            if 'disp_rmse' in stats:
                print(f"   Disp RMSE:        {stats['disp_rmse']['mean']:.4f} ± {stats['disp_rmse']['std']:.4f} px")
            
            results[loss_name] = {
                'passed': all(r['passed'] for r in trial_results),
                'stats': stats,
                'n_repeats': n_repeats,
            }
        else:
            results[loss_name] = trial_results[0]
        
        passed = results[loss_name]['passed']
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
    
    return all_passed, results


# ============================================================================
# 3D Registration Tests
# ============================================================================

def run_3d_single_test(
    target: np.ndarray,
    deformed: np.ndarray,
    true_displacement: np.ndarray,
    loss_name: str,
    loss_fn: Any,
    grid_spacing: int,
    n_iterations: int,
    boundary_layers: int,
    output_dir: Optional[Path],
    save_outputs: bool,
    test_idx: int = 0,
) -> Dict[str, Any]:
    """Run a single 3D registration test with given loss function."""
    from accel_deform_registration import register_ffd_3d, apply_ffd_3d
    from test_output_utils import (
        save_2d_image, save_3d_volume, create_3d_grid_image_with_circles,
        compute_centered_grid_positions_3d, create_checkerboard_pattern_3d,
        apply_displacement_to_volume_3d, compute_displacement_field_error,
    )
    
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    def compute_mae(vol1, vol2):
        return np.mean(np.abs(vol1 - vol2))
    
    initial_corr = compute_correlation(deformed, target)
    initial_mae = compute_mae(deformed, target)
    
    # Run registration
    t_start = time.time()
    displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_3d(
        deformed, target,
        grid_spacing=grid_spacing,
        n_iterations=n_iterations,
        loss_fn=loss_fn,
        use_boundary_layer=(boundary_layers > 0),
        verbose=(test_idx == 0),
    )
    t_elapsed = time.time() - t_start
    
    # Apply registration
    registered = apply_ffd_3d(deformed, displacement)
    
    final_corr = compute_correlation(registered, target)
    final_mae = compute_mae(registered, target)
    
    # Displacement field error
    disp_error = compute_displacement_field_error(true_displacement, displacement) if true_displacement is not None else {}
    
    result = {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'initial_mae': initial_mae,
        'final_mae': final_mae,
        'time_s': t_elapsed,
        'iterations': info['iterations'],
        'max_displacement': info['max_displacement'],
        'disp_rmse': disp_error.get('rmse', None),
        'disp_mae': disp_error.get('mae', None),
        'passed': final_corr > initial_corr + 0.003,
    }
    
    # Save outputs only on first run
    if save_outputs and output_dir and test_idx == 0:
        suffix = f"_{loss_name}" if loss_name else ""
        
        # Registered volume
        save_3d_volume(registered, str(output_dir / f"registered{suffix}.nii.gz"))
        save_2d_image(registered.max(axis=0), str(output_dir / f"registered{suffix}_mip.jpg"))
        
        # Grids (MIP)
        identity_pos = compute_centered_grid_positions_3d(target.shape, grid_spacing, boundary_layers)
        zero_disps = np.zeros_like(identity_pos)
        grid_identity = create_3d_grid_image_with_circles(
            zero_disps, identity_pos, target.shape, axis=0,
            point_radius=3, point_value=255, line_value=128
        )
        save_2d_image(grid_identity, str(output_dir / f"grid_identity{suffix}_mip.jpg"), normalize=False)
        
        grid_registered = create_3d_grid_image_with_circles(
            ctrl_disps, ctrl_pos, target.shape, axis=0,
            point_radius=3, point_value=255, line_value=128
        )
        save_2d_image(grid_registered, str(output_dir / f"grid_registered{suffix}_mip.jpg"), normalize=False)
        
        # Checkerboard 3D
        checker = create_checkerboard_pattern_3d(target.shape, grid_spacing, boundary_layers)
        checker_deformed = apply_displacement_to_volume_3d(checker.astype(np.float32), true_displacement)
        save_3d_volume(checker_deformed, str(output_dir / f"checkerboard_deformed{suffix}.nii.gz"))
        
        checker_registered = apply_displacement_to_volume_3d(checker.astype(np.float32), displacement)
        save_3d_volume(checker_registered, str(output_dir / f"checkerboard_registered{suffix}.nii.gz"))
    
    return result


def run_3d_tests(
    save_outputs: bool = True,
    n_repeats: int = 1,
    phantom_size: int = 64,
    boundary_layers: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """Run 3D registration tests with multiple loss functions."""
    from accel_deform_registration import shepp_logan_3d, apply_random_deformation
    from accel_deform_registration.losses import CorrelationLoss, MutualInformationLoss, check_monai_available
    from test_output_utils import (
        save_2d_image, save_3d_volume, create_checkerboard_pattern_3d, compute_stats,
    )
    
    print("\n" + "=" * 70)
    print(" 3D REGISTRATION TESTS")
    print("=" * 70)
    
    results = {}
    all_passed = True
    
    output_dir = get_output_dir("3d")
    grid_spacing = 20
    n_iterations = 250
    max_displacement = 6.0
    
    # Create phantom
    print(f"\n1. Creating 3D Shepp-Logan phantom ({phantom_size}x{phantom_size}x{phantom_size})...")
    target = shepp_logan_3d(size=phantom_size)
    
    if save_outputs:
        save_3d_volume(target, str(output_dir / "shepp_logan_target.nii.gz"))
        save_2d_image(target.max(axis=0), str(output_dir / "shepp_logan_target_mip.jpg"))
        print(f"   Saved: {output_dir / 'shepp_logan_target.nii.gz'}")
        
        # Save undeformed checkerboard
        checker = create_checkerboard_pattern_3d(target.shape, grid_spacing, boundary_layers)
        save_3d_volume(checker, str(output_dir / "checkerboard_identity.nii.gz"))
    
    # Define loss functions to test
    loss_functions = [
        ('correlation', CorrelationLoss()),
        ('mi_histogram', MutualInformationLoss(num_bins=32)),
    ]
    
    if check_monai_available():
        from accel_deform_registration.losses import MonaiGlobalMILoss, MonaiLocalNCCLoss
        loss_functions.extend([
            ('monai_global_mi', MonaiGlobalMILoss(num_bins=32)),
            ('monai_lncc', MonaiLocalNCCLoss(spatial_dims=3, kernel_size=7)),
        ])
    
    for loss_name, loss_fn in loss_functions:
        print("\n" + "-" * 50)
        print(f"3D Registration: {loss_name}")
        print("-" * 50)
        
        trial_results = []
        
        for i in range(n_repeats):
            seed = 42 + i
            
            deformed, true_displacement = apply_random_deformation(
                target,
                max_displacement=max_displacement,
                n_control_points=4,
                seed=seed,
            )
            
            if i == 0 and save_outputs:
                save_3d_volume(deformed, str(output_dir / f"deformed_{loss_name}.nii.gz"))
                save_2d_image(deformed.max(axis=0), str(output_dir / f"deformed_{loss_name}_mip.jpg"))
            
            result = run_3d_single_test(
                target, deformed, true_displacement,
                loss_name, loss_fn,
                grid_spacing, n_iterations, boundary_layers,
                output_dir, save_outputs, i
            )
            trial_results.append(result)
            
            if n_repeats == 1:
                print(f"   Initial corr: {result['initial_corr']:.4f}")
                print(f"   Final corr:   {result['final_corr']:.4f} (+{result['final_corr'] - result['initial_corr']:.4f})")
                print(f"   Time:         {result['time_s']:.2f}s")
                if result['disp_rmse'] is not None:
                    print(f"   Disp RMSE:    {result['disp_rmse']:.4f} voxels")
        
        # Aggregate results
        if n_repeats > 1:
            corr_improvements = [r['final_corr'] - r['initial_corr'] for r in trial_results]
            times = [r['time_s'] for r in trial_results]
            disp_rmses = [r['disp_rmse'] for r in trial_results if r['disp_rmse'] is not None]
            
            stats = {
                'corr_improvement': compute_stats(corr_improvements),
                'time_s': compute_stats(times),
            }
            if disp_rmses:
                stats['disp_rmse'] = compute_stats(disp_rmses)
            
            print(f"   Corr improvement: {stats['corr_improvement']['mean']:.4f} ± {stats['corr_improvement']['std']:.4f}")
            print(f"   Time:             {stats['time_s']['mean']:.2f}s ± {stats['time_s']['std']:.2f}s")
            if 'disp_rmse' in stats:
                print(f"   Disp RMSE:        {stats['disp_rmse']['mean']:.4f} ± {stats['disp_rmse']['std']:.4f} voxels")
            
            results[loss_name] = {
                'passed': all(r['passed'] for r in trial_results),
                'stats': stats,
                'n_repeats': n_repeats,
            }
        else:
            results[loss_name] = trial_results[0]
        
        passed = results[loss_name]['passed']
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
    
    return all_passed, results


# ============================================================================
# Pyramid Registration Tests
# ============================================================================

def run_pyramid_tests(
    save_outputs: bool = True,
    phantom_size_2d: int = 256,
    phantom_size_3d: int = 64,
    boundary_layers: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """Run pyramid (multi-resolution) registration tests."""
    from accel_deform_registration import (
        pyramid_register_2d, pyramid_register_3d, apply_ffd_2d, apply_ffd_3d,
        shepp_logan_2d, shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import CorrelationLoss
    from test_output_utils import save_2d_image, save_3d_volume, compute_displacement_field_error
    
    print("\n" + "=" * 70)
    print(" PYRAMID REGISTRATION TESTS")
    print("=" * 70)
    
    results = {}
    all_passed = True
    
    output_dir = get_output_dir("pyramid")
    
    # 2D Pyramid Test
    print("\n" + "-" * 50)
    print("2D Pyramid Registration")
    print("-" * 50)
    
    try:
        target_2d = shepp_logan_2d(size=phantom_size_2d)
        deformed_2d, true_disp_2d = apply_random_deformation(
            target_2d, max_displacement=20.0, n_control_points=6, seed=42
        )
        
        initial_corr = np.corrcoef(deformed_2d.flatten(), target_2d.flatten())[0, 1]
        
        t_start = time.time()
        displacement_2d, info_2d = pyramid_register_2d(
            deformed_2d, target_2d,
            n_levels=3,
            grid_spacings=[100, 50, 25],
            iterations_per_level=[200, 200, 200],
            loss_fn=CorrelationLoss(),
            verbose=True,
        )
        t_elapsed = time.time() - t_start
        
        registered_2d = apply_ffd_2d(deformed_2d, displacement_2d)
        final_corr = np.corrcoef(registered_2d.flatten(), target_2d.flatten())[0, 1]
        
        disp_error = compute_displacement_field_error(true_disp_2d, displacement_2d)
        
        print(f"\n   Initial corr: {initial_corr:.4f}")
        print(f"   Final corr:   {final_corr:.4f} (+{final_corr - initial_corr:.4f})")
        print(f"   Time:         {t_elapsed:.2f}s")
        print(f"   Disp RMSE:    {disp_error['rmse']:.4f} px")
        
        passed = final_corr > initial_corr + 0.01
        results['pyramid_2d'] = {
            'passed': passed,
            'initial_corr': initial_corr,
            'final_corr': final_corr,
            'time_s': t_elapsed,
            'disp_rmse': disp_error['rmse'],
        }
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
        
        if save_outputs:
            save_2d_image(target_2d, str(output_dir / "target_2d.jpg"))
            save_2d_image(deformed_2d, str(output_dir / "deformed_2d.jpg"))
            save_2d_image(registered_2d, str(output_dir / "registered_2d.jpg"))
    
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['pyramid_2d'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # 3D Pyramid Test
    print("\n" + "-" * 50)
    print("3D Pyramid Registration")
    print("-" * 50)
    
    try:
        target_3d = shepp_logan_3d(size=phantom_size_3d)
        deformed_3d, true_disp_3d = apply_random_deformation(
            target_3d, max_displacement=8.0, n_control_points=4, seed=42
        )
        
        initial_corr = np.corrcoef(deformed_3d.flatten(), target_3d.flatten())[0, 1]
        
        t_start = time.time()
        displacement_3d, info_3d = pyramid_register_3d(
            deformed_3d, target_3d,
            n_levels=2,
            grid_spacings=[30, 15],
            iterations_per_level=[150, 150],
            loss_fn=CorrelationLoss(),
            verbose=True,
        )
        t_elapsed = time.time() - t_start
        
        registered_3d = apply_ffd_3d(deformed_3d, displacement_3d)
        final_corr = np.corrcoef(registered_3d.flatten(), target_3d.flatten())[0, 1]
        
        disp_error = compute_displacement_field_error(true_disp_3d, displacement_3d)
        
        print(f"\n   Initial corr: {initial_corr:.4f}")
        print(f"   Final corr:   {final_corr:.4f} (+{final_corr - initial_corr:.4f})")
        print(f"   Time:         {t_elapsed:.2f}s")
        print(f"   Disp RMSE:    {disp_error['rmse']:.4f} voxels")
        
        passed = final_corr > initial_corr + 0.005
        results['pyramid_3d'] = {
            'passed': passed,
            'initial_corr': initial_corr,
            'final_corr': final_corr,
            'time_s': t_elapsed,
            'disp_rmse': disp_error['rmse'],
        }
        print(f"   STATUS: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
        
        if save_outputs:
            save_3d_volume(target_3d, str(output_dir / "target_3d.nii.gz"))
            save_3d_volume(deformed_3d, str(output_dir / "deformed_3d.nii.gz"))
            save_3d_volume(registered_3d, str(output_dir / "registered_3d.nii.gz"))
            save_2d_image(registered_3d.max(axis=0), str(output_dir / "registered_3d_mip.jpg"))
    
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['pyramid_3d'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    return all_passed, results


# ============================================================================
# OCTA Data Tests
# ============================================================================

def run_octa_tests(
    save_outputs: bool = True,
    boundary_layers: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """Run tests on OCTA data if available."""
    from test_output_utils import check_octa_data_available, load_octa_image, save_2d_image, save_3d_volume
    from accel_deform_registration import register_ffd_2d, register_ffd_3d, apply_ffd_2d, apply_ffd_3d
    from accel_deform_registration.losses import CorrelationLoss
    
    print("\n" + "=" * 70)
    print(" OCTA DATA TESTS")
    print("=" * 70)
    
    available, files = check_octa_data_available()
    
    if not available:
        print("\n   OCTA test data not found.")
        print("   Place your OCTA images in: tests/data/octa/")
        print("   Expected files: target_2d.tif, moving_2d.tif, target_3d.nii.gz, moving_3d.nii.gz")
        return True, {'skipped': True, 'reason': 'No OCTA data found'}
    
    results = {}
    all_passed = True
    output_dir = get_output_dir("octa")
    
    # 2D OCTA Test
    if files['target_2d'] and files['moving_2d']:
        print("\n" + "-" * 50)
        print("2D OCTA Registration")
        print("-" * 50)
        
        try:
            target = load_octa_image(files['target_2d'])
            moving = load_octa_image(files['moving_2d'])
            
            print(f"   Target shape: {target.shape}")
            print(f"   Moving shape: {moving.shape}")
            
            initial_corr = np.corrcoef(moving.flatten(), target.flatten())[0, 1]
            
            t_start = time.time()
            displacement, _, _, _, info = register_ffd_2d(
                moving, target,
                grid_spacing=50,
                n_iterations=500,
                loss_fn=CorrelationLoss(),
                verbose=True,
            )
            t_elapsed = time.time() - t_start
            
            registered = apply_ffd_2d(moving, displacement)
            final_corr = np.corrcoef(registered.flatten(), target.flatten())[0, 1]
            
            print(f"\n   Initial corr: {initial_corr:.4f}")
            print(f"   Final corr:   {final_corr:.4f} (+{final_corr - initial_corr:.4f})")
            print(f"   Time:         {t_elapsed:.2f}s")
            
            results['octa_2d'] = {
                'passed': True,
                'initial_corr': initial_corr,
                'final_corr': final_corr,
                'time_s': t_elapsed,
            }
            print("   STATUS: PASSED")
            
            if save_outputs:
                save_2d_image(target, str(output_dir / "target_2d.jpg"))
                save_2d_image(moving, str(output_dir / "moving_2d.jpg"))
                save_2d_image(registered, str(output_dir / "registered_2d.jpg"))
        
        except Exception as e:
            print(f"   ERROR: {e}")
            results['octa_2d'] = {'passed': False, 'error': str(e)}
            all_passed = False
    
    # 3D OCTA Test
    if files['target_3d'] and files['moving_3d']:
        print("\n" + "-" * 50)
        print("3D OCTA Registration")
        print("-" * 50)
        
        try:
            target = load_octa_image(files['target_3d'])
            moving = load_octa_image(files['moving_3d'])
            
            print(f"   Target shape: {target.shape}")
            print(f"   Moving shape: {moving.shape}")
            
            initial_corr = np.corrcoef(moving.flatten(), target.flatten())[0, 1]
            
            t_start = time.time()
            displacement, _, _, _, info = register_ffd_3d(
                moving, target,
                grid_spacing=30,
                n_iterations=300,
                loss_fn=CorrelationLoss(),
                verbose=True,
            )
            t_elapsed = time.time() - t_start
            
            registered = apply_ffd_3d(moving, displacement)
            final_corr = np.corrcoef(registered.flatten(), target.flatten())[0, 1]
            
            print(f"\n   Initial corr: {initial_corr:.4f}")
            print(f"   Final corr:   {final_corr:.4f} (+{final_corr - initial_corr:.4f})")
            print(f"   Time:         {t_elapsed:.2f}s")
            
            results['octa_3d'] = {
                'passed': True,
                'initial_corr': initial_corr,
                'final_corr': final_corr,
                'time_s': t_elapsed,
            }
            print("   STATUS: PASSED")
            
            if save_outputs:
                save_3d_volume(target, str(output_dir / "target_3d.nii.gz"))
                save_3d_volume(moving, str(output_dir / "moving_3d.nii.gz"))
                save_3d_volume(registered, str(output_dir / "registered_3d.nii.gz"))
        
        except Exception as e:
            print(f"   ERROR: {e}")
            results['octa_3d'] = {'passed': False, 'error': str(e)}
            all_passed = False
    
    return all_passed, results


# ============================================================================
# Summary and Main
# ============================================================================

def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print test summary with metrics."""
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    
    for category, results in all_results.items():
        if not results:
            continue
        
        print(f"\n{category}:")
        for name, result in results.items():
            if result.get('skipped'):
                status = "SKIPPED"
            elif result.get('passed'):
                status = "PASSED"
            else:
                status = "FAILED"
            
            metrics = ""
            if 'initial_corr' in result and 'final_corr' in result:
                metrics = f" (corr: {result['initial_corr']:.3f} -> {result['final_corr']:.3f})"
            elif 'stats' in result and 'corr_improvement' in result['stats']:
                ci = result['stats']['corr_improvement']
                metrics = f" (Δcorr: {ci['mean']:.3f} ± {ci['std']:.3f})"
            
            print(f"  {name:25s}: {status}{metrics}")
    
    # Overall counts
    n_passed = 0
    n_failed = 0
    n_skipped = 0
    
    for results in all_results.values():
        for result in results.values():
            if result.get('skipped'):
                n_skipped += 1
            elif result.get('passed'):
                n_passed += 1
            else:
                n_failed += 1
    
    print("\n" + "-" * 70)
    print(f"TOTAL: {n_passed} passed, {n_failed} failed, {n_skipped} skipped")
    print("-" * 70)
    
    return n_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run accel-deform-registration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                      # Run all tests
  python tests/run_tests.py --losses             # Run only loss tests
  python tests/run_tests.py --2d                 # Run only 2D tests
  python tests/run_tests.py --3d                 # Run only 3D tests
  python tests/run_tests.py --pyramid            # Run pyramid tests
  python tests/run_tests.py --octa               # Run OCTA data tests
  python tests/run_tests.py --no-output          # Skip output file generation
  python tests/run_tests.py --repeats 5          # Run 5 trials with random seeds
  python tests/run_tests.py --size 512           # Use 512x512 2D phantom
  python tests/run_tests.py --size-3d 128        # Use 128x128x128 3D phantom
  python tests/run_tests.py --boundary-layers 2  # Use 2 boundary layers
        """
    )
    parser.add_argument('--losses', action='store_true', help='Run loss function tests')
    parser.add_argument('--2d', dest='run_2d', action='store_true', help='Run 2D registration tests')
    parser.add_argument('--3d', dest='run_3d', action='store_true', help='Run 3D registration tests')
    parser.add_argument('--pyramid', action='store_true', help='Run pyramid registration tests')
    parser.add_argument('--octa', action='store_true', help='Run OCTA data tests')
    parser.add_argument('--no-output', dest='no_output', action='store_true', 
                        help='Skip output file generation')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of random deformation trials (default: 1)')
    parser.add_argument('--size', type=int, default=256,
                        help='2D phantom size in pixels (default: 256)')
    parser.add_argument('--size-3d', type=int, default=64,
                        help='3D phantom size in voxels (default: 64)')
    parser.add_argument('--boundary-layers', type=int, default=1,
                        help='Number of control point boundary layers (default: 1)')
    
    args = parser.parse_args()
    
    # If no specific test requested, run all
    run_all = not (args.losses or args.run_2d or args.run_3d or args.pyramid or args.octa)
    
    save_outputs = not args.no_output
    
    print("\n" + "#" * 70)
    print("#  ACCEL-DEFORM-REGISTRATION TEST SUITE")
    print("#" * 70)
    
    if save_outputs:
        print(f"\nOutput directory: {OUTPUT_DIR}")
    
    print(f"Settings: size_2d={args.size}, size_3d={args.size_3d}, "
          f"repeats={args.repeats}, boundary_layers={args.boundary_layers}")
    
    all_results = {}
    
    if run_all or args.losses:
        _, all_results['Loss Functions'] = run_loss_tests()
    
    if run_all or args.run_2d:
        _, all_results['2D Registration'] = run_2d_tests(
            save_outputs, args.repeats, args.size, args.boundary_layers
        )
    
    if run_all or args.run_3d:
        _, all_results['3D Registration'] = run_3d_tests(
            save_outputs, args.repeats, args.size_3d, args.boundary_layers
        )
    
    if run_all or args.pyramid:
        _, all_results['Pyramid Registration'] = run_pyramid_tests(
            save_outputs, args.size, args.size_3d, args.boundary_layers
        )
    
    if run_all or args.octa:
        _, all_results['OCTA Data'] = run_octa_tests(save_outputs, args.boundary_layers)
    
    success = print_summary(all_results)
    
    if save_outputs:
        print(f"\nTest outputs saved to: {OUTPUT_DIR}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
