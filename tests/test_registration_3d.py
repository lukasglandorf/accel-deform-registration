# -*- coding: utf-8 -*-
"""
Tests for 3D FFD registration using synthetic phantoms.

This test script:
1. Creates a 3D Shepp-Logan phantom
2. Applies a known deformation
3. Runs registration to recover the original
4. Validates that the registration converges
"""

import sys
import numpy as np


def test_registration_3d_correlation():
    """Test 3D registration with correlation loss."""
    from accel_deform_registration import (
        register_ffd_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import CorrelationLoss
    
    print("\n" + "=" * 60)
    print("Test: 3D Registration with Correlation Loss")
    print("=" * 60)
    
    # Create phantom (smaller for faster testing)
    print("\n1. Creating 3D Shepp-Logan phantom (64x64x64)...")
    fixed = shepp_logan_3d(size=64)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=6 voxels)...")
    moving, true_displacement = apply_random_deformation(
        fixed, 
        max_displacement=6.0,
        n_control_points=4,
        seed=42,
    )
    
    print(f"   True max displacement: {np.abs(true_displacement).max():.1f} voxels")
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 3D FFD registration...")
    displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=20,
        n_iterations=200,
        loss_fn=CorrelationLoss(),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Correlation improved")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'max_displacement': info['max_displacement'],
    }


def test_registration_3d_mae():
    """Test 3D registration with MAE loss."""
    from accel_deform_registration import (
        register_ffd_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import MAELoss
    
    print("\n" + "=" * 60)
    print("Test: 3D Registration with MAE Loss")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 3D Shepp-Logan phantom (64x64x64)...")
    fixed = shepp_logan_3d(size=64)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=5 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=5.0,
        n_control_points=4,
        seed=123,
    )
    
    # Compute initial MAE
    initial_mae = np.abs(moving - fixed).mean()
    print(f"   Initial MAE: {initial_mae:.4f}")
    
    # Run registration
    print("\n3. Running 3D FFD registration with MAE loss...")
    displacement, _, _, _, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=20,
        n_iterations=200,
        loss_fn=MAELoss(),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_mae = np.abs(warped - fixed).mean()
    print(f"   Final MAE: {final_mae:.4f}")
    print(f"   Improvement: {initial_mae - final_mae:.4f}")
    
    # Validate
    success = final_mae < initial_mae * 0.95  # At least 5% improvement
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: MAE decreased")
    
    return success, {
        'initial_mae': initial_mae,
        'final_mae': final_mae,
    }


def test_registration_3d_mi():
    """Test 3D registration with Mutual Information loss."""
    from accel_deform_registration import (
        register_ffd_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import MutualInformationLoss
    
    print("\n" + "=" * 60)
    print("Test: 3D Registration with Mutual Information")
    print("=" * 60)
    
    # Create phantom (smaller for MI)
    print("\n1. Creating 3D Shepp-Logan phantom (48x48x48)...")
    fixed = shepp_logan_3d(size=48)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=5 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=5.0,
        n_control_points=3,
        seed=456,
    )
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration (histogram MI is faster)
    print("\n3. Running 3D FFD registration with MI (histogram) loss...")
    displacement, _, _, _, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=16,
        n_iterations=150,
        loss_fn=MutualInformationLoss(num_bins=24),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate (MI can be less stable, so smaller threshold)
    success = final_corr >= initial_corr  # At least no degradation
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Registration with MI loss")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_registration_3d_monai_mi():
    """Test 3D registration with MONAI Global Mutual Information loss."""
    from accel_deform_registration import (
        register_ffd_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import check_monai_available
    
    print("\n" + "=" * 60)
    print("Test: 3D Registration with MONAI Global MI")
    print("=" * 60)
    
    if not check_monai_available():
        print("\n   SKIPPED: MONAI not installed")
        return None, {}
    
    from accel_deform_registration.losses import MonaiGlobalMILoss
    
    # Create phantom (smaller for speed)
    print("\n1. Creating 3D Shepp-Logan phantom (48x48x48)...")
    fixed = shepp_logan_3d(size=48)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=5 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=5.0,
        n_control_points=3,
        seed=654,
    )
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 3D FFD registration with MONAI Global MI loss...")
    displacement, _, _, _, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=16,
        n_iterations=150,
        loss_fn=MonaiGlobalMILoss(num_bins=24),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr >= initial_corr  # At least no degradation
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Registration with MONAI MI loss")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_registration_3d_monai_lncc():
    """Test 3D registration with MONAI Local NCC loss."""
    from accel_deform_registration import (
        register_ffd_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import check_monai_available
    
    print("\n" + "=" * 60)
    print("Test: 3D Registration with MONAI Local NCC")
    print("=" * 60)
    
    if not check_monai_available():
        print("\n   SKIPPED: MONAI not installed")
        return None, {}
    
    from accel_deform_registration.losses import MonaiLocalNCCLoss
    
    # Create phantom (smaller for speed)
    print("\n1. Creating 3D Shepp-Logan phantom (48x48x48)...")
    fixed = shepp_logan_3d(size=48)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=5 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=5.0,
        n_control_points=3,
        seed=987,
    )
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 3D FFD registration with MONAI Local NCC loss...")
    displacement, _, _, _, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=16,
        n_iterations=150,
        loss_fn=MonaiLocalNCCLoss(spatial_dims=3, kernel_size=7),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr >= initial_corr  # At least no degradation
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Registration with MONAI LNCC loss")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_registration_3d_with_mask():
    """Test 3D registration with a mask for weighted loss."""
    from accel_deform_registration import (
        register_ffd_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    from accel_deform_registration.losses import CorrelationLoss
    
    print("\n" + "=" * 60)
    print("Test: 3D Registration with Mask")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 3D Shepp-Logan phantom (64x64x64)...")
    fixed = shepp_logan_3d(size=64)
    
    # Create a spherical mask (center region)
    print("2. Creating spherical ROI mask...")
    Z, Y, X = 64, 64, 64
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    center = (Z // 2, Y // 2, X // 2)
    radius = 20
    mask = ((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2) <= radius**2
    mask = mask.astype(np.float32)
    print(f"   Mask coverage: {mask.mean() * 100:.1f}%")
    
    # Apply deformation
    print("3. Applying random deformation (max_disp=6 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=6.0,
        n_control_points=4,
        seed=777,
    )
    
    # Compute initial correlation within mask
    def compute_masked_correlation(vol1, vol2, mask):
        v1 = vol1[mask > 0.5]
        v2 = vol2[mask > 0.5]
        return np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
    
    initial_corr = compute_masked_correlation(moving, fixed, mask)
    print(f"   Initial correlation (masked): {initial_corr:.4f}")
    
    # Run registration with mask
    print("\n4. Running 3D FFD registration with mask...")
    displacement, _, _, _, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=20,
        n_iterations=200,
        mask=mask,
        loss_fn=CorrelationLoss(),
        verbose=True,
    )
    
    # Apply registration
    print("\n5. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = compute_masked_correlation(warped, fixed, mask)
    print(f"   Final correlation (masked): {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Masked registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_pyramid_registration_3d():
    """Test 3D pyramid (multi-resolution) registration."""
    from accel_deform_registration import (
        pyramid_register_3d, apply_ffd_3d,
        shepp_logan_3d, apply_random_deformation
    )
    
    print("\n" + "=" * 60)
    print("Test: 3D Pyramid Registration")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 3D Shepp-Logan phantom (64x64x64)...")
    fixed = shepp_logan_3d(size=64)
    
    # Apply larger deformation
    print("2. Applying random deformation (max_disp=10 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=10.0,
        n_control_points=3,
        seed=888,
    )
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run pyramid registration
    print("\n3. Running pyramid 3D registration...")
    displacement, info = pyramid_register_3d(
        moving, fixed,
        levels=[2, 1],
        grid_spacings=[20, 12],
        iterations_per_level=[100, 80],
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Pyramid registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'total_iterations': info['total_iterations'],
    }


def test_multiscale_registration_3d():
    """Test 3D multiscale (control point pyramid) registration."""
    from accel_deform_registration import (
        multiscale_register_3d, apply_transforms_3d,
        shepp_logan_3d, apply_random_deformation
    )
    
    print("\n" + "=" * 60)
    print("Test: 3D Multiscale Registration (Control Point Pyramid)")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 3D Shepp-Logan phantom (64x64x64)...")
    fixed = shepp_logan_3d(size=64)
    
    # Apply larger deformation
    print("2. Applying random deformation (max_disp=10 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=10.0,
        n_control_points=3,
        seed=888,
    )
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run multiscale registration
    print("\n3. Running multiscale 3D registration...")
    displacements, info = multiscale_register_3d(
        moving, fixed,
        grid_spacings=[40, 20, 12],  # Coarse to fine control points
        iterations_per_level=[150, 100, 75],
        smooth_weights=[0.02, 0.01, 0.005],
        bending_weights=[0.02, 0.01, 0.005],
        verbose=True,
    )
    
    # Apply transforms sequentially
    print("\n4. Applying displacement fields sequentially...")
    print(f"   Number of transform levels: {len(displacements)}")
    warped = apply_transforms_3d(moving, displacements)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Total iterations: {info['total_iterations']}")
    print(f"   Max displacement per level: {info['max_displacements']}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Multiscale registration converged")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'total_iterations': info['total_iterations'],
        'n_levels': len(displacements),
    }


def test_multiscale_per_dimension_3d():
    """Test 3D multiscale registration with per-dimension spacing."""
    from accel_deform_registration import (
        multiscale_register_3d, apply_transforms_3d,
        shepp_logan_3d, apply_random_deformation
    )
    
    print("\n" + "=" * 60)
    print("Test: 3D Multiscale with Per-Dimension Spacing")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 3D Shepp-Logan phantom (64x64x64)...")
    fixed = shepp_logan_3d(size=64)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=8 voxels)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=8.0,
        n_control_points=3,
        seed=123,
    )
    
    # Compute initial correlation
    def compute_correlation(vol1, vol2):
        return np.corrcoef(vol1.flatten(), vol2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run multiscale with per-dimension spacing (Z, Y, X)
    print("\n3. Running multiscale 3D with per-dimension spacing...")
    print("   Using different spacing for Z, Y, X dimensions")
    displacements, info = multiscale_register_3d(
        moving, fixed,
        grid_spacings=[(50, 40, 40), (25, 20, 20), (12, 10, 10)],  # (Z, Y, X) per level
        iterations_per_level=[120, 80, 60],
        verbose=True,
    )
    
    # Apply transforms
    print("\n4. Applying displacement fields...")
    warped = apply_transforms_3d(moving, displacements)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Grid spacings used: {info['grid_spacings']}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Per-dimension multiscale converged")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'grid_spacings': info['grid_spacings'],
    }


def run_all_tests():
    """Run all 3D registration tests."""
    print("\n" + "#" * 60)
    print("# 3D Registration Tests")
    print("#" * 60)
    
    results = {}
    all_passed = True
    
    # Test 1: Correlation loss
    try:
        success, metrics = test_registration_3d_correlation()
        results['correlation'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['correlation'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 2: MAE loss
    try:
        success, metrics = test_registration_3d_mae()
        results['mae'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mae'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 3: MI loss
    try:
        success, metrics = test_registration_3d_mi()
        results['mi'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mi'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 4: MONAI Global MI (optional)
    try:
        result, metrics = test_registration_3d_monai_mi()
        if result is None:
            results['monai_mi'] = {'passed': True, 'skipped': True}
        else:
            results['monai_mi'] = {'passed': result, **metrics}
            all_passed = all_passed and result
    except Exception as e:
        print(f"   ERROR: {e}")
        results['monai_mi'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 5: MONAI Local NCC (optional)
    try:
        result, metrics = test_registration_3d_monai_lncc()
        if result is None:
            results['monai_lncc'] = {'passed': True, 'skipped': True}
        else:
            results['monai_lncc'] = {'passed': result, **metrics}
            all_passed = all_passed and result
    except Exception as e:
        print(f"   ERROR: {e}")
        results['monai_lncc'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 6: Masked registration
    try:
        success, metrics = test_registration_3d_with_mask()
        results['masked'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['masked'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 7: Pyramid registration
    try:
        success, metrics = test_pyramid_registration_3d()
        results['pyramid'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['pyramid'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 8: Multiscale registration
    try:
        success, metrics = test_multiscale_registration_3d()
        results['multiscale'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['multiscale'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 9: Multiscale with per-dimension spacing
    try:
        success, metrics = test_multiscale_per_dimension_3d()
        results['multiscale_per_dim'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['multiscale_per_dim'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Summary
    print("\n" + "#" * 60)
    print("# Summary")
    print("#" * 60)
    n_passed = sum(1 for r in results.values() if r.get('passed', False))
    n_total = len(results)
    print(f"\nTests passed: {n_passed}/{n_total}")
    
    for name, result in results.items():
        status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
        print(f"  {name}: {status}")
    
    return all_passed, results


if __name__ == "__main__":
    all_passed, results = run_all_tests()
    sys.exit(0 if all_passed else 1)
