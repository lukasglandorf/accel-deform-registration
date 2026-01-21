# -*- coding: utf-8 -*-
"""
Tests for 2D FFD registration using synthetic phantoms.

This test script:
1. Creates a 2D Shepp-Logan phantom
2. Applies a known deformation
3. Runs registration to recover the original
4. Validates that the registration converges
"""

import sys
import numpy as np

# Test with different loss functions
def test_registration_2d_correlation():
    """Test 2D registration with correlation loss."""
    from accel_deform_registration import (
        register_ffd_2d, apply_ffd_2d,
        shepp_logan_2d, apply_random_deformation
    )
    from accel_deform_registration.losses import CorrelationLoss
    
    print("\n" + "=" * 60)
    print("Test: 2D Registration with Correlation Loss")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply deformation to create moving image
    print("2. Applying random deformation (max_disp=12px)...")
    moving, true_displacement = apply_random_deformation(
        fixed, 
        max_displacement=12.0,
        n_control_points=6,
        seed=42,
    )
    
    print(f"   True max displacement: {np.abs(true_displacement).max():.1f} px")
    
    # Compute initial correlation
    def compute_correlation(img1, img2):
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        return np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 2D FFD registration...")
    displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_2d(
        moving, fixed,
        grid_spacing=50,
        n_iterations=500,
        loss_fn=CorrelationLoss(),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr > initial_corr + 0.01
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Correlation improved")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'max_displacement': info['max_displacement'],
    }


def test_registration_2d_mae():
    """Test 2D registration with MAE loss."""
    from accel_deform_registration import (
        register_ffd_2d, apply_ffd_2d,
        shepp_logan_2d, apply_random_deformation
    )
    from accel_deform_registration.losses import MAELoss
    
    print("\n" + "=" * 60)
    print("Test: 2D Registration with MAE Loss")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=10px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=10.0,
        n_control_points=5,
        seed=123,
    )
    
    # Compute initial MAE
    initial_mae = np.abs(moving - fixed).mean()
    print(f"   Initial MAE: {initial_mae:.4f}")
    
    # Run registration
    print("\n3. Running 2D FFD registration with MAE loss...")
    displacement, _, _, _, info = register_ffd_2d(
        moving, fixed,
        grid_spacing=50,
        n_iterations=500,
        loss_fn=MAELoss(),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    final_mae = np.abs(warped - fixed).mean()
    print(f"   Final MAE: {final_mae:.4f}")
    print(f"   Improvement: {initial_mae - final_mae:.4f}")
    
    # Validate
    success = final_mae < initial_mae * 0.9  # At least 10% improvement
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: MAE decreased")
    
    return success, {
        'initial_mae': initial_mae,
        'final_mae': final_mae,
    }


def test_registration_2d_mi_histogram():
    """Test 2D registration with Mutual Information (histogram) loss."""
    from accel_deform_registration import (
        register_ffd_2d, apply_ffd_2d,
        shepp_logan_2d, apply_random_deformation
    )
    from accel_deform_registration.losses import MutualInformationLoss
    
    print("\n" + "=" * 60)
    print("Test: 2D Registration with Mutual Information (Histogram)")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=10px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=10.0,
        n_control_points=5,
        seed=456,
    )
    
    # Compute initial correlation (for validation)
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 2D FFD registration with MI (histogram) loss...")
    displacement, _, _, _, info = register_ffd_2d(
        moving, fixed,
        grid_spacing=50,
        n_iterations=500,
        loss_fn=MutualInformationLoss(num_bins=32),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Correlation improved with MI loss")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_registration_2d_monai_mi():
    """Test 2D registration with MONAI Global Mutual Information loss."""
    from accel_deform_registration import (
        register_ffd_2d, apply_ffd_2d,
        shepp_logan_2d, apply_random_deformation
    )
    from accel_deform_registration.losses import check_monai_available
    
    print("\n" + "=" * 60)
    print("Test: 2D Registration with MONAI Global MI")
    print("=" * 60)
    
    if not check_monai_available():
        print("\n   SKIPPED: MONAI not installed")
        return None, {}
    
    from accel_deform_registration.losses import MonaiGlobalMILoss
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=10px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=10.0,
        n_control_points=5,
        seed=789,
    )
    
    # Compute initial correlation (for validation)
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 2D FFD registration with MONAI Global MI loss...")
    displacement, _, _, _, info = register_ffd_2d(
        moving, fixed,
        grid_spacing=50,
        n_iterations=500,
        loss_fn=MonaiGlobalMILoss(num_bins=32),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Correlation improved with MONAI MI loss")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_registration_2d_monai_lncc():
    """Test 2D registration with MONAI Local NCC loss."""
    from accel_deform_registration import (
        register_ffd_2d, apply_ffd_2d,
        shepp_logan_2d, apply_random_deformation
    )
    from accel_deform_registration.losses import check_monai_available
    
    print("\n" + "=" * 60)
    print("Test: 2D Registration with MONAI Local NCC")
    print("=" * 60)
    
    if not check_monai_available():
        print("\n   SKIPPED: MONAI not installed")
        return None, {}
    
    from accel_deform_registration.losses import MonaiLocalNCCLoss
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=10px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=10.0,
        n_control_points=5,
        seed=321,
    )
    
    # Compute initial correlation (for validation)
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 2D FFD registration with MONAI Local NCC loss...")
    displacement, _, _, _, info = register_ffd_2d(
        moving, fixed,
        grid_spacing=50,
        n_iterations=500,
        loss_fn=MonaiLocalNCCLoss(spatial_dims=2, kernel_size=9),
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    
    # Validate
    success = final_corr > initial_corr + 0.005
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Correlation improved with MONAI LNCC loss")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
    }


def test_pyramid_registration_2d():
    """Test 2D pyramid (multi-resolution) registration."""
    from accel_deform_registration import (
        pyramid_register_2d, apply_ffd_2d,
        shepp_logan_2d, apply_random_deformation
    )
    
    print("\n" + "=" * 60)
    print("Test: 2D Pyramid Registration")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply larger deformation (pyramid handles this better)
    print("2. Applying random deformation (max_disp=20px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=20.0,
        n_control_points=4,
        seed=999,
    )
    
    # Compute initial correlation
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run pyramid registration
    print("\n3. Running pyramid 2D registration...")
    displacement, info = pyramid_register_2d(
        moving, fixed,
        levels=[4, 2, 1],
        grid_spacings=[40, 25, 15],
        iterations_per_level=[200, 150, 100],
        verbose=True,
    )
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Validate
    success = final_corr > initial_corr + 0.01
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Pyramid registration converged")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'total_iterations': info['total_iterations'],
    }


def test_multiscale_registration_2d():
    """Test 2D multiscale (control point pyramid) registration."""
    from accel_deform_registration import (
        multiscale_register_2d, apply_transforms_2d,
        shepp_logan_2d, apply_random_deformation
    )
    
    print("\n" + "=" * 60)
    print("Test: 2D Multiscale Registration (Control Point Pyramid)")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply larger deformation
    print("2. Applying random deformation (max_disp=20px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=20.0,
        n_control_points=4,
        seed=999,
    )
    
    # Compute initial correlation
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run multiscale registration
    print("\n3. Running multiscale 2D registration...")
    displacements, info = multiscale_register_2d(
        moving, fixed,
        grid_spacings=[100, 50, 25],  # Coarse to fine control points
        iterations_per_level=[300, 200, 100],
        smooth_weights=[0.01, 0.005, 0.002],
        bending_weights=[0.02, 0.01, 0.005],
        verbose=True,
    )
    
    # Apply transforms sequentially
    print("\n4. Applying displacement fields sequentially...")
    print(f"   Number of transform levels: {len(displacements)}")
    warped = apply_transforms_2d(moving, displacements)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Total iterations: {info['total_iterations']}")
    print(f"   Max displacement per level: {info['max_displacements']}")
    
    # Validate
    success = final_corr > initial_corr + 0.01
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Multiscale registration converged")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'total_iterations': info['total_iterations'],
        'n_levels': len(displacements),
    }


def test_multiscale_per_dimension_2d():
    """Test 2D multiscale registration with per-dimension spacing."""
    from accel_deform_registration import (
        multiscale_register_2d, apply_transforms_2d,
        shepp_logan_2d, apply_random_deformation
    )
    
    print("\n" + "=" * 60)
    print("Test: 2D Multiscale with Per-Dimension Spacing")
    print("=" * 60)
    
    # Create phantom
    print("\n1. Creating 2D Shepp-Logan phantom (256x256)...")
    fixed = shepp_logan_2d(size=256)
    
    # Apply deformation
    print("2. Applying random deformation (max_disp=15px)...")
    moving, _ = apply_random_deformation(
        fixed, 
        max_displacement=15.0,
        n_control_points=5,
        seed=123,
    )
    
    # Compute initial correlation
    def compute_correlation(img1, img2):
        return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    
    initial_corr = compute_correlation(moving, fixed)
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run multiscale with per-dimension spacing (Y, X)
    print("\n3. Running multiscale 2D with per-dimension spacing...")
    print("   Using different spacing for Y and X dimensions")
    displacements, info = multiscale_register_2d(
        moving, fixed,
        grid_spacings=[(80, 100), (40, 50), (20, 25)],  # (Y, X) per level
        iterations_per_level=[250, 150, 100],
        verbose=True,
    )
    
    # Apply transforms
    print("\n4. Applying displacement fields...")
    warped = apply_transforms_2d(moving, displacements)
    
    final_corr = compute_correlation(warped, fixed)
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Grid spacings used: {info['grid_spacings']}")
    
    # Validate
    success = final_corr > initial_corr + 0.01
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Per-dimension multiscale converged")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'grid_spacings': info['grid_spacings'],
    }


def run_all_tests():
    """Run all 2D registration tests."""
    print("\n" + "#" * 60)
    print("# 2D Registration Tests")
    print("#" * 60)
    
    results = {}
    all_passed = True
    
    # Test 1: Correlation loss
    try:
        success, metrics = test_registration_2d_correlation()
        results['correlation'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['correlation'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 2: MAE loss
    try:
        success, metrics = test_registration_2d_mae()
        results['mae'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mae'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 3: MI histogram
    try:
        success, metrics = test_registration_2d_mi_histogram()
        results['mi_histogram'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mi_histogram'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 4: MONAI Global MI (optional)
    try:
        result, metrics = test_registration_2d_monai_mi()
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
        result, metrics = test_registration_2d_monai_lncc()
        if result is None:
            results['monai_lncc'] = {'passed': True, 'skipped': True}
        else:
            results['monai_lncc'] = {'passed': result, **metrics}
            all_passed = all_passed and result
    except Exception as e:
        print(f"   ERROR: {e}")
        results['monai_lncc'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 6: Pyramid registration
    try:
        success, metrics = test_pyramid_registration_2d()
        results['pyramid'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['pyramid'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 7: Multiscale registration
    try:
        success, metrics = test_multiscale_registration_2d()
        results['multiscale'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['multiscale'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 8: Multiscale with per-dimension spacing
    try:
        success, metrics = test_multiscale_per_dimension_2d()
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
