# -*- coding: utf-8 -*-
"""
Tests for OCTA volume registration.

This test script validates registration on real OCTA data:
1. Single-level registration (2D MIP and 3D volume)
2. Image pyramid registration
3. Control point pyramid (multiscale) registration

Prerequisites:
    Download OCTA data first: python scripts/download_octa_data.py
"""

import sys
import numpy as np
from pathlib import Path
from time import time

# Data paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "octa"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "octa"

# =============================================================================
# Registration Parameters (following convention for OCTA ~512x512x266)
# =============================================================================

# 2D MIP FFD parameters
MIP_2D_GRID_SPACING = 100
MIP_2D_ITERATIONS = 2000
MIP_2D_LR = 0.5
MIP_2D_BENDING_WEIGHT = 0.01
MIP_2D_SMOOTH_WEIGHT = 0.001

# 2D Multiscale parameters (coarse to fine)
MIP_2D_MS_SPACINGS = [200, 100, 50]
MIP_2D_MS_ITERATIONS = [1000, 750, 500]
MIP_2D_MS_SMOOTH = [0.005, 0.002, 0.001]
MIP_2D_MS_BENDING = [0.02, 0.01, 0.005]

# 3D FFD parameters (for subsampled ~256x256x133 volumes)
FFD_3D_GRID_SPACING = 40
FFD_3D_ITERATIONS = 500
FFD_3D_LR = 0.5
FFD_3D_BENDING_WEIGHT = 0.05
FFD_3D_SMOOTH_WEIGHT = 0.05

# 3D Multiscale parameters (coarse to fine, for subsampled volumes)
FFD_3D_MS_SPACINGS = [80, 40, 20]
FFD_3D_MS_ITERATIONS = [300, 200, 150]
FFD_3D_MS_SMOOTH = [0.05, 0.03, 0.01]
FFD_3D_MS_BENDING = [0.05, 0.03, 0.01]


def save_2d_outputs(target, source, registered, test_name):
    """Save 2D test outputs as PNG images."""
    from PIL import Image
    
    output_dir = OUTPUT_DIR / "2d" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_to_uint8(arr):
        """Normalize array to 0-255 uint8."""
        arr = arr.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        return (arr * 255).astype(np.uint8)
    
    Image.fromarray(normalize_to_uint8(target)).save(output_dir / "target.png")
    Image.fromarray(normalize_to_uint8(source)).save(output_dir / "source.png")
    Image.fromarray(normalize_to_uint8(registered)).save(output_dir / "registered.png")
    
    # Also save a difference image
    diff = np.abs(registered - target)
    Image.fromarray(normalize_to_uint8(diff)).save(output_dir / "difference.png")
    
    print(f"   Saved outputs to: {output_dir.relative_to(PROJECT_ROOT)}")


def save_3d_outputs(target, source, registered, test_name):
    """Save 3D test outputs as NIfTI volumes and MIP images."""
    import nibabel as nib
    from PIL import Image
    
    output_dir = OUTPUT_DIR / "3d" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as NIfTI volumes
    nib.save(nib.Nifti1Image(target.astype(np.float32), np.eye(4)), 
             output_dir / "target.nii.gz")
    nib.save(nib.Nifti1Image(source.astype(np.float32), np.eye(4)), 
             output_dir / "source.nii.gz")
    nib.save(nib.Nifti1Image(registered.astype(np.float32), np.eye(4)), 
             output_dir / "registered.nii.gz")
    
    # Also save MIP images for quick visual inspection
    def normalize_to_uint8(arr):
        arr = arr.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        return (arr * 255).astype(np.uint8)
    
    # Save MIPs along Z axis (axis 2 in ZYX ordering)
    target_mip = target.max(axis=2)
    source_mip = source.max(axis=2)
    registered_mip = registered.max(axis=2)
    
    Image.fromarray(normalize_to_uint8(target_mip)).save(output_dir / "target_mip.png")
    Image.fromarray(normalize_to_uint8(source_mip)).save(output_dir / "source_mip.png")
    Image.fromarray(normalize_to_uint8(registered_mip)).save(output_dir / "registered_mip.png")
    
    print(f"   Saved outputs to: {output_dir.relative_to(PROJECT_ROOT)}")


def check_data_available():
    """Check if OCTA data is available."""
    source_file = DATA_DIR / "sourceExample.nii.gz"
    target_file = DATA_DIR / "targetExample.nii.gz"
    
    if not source_file.exists() or not target_file.exists():
        print("OCTA data not found. Please download first:")
        print("  python scripts/download_octa_data.py")
        return False
    return True


def load_octa_data():
    """Load OCTA source and target volumes."""
    import nibabel as nib
    
    source = nib.load(DATA_DIR / "sourceExample.nii.gz").get_fdata()
    target = nib.load(DATA_DIR / "targetExample.nii.gz").get_fdata()
    
    # Convert to float32 for efficiency
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    
    return source, target


def compute_metrics(moving, fixed, warped):
    """Compute registration quality metrics."""
    # Correlation
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    final_corr = np.corrcoef(warped.flatten(), fixed.flatten())[0, 1]
    
    # MAE
    initial_mae = np.abs(moving - fixed).mean()
    final_mae = np.abs(warped - fixed).mean()
    
    return {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'corr_improvement': final_corr - initial_corr,
        'initial_mae': initial_mae,
        'final_mae': final_mae,
        'mae_improvement': initial_mae - final_mae,
    }


# =============================================================================
# 2D MIP Registration Tests
# =============================================================================

def test_octa_2d_single():
    """Test single-level 2D registration on OCTA MIP."""
    from accel_deform_registration import register_ffd_2d, apply_ffd_2d
    
    print("\n" + "=" * 60)
    print("Test: OCTA 2D Single-Level Registration (MIP)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    
    # Create MIPs (maximum intensity projection along Z axis)
    print("2. Creating maximum intensity projections...")
    moving_mip = source.max(axis=2)
    fixed_mip = target.max(axis=2)
    print(f"   MIP shape: {moving_mip.shape}")
    
    # Compute initial metrics
    initial_corr = np.corrcoef(moving_mip.flatten(), fixed_mip.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 2D FFD registration...")
    t0 = time()
    displacement, _, _, _, info = register_ffd_2d(
        moving_mip, fixed_mip,
        grid_spacing=MIP_2D_GRID_SPACING,
        n_iterations=MIP_2D_ITERATIONS,
        lr=MIP_2D_LR,
        smooth_weight=MIP_2D_SMOOTH_WEIGHT,
        bending_weight=MIP_2D_BENDING_WEIGHT,
        verbose=True,
    )
    reg_time = time() - t0
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped_mip = apply_ffd_2d(moving_mip, displacement)
    
    final_corr = np.corrcoef(warped_mip.flatten(), fixed_mip.flatten())[0, 1]
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Time: {reg_time:.1f}s")
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_2d_outputs(fixed_mip, moving_mip, warped_mip, "single")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: 2D MIP registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'time_sec': reg_time,
        'max_displacement': info['max_displacement'],
    }


def test_octa_2d_pyramid():
    """Test image pyramid 2D registration on OCTA MIP."""
    from accel_deform_registration import pyramid_register_2d, apply_ffd_2d
    
    print("\n" + "=" * 60)
    print("Test: OCTA 2D Image Pyramid Registration (MIP)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    
    # Create MIPs
    print("2. Creating maximum intensity projections...")
    moving_mip = source.max(axis=2)
    fixed_mip = target.max(axis=2)
    
    initial_corr = np.corrcoef(moving_mip.flatten(), fixed_mip.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run pyramid registration
    print("\n3. Running pyramid 2D registration...")
    t0 = time()
    displacement, info = pyramid_register_2d(
        moving_mip, fixed_mip,
        levels=[2, 1],  # 2 pyramid levels
        grid_spacings=[MIP_2D_GRID_SPACING, MIP_2D_GRID_SPACING // 2],
        iterations_per_level=[MIP_2D_ITERATIONS, MIP_2D_ITERATIONS // 2],
        lr=MIP_2D_LR,
        smooth_weight=MIP_2D_SMOOTH_WEIGHT,
        bending_weight=MIP_2D_BENDING_WEIGHT,
        verbose=True,
    )
    reg_time = time() - t0
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped_mip = apply_ffd_2d(moving_mip, displacement)
    
    final_corr = np.corrcoef(warped_mip.flatten(), fixed_mip.flatten())[0, 1]
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Time: {reg_time:.1f}s")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_2d_outputs(fixed_mip, moving_mip, warped_mip, "pyramid")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: 2D image pyramid registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'time_sec': reg_time,
        'total_iterations': info['total_iterations'],
    }


def test_octa_2d_multiscale():
    """Test control point pyramid 2D registration on OCTA MIP."""
    from accel_deform_registration import (
        multiscale_register_2d, apply_transforms_2d
    )
    
    print("\n" + "=" * 60)
    print("Test: OCTA 2D Control Point Pyramid Registration (MIP)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    
    # Create MIPs
    print("2. Creating maximum intensity projections...")
    moving_mip = source.max(axis=2)
    fixed_mip = target.max(axis=2)
    
    initial_corr = np.corrcoef(moving_mip.flatten(), fixed_mip.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run multiscale registration
    print("\n3. Running multiscale 2D registration...")
    t0 = time()
    displacements, info = multiscale_register_2d(
        moving_mip, fixed_mip,
        grid_spacings=MIP_2D_MS_SPACINGS,
        iterations_per_level=MIP_2D_MS_ITERATIONS,
        lr=MIP_2D_LR,
        smooth_weights=MIP_2D_MS_SMOOTH,
        bending_weights=MIP_2D_MS_BENDING,
        verbose=True,
    )
    reg_time = time() - t0
    
    # Apply transforms sequentially
    print("\n4. Applying displacement fields sequentially...")
    print(f"   Number of transform levels: {len(displacements)}")
    warped_mip = apply_transforms_2d(moving_mip, displacements)
    
    final_corr = np.corrcoef(warped_mip.flatten(), fixed_mip.flatten())[0, 1]
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Time: {reg_time:.1f}s")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_2d_outputs(fixed_mip, moving_mip, warped_mip, "multiscale")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: 2D control point pyramid registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'time_sec': reg_time,
        'total_iterations': info['total_iterations'],
        'n_levels': len(displacements),
    }


# =============================================================================
# 3D Volume Registration Tests
# =============================================================================

def test_octa_3d_single():
    """Test single-level 3D registration on OCTA volume (subsampled)."""
    from accel_deform_registration import register_ffd_3d, apply_ffd_3d
    
    print("\n" + "=" * 60)
    print("Test: OCTA 3D Single-Level Registration (Subsampled)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    print(f"   Full volume shape: {source.shape}")
    
    moving = source.copy()
    fixed = target.copy()
    print(f"   Subsampled shape: {moving.shape}")
    
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run registration
    print("\n3. Running 3D FFD registration...")
    t0 = time()
    displacement, _, _, _, info = register_ffd_3d(
        moving, fixed,
        grid_spacing=FFD_3D_GRID_SPACING,
        n_iterations=FFD_3D_ITERATIONS,
        lr=FFD_3D_LR,
        smooth_weight=FFD_3D_SMOOTH_WEIGHT,
        bending_weight=FFD_3D_BENDING_WEIGHT,
        verbose=True,
    )
    reg_time = time() - t0
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = np.corrcoef(warped.flatten(), fixed.flatten())[0, 1]
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Time: {reg_time:.1f}s")
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_3d_outputs(fixed, moving, warped, "single")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: 3D single-level registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'time_sec': reg_time,
        'max_displacement': info['max_displacement'],
    }


def test_octa_3d_pyramid():
    """Test image pyramid 3D registration on OCTA volume (subsampled)."""
    from accel_deform_registration import pyramid_register_3d, apply_ffd_3d
    
    print("\n" + "=" * 60)
    print("Test: OCTA 3D Image Pyramid Registration (Subsampled)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    
    moving = source.copy()
    fixed = target.copy()
    print(f"   Subsampled shape: {moving.shape}")
    
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run pyramid registration
    print("\n3. Running pyramid 3D registration...")
    t0 = time()
    displacement, info = pyramid_register_3d(
        moving, fixed,
        levels=[2, 1],
        grid_spacings=[FFD_3D_GRID_SPACING, FFD_3D_GRID_SPACING // 2],
        iterations_per_level=[FFD_3D_ITERATIONS, FFD_3D_ITERATIONS],
        lr=FFD_3D_LR,
        smooth_weight=FFD_3D_SMOOTH_WEIGHT,
        bending_weight=FFD_3D_BENDING_WEIGHT,
        verbose=True,
    )
    reg_time = time() - t0
    
    # Apply registration
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    final_corr = np.corrcoef(warped.flatten(), fixed.flatten())[0, 1]
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Time: {reg_time:.1f}s")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_3d_outputs(fixed, moving, warped, "pyramid")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: 3D image pyramid registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'time_sec': reg_time,
        'total_iterations': info['total_iterations'],
    }


def test_octa_3d_multiscale():
    """Test control point pyramid 3D registration on OCTA volume (subsampled)."""
    from accel_deform_registration import (
        multiscale_register_3d, apply_transforms_3d
    )
    
    print("\n" + "=" * 60)
    print("Test: OCTA 3D Control Point Pyramid Registration (Subsampled)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    
    # Subsample
    moving = source.copy()
    fixed = target.copy()
    print(f"   Subsampled shape: {moving.shape}")
    
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Run multiscale registration
    print("\n3. Running multiscale 3D registration...")
    t0 = time()
    displacements, info = multiscale_register_3d(
        moving, fixed,
        grid_spacings=FFD_3D_MS_SPACINGS,
        iterations_per_level=FFD_3D_MS_ITERATIONS,
        lr=FFD_3D_LR,
        smooth_weights=FFD_3D_MS_SMOOTH,
        bending_weights=FFD_3D_MS_BENDING,
        verbose=True,
    )
    reg_time = time() - t0
    
    # Apply transforms sequentially
    print("\n4. Applying displacement fields sequentially...")
    print(f"   Number of transform levels: {len(displacements)}")
    warped = apply_transforms_3d(moving, displacements)
    
    final_corr = np.corrcoef(warped.flatten(), fixed.flatten())[0, 1]
    print(f"   Final correlation: {final_corr:.4f}")
    print(f"   Improvement: {final_corr - initial_corr:.4f}")
    print(f"   Time: {reg_time:.1f}s")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Save outputs
    print("\n5. Saving outputs...")
    save_3d_outputs(fixed, moving, warped, "multiscale")
    
    # Validate
    success = final_corr > initial_corr
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: 3D control point pyramid registration")
    
    return success, {
        'initial_corr': initial_corr,
        'final_corr': final_corr,
        'time_sec': reg_time,
        'total_iterations': info['total_iterations'],
        'n_levels': len(displacements),
    }


def test_octa_apply_transforms_to_other():
    """Test applying transforms to other volumes (e.g., different channels)."""
    from accel_deform_registration import (
        multiscale_register_3d, apply_transforms_3d
    )
    
    print("\n" + "=" * 60)
    print("Test: Apply Transforms to Other Volumes")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading OCTA data...")
    source, target = load_octa_data()
    
    # Subsample heavily for speed
    moving = source.copy()
    fixed = target.copy()
    print(f"   Subsampled shape: {moving.shape}")
    
    # Create a "simulated other channel" - e.g., inverted version
    other_channel = 1.0 - moving  # Inverted
    
    # Run registration
    print("\n3. Running multiscale 3D registration...")
    displacements, info = multiscale_register_3d(
        moving, fixed,
        grid_spacings=[60, 30],  # Coarser for fast test on 4x subsampled data
        iterations_per_level=[100, 80],
        lr=FFD_3D_LR,
        smooth_weights=[FFD_3D_SMOOTH_WEIGHT, FFD_3D_SMOOTH_WEIGHT],
        bending_weights=[FFD_3D_BENDING_WEIGHT, FFD_3D_BENDING_WEIGHT],
        verbose=True,
    )
    
    # Apply to primary channel
    print("\n4. Applying transforms to primary channel...")
    warped_primary = apply_transforms_3d(moving, displacements)
    
    # Apply same transforms to other channel
    print("5. Applying same transforms to other channel...")
    warped_other = apply_transforms_3d(other_channel, displacements)
    
    # Verify shapes match
    print(f"   Primary warped shape: {warped_primary.shape}")
    print(f"   Other warped shape: {warped_other.shape}")
    
    # Verify the relationship is preserved (inverted should still sum to ~1)
    sum_before = (moving + other_channel).mean()
    sum_after = (warped_primary + warped_other).mean()
    print(f"   Sum before: {sum_before:.4f}")
    print(f"   Sum after: {sum_after:.4f}")
    
    # Validate
    shapes_match = warped_primary.shape == warped_other.shape
    relationship_preserved = abs(sum_after - sum_before) < 0.1  # Allow some interpolation error
    success = shapes_match and relationship_preserved
    
    print(f"\n   TEST {'PASSED' if success else 'FAILED'}: Apply transforms to other volumes")
    
    return success, {
        'shapes_match': shapes_match,
        'relationship_preserved': relationship_preserved,
        'sum_before': sum_before,
        'sum_after': sum_after,
    }


def run_all_tests():
    """Run all OCTA registration tests."""
    if not check_data_available():
        return False, {}
    
    print("\n" + "#" * 60)
    print("# OCTA Registration Tests")
    print("#" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    all_passed = True
    
    # Test 1: 2D single-level
    try:
        success, metrics = test_octa_2d_single()
        results['2d_single'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['2d_single'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 2: 2D image pyramid
    try:
        success, metrics = test_octa_2d_pyramid()
        results['2d_pyramid'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['2d_pyramid'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 3: 2D control point pyramid
    try:
        success, metrics = test_octa_2d_multiscale()
        results['2d_multiscale'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['2d_multiscale'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 4: 3D single-level
    try:
        success, metrics = test_octa_3d_single()
        results['3d_single'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['3d_single'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 5: 3D image pyramid
    try:
        success, metrics = test_octa_3d_pyramid()
        results['3d_pyramid'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['3d_pyramid'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 6: 3D control point pyramid
    try:
        success, metrics = test_octa_3d_multiscale()
        results['3d_multiscale'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['3d_multiscale'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 7: Apply transforms to other volumes
    try:
        success, metrics = test_octa_apply_transforms_to_other()
        results['apply_to_other'] = {'passed': success, **metrics}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['apply_to_other'] = {'passed': False, 'error': str(e)}
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
        if 'initial_corr' in result and 'final_corr' in result:
            print(f"  {name}: {status} (corr: {result['initial_corr']:.4f} -> {result['final_corr']:.4f})")
        else:
            print(f"  {name}: {status}")
    
    return all_passed, results


if __name__ == "__main__":
    all_passed, results = run_all_tests()
    sys.exit(0 if all_passed else 1)
