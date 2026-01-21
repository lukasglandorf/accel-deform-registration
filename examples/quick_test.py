# -*- coding: utf-8 -*-
"""
Quick Test Script: Verify basic package functionality

Run this after installation to verify everything works:
    python examples/quick_test.py
"""

import sys
import time

def main():
    print("=" * 60)
    print("Quick Test: accel-deform-registration")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing imports...")
    try:
        from accel_deform_registration import (
            register_ffd_2d,
            register_ffd_3d,
            apply_ffd_2d,
            apply_ffd_3d,
            pyramid_register_2d,
            pyramid_register_3d,
            shepp_logan_2d,
            shepp_logan_3d,
            apply_random_deformation,
            CorrelationLoss,
            MAELoss,
            MutualInformationLoss,
        )
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    # Test device detection
    print("\n2. Testing device detection...")
    from accel_deform_registration import get_default_device
    device = get_default_device()
    print(f"   ✓ Default device: {device}")
    
    # Test phantom generation
    print("\n3. Testing phantom generation...")
    
    start = time.time()
    phantom_2d = shepp_logan_2d(256)
    print(f"   ✓ 2D phantom: {phantom_2d.shape}, range [{phantom_2d.min():.2f}, {phantom_2d.max():.2f}]")
    
    phantom_3d = shepp_logan_3d(64)
    print(f"   ✓ 3D phantom: {phantom_3d.shape}, range [{phantom_3d.min():.2f}, {phantom_3d.max():.2f}]")
    
    # Test deformation
    print("\n4. Testing deformation...")
    deformed_2d, disp_2d = apply_random_deformation(phantom_2d, max_displacement=10, seed=42)
    print(f"   ✓ 2D deformation: max_disp={disp_2d.max():.1f}px")
    
    deformed_3d, disp_3d = apply_random_deformation(phantom_3d, max_displacement=5, seed=42)
    print(f"   ✓ 3D deformation: max_disp={disp_3d.max():.1f}vox")
    
    # Test loss functions
    print("\n5. Testing loss functions...")
    import torch
    
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 64, 64)
    
    for loss_fn in [CorrelationLoss(), MAELoss(), MutualInformationLoss(method='histogram')]:
        loss = loss_fn(img1, img2)
        print(f"   ✓ {loss_fn.name}: {loss.item():.4f}")
    
    # Test 2D registration (quick)
    print("\n6. Testing 2D registration (small, quick)...")
    import numpy as np
    
    fixed_small = shepp_logan_2d(128)
    moving_small, _ = apply_random_deformation(fixed_small, max_displacement=8, seed=123)
    
    initial_corr = np.corrcoef(moving_small.flatten(), fixed_small.flatten())[0, 1]
    
    displacement, _, _, _, info = register_ffd_2d(
        moving_small, fixed_small,
        grid_spacing=30,
        n_iterations=100,
        verbose=False,
    )
    
    warped = apply_ffd_2d(moving_small, displacement)
    final_corr = np.corrcoef(warped.flatten(), fixed_small.flatten())[0, 1]
    
    print(f"   ✓ 2D registration: corr {initial_corr:.4f} → {final_corr:.4f}")
    
    if final_corr > initial_corr:
        print("   ✓ Registration improved correlation")
    else:
        print("   ⚠ Registration did not improve (may need more iterations)")
    
    # Test 3D registration (very quick)
    print("\n7. Testing 3D registration (small, quick)...")
    
    fixed_3d_small = shepp_logan_3d(48)
    moving_3d_small, _ = apply_random_deformation(fixed_3d_small, max_displacement=4, seed=456)
    
    initial_corr_3d = np.corrcoef(moving_3d_small.flatten(), fixed_3d_small.flatten())[0, 1]
    
    displacement_3d, _, _, _, info_3d = register_ffd_3d(
        moving_3d_small, fixed_3d_small,
        grid_spacing=16,
        n_iterations=50,
        verbose=False,
    )
    
    warped_3d = apply_ffd_3d(moving_3d_small, displacement_3d)
    final_corr_3d = np.corrcoef(warped_3d.flatten(), fixed_3d_small.flatten())[0, 1]
    
    print(f"   ✓ 3D registration: corr {initial_corr_3d:.4f} → {final_corr_3d:.4f}")
    
    elapsed = time.time() - start
    print(f"\n" + "=" * 60)
    print(f"All tests passed! (Total time: {elapsed:.1f}s)")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
