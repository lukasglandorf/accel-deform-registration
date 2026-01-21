# -*- coding: utf-8 -*-
"""
Tests for loss functions.

Validates that loss functions:
1. Compute correct values
2. Have proper gradients for optimization
"""

import sys
import numpy as np
import torch


def test_correlation_loss():
    """Test CorrelationLoss computation and gradients."""
    from accel_deform_registration.losses import CorrelationLoss
    
    print("\n" + "=" * 60)
    print("Test: CorrelationLoss")
    print("=" * 60)
    
    loss_fn = CorrelationLoss()
    
    # Test 1: Identical images should have correlation = 1, loss = -1
    print("\n1. Identical images...")
    img = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img, img)
    print(f"   Loss for identical images: {loss.item():.4f} (expected: -1.0)")
    assert abs(loss.item() + 1.0) < 1e-5, "Correlation loss for identical images should be -1"
    
    # Test 2: Negatively correlated images
    print("\n2. Negatively correlated images...")
    img1 = torch.rand(1, 1, 64, 64)
    img2 = 1.0 - img1
    loss = loss_fn(img1, img2)
    print(f"   Loss for negated images: {loss.item():.4f} (expected: ~1.0)")
    assert loss.item() > 0.9, "Correlation loss for negated images should be positive"
    
    # Test 3: Gradient computation
    print("\n3. Gradient computation...")
    img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
    img2 = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img1, img2)
    loss.backward()
    assert img1.grad is not None, "Gradient should be computed"
    print(f"   Gradient norm: {img1.grad.norm().item():.4f}")
    
    # Test 4: With mask
    print("\n4. With mask...")
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0
    img1 = torch.rand(1, 1, 64, 64)
    loss_masked = loss_fn(img1, img1, mask=mask)
    print(f"   Masked loss for identical images: {loss_masked.item():.4f}")
    assert abs(loss_masked.item() + 1.0) < 1e-5
    
    print("\n   TEST PASSED: CorrelationLoss")
    return True


def test_mae_loss():
    """Test MAELoss computation and gradients."""
    from accel_deform_registration.losses import MAELoss
    
    print("\n" + "=" * 60)
    print("Test: MAELoss")
    print("=" * 60)
    
    loss_fn = MAELoss(normalize=False)
    
    # Test 1: Identical images should have MAE = 0
    print("\n1. Identical images...")
    img = torch.rand(1, 1, 64, 64) * 0.5 + 0.25  # Values in [0.25, 0.75]
    loss = loss_fn(img, img)
    print(f"   Loss for identical images: {loss.item():.6f} (expected: 0.0)")
    assert loss.item() < 1e-6, "MAE for identical images should be 0"
    
    # Test 2: Known difference
    print("\n2. Known difference...")
    img1 = torch.ones(1, 1, 64, 64) * 0.5
    img2 = torch.ones(1, 1, 64, 64) * 0.6
    loss = loss_fn(img1, img2)
    print(f"   Loss for 0.1 difference: {loss.item():.4f} (expected: 0.1)")
    assert abs(loss.item() - 0.1) < 1e-5
    
    # Test 3: Gradient computation
    print("\n3. Gradient computation...")
    img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
    img2 = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img1, img2)
    loss.backward()
    assert img1.grad is not None, "Gradient should be computed"
    print(f"   Gradient norm: {img1.grad.norm().item():.4f}")
    
    # Test 4: With mask
    print("\n4. With mask...")
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0
    loss_masked = loss_fn(img, img, mask=mask)
    print(f"   Masked loss for identical images: {loss_masked.item():.6f}")
    assert loss_masked.item() < 1e-6
    
    print("\n   TEST PASSED: MAELoss")
    return True


def test_mi_histogram_loss():
    """Test MutualInformationLoss (histogram method)."""
    from accel_deform_registration.losses import MutualInformationLoss
    
    print("\n" + "=" * 60)
    print("Test: MutualInformationLoss (Histogram)")
    print("=" * 60)
    
    loss_fn = MutualInformationLoss(num_bins=32)
    
    # Test 1: Identical images should have high MI (negative loss)
    print("\n1. Identical images...")
    img = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img, img)
    print(f"   Loss for identical images: {loss.item():.4f}")
    assert loss.item() < 0, "MI loss for identical images should be negative"
    
    # Test 2: Independent images should have lower MI (less negative loss)
    print("\n2. Independent images...")
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 64, 64)
    loss_indep = loss_fn(img1, img2)
    loss_same = loss_fn(img1, img1)
    print(f"   Loss for independent: {loss_indep.item():.4f}")
    print(f"   Loss for identical: {loss_same.item():.4f}")
    assert loss_indep > loss_same, "Independent images should have less MI"
    
    # Test 3: Gradient computation
    print("\n3. Gradient computation...")
    img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
    img2 = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img1, img2)
    loss.backward()
    assert img1.grad is not None, "Gradient should be computed"
    print(f"   Gradient norm: {img1.grad.norm().item():.4f}")
    
    print("\n   TEST PASSED: MutualInformationLoss (Histogram)")
    return True


def test_get_loss_function():
    """Test the get_loss_function convenience function."""
    from accel_deform_registration.losses import get_loss_function, CorrelationLoss, MAELoss, MutualInformationLoss
    
    print("\n" + "=" * 60)
    print("Test: get_loss_function")
    print("=" * 60)
    
    # Test various names
    print("\n1. Testing loss function lookup...")
    
    loss = get_loss_function('correlation')
    assert isinstance(loss, CorrelationLoss)
    print("   'correlation' -> CorrelationLoss ✓")
    
    loss = get_loss_function('mae')
    assert isinstance(loss, MAELoss)
    print("   'mae' -> MAELoss ✓")
    
    loss = get_loss_function('mi')
    assert isinstance(loss, MutualInformationLoss)
    print("   'mi' -> MutualInformationLoss ✓")
    
    loss = get_loss_function('mi_histogram')
    assert isinstance(loss, MutualInformationLoss)
    print("   'mi_histogram' -> MutualInformationLoss ✓")
    
    # Test with kwargs
    print("\n2. Testing with kwargs...")
    loss = get_loss_function('mi', num_bins=64)
    assert loss.num_bins == 64
    print("   'mi' with num_bins=64 ✓")
    
    # Test MONAI losses if available
    from accel_deform_registration.losses import check_monai_available
    if check_monai_available():
        from accel_deform_registration.losses import MonaiLocalNCCLoss
        print("\n3. Testing MONAI losses via get_loss_function...")
        
        loss = get_loss_function('lncc', spatial_dims=2, kernel_size=9)
        assert isinstance(loss, MonaiLocalNCCLoss)
        print("   'lncc' -> MonaiLocalNCCLoss ✓")
        
        loss = get_loss_function('local_ncc', spatial_dims=3, kernel_size=7)
        assert isinstance(loss, MonaiLocalNCCLoss)
        print("   'local_ncc' -> MonaiLocalNCCLoss ✓")
    else:
        print("\n3. MONAI not available, skipping MONAI loss lookup tests")
    
    print("\n   TEST PASSED: get_loss_function")
    return True


def test_monai_global_mi_loss():
    """Test MonaiGlobalMILoss (if MONAI is available)."""
    from accel_deform_registration.losses import check_monai_available
    
    print("\n" + "=" * 60)
    print("Test: MonaiGlobalMILoss")
    print("=" * 60)
    
    if not check_monai_available():
        print("\n   SKIPPED: MONAI not installed")
        return None  # Not a failure, just skipped
    
    from accel_deform_registration.losses import MonaiGlobalMILoss
    
    loss_fn = MonaiGlobalMILoss(num_bins=32)
    
    # Test 1: Identical images should have high MI (negative loss)
    print("\n1. Identical images...")
    img = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img, img)
    print(f"   Loss for identical images: {loss.item():.4f}")
    # MONAI's GMI returns negative MI, so identical images should have negative loss
    
    # Test 2: Independent images
    print("\n2. Independent images...")
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 64, 64)
    loss_indep = loss_fn(img1, img2)
    loss_same = loss_fn(img1, img1)
    print(f"   Loss for independent: {loss_indep.item():.4f}")
    print(f"   Loss for identical: {loss_same.item():.4f}")
    
    # Test 3: Gradient computation
    print("\n3. Gradient computation...")
    img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
    img2 = torch.rand(1, 1, 64, 64)
    loss = loss_fn(img1, img2)
    loss.backward()
    assert img1.grad is not None, "Gradient should be computed"
    print(f"   Gradient norm: {img1.grad.norm().item():.4f}")
    
    # Test 4: GPU test (if available)
    if torch.cuda.is_available():
        print("\n4. GPU test...")
        img1_gpu = torch.rand(1, 1, 64, 64, device='cuda')
        img2_gpu = torch.rand(1, 1, 64, 64, device='cuda')
        loss_gpu = loss_fn(img1_gpu, img2_gpu)
        print(f"   GPU loss: {loss_gpu.item():.4f} ✓")
    else:
        print("\n4. GPU test... SKIPPED (no CUDA)")
    
    print("\n   TEST PASSED: MonaiGlobalMILoss")
    return True


def test_monai_local_ncc_loss():
    """Test MonaiLocalNCCLoss (if MONAI is available)."""
    from accel_deform_registration.losses import check_monai_available
    
    print("\n" + "=" * 60)
    print("Test: MonaiLocalNCCLoss")
    print("=" * 60)
    
    if not check_monai_available():
        print("\n   SKIPPED: MONAI not installed")
        return None  # Not a failure, just skipped
    
    from accel_deform_registration.losses import MonaiLocalNCCLoss
    
    # Test 2D
    print("\n1. Testing 2D with rectangular kernel...")
    loss_fn_2d = MonaiLocalNCCLoss(spatial_dims=2, kernel_size=9)
    
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 64, 64)
    
    loss_same = loss_fn_2d(img1, img1)
    loss_diff = loss_fn_2d(img1, img2)
    print(f"   Loss for identical 2D: {loss_same.item():.4f} (expected: ~ -1)")
    print(f"   Loss for different 2D: {loss_diff.item():.4f} (expected: > -1)")
    assert loss_same.item() < loss_diff.item(), "Identical images should have lower loss"
    
    # Test different kernel types
    print("\n2. Testing different kernel types...")
    for kernel_type in ['rectangular', 'triangular', 'gaussian']:
        loss_fn = MonaiLocalNCCLoss(spatial_dims=2, kernel_size=9, kernel_type=kernel_type)
        loss = loss_fn(img1, img1)
        print(f"   {kernel_type}: {loss.item():.4f}")
    
    # Test 3D
    print("\n3. Testing 3D...")
    loss_fn_3d = MonaiLocalNCCLoss(spatial_dims=3, kernel_size=7)
    
    vol1 = torch.rand(1, 1, 32, 32, 32)
    vol2 = torch.rand(1, 1, 32, 32, 32)
    
    loss_same_3d = loss_fn_3d(vol1, vol1)
    loss_diff_3d = loss_fn_3d(vol1, vol2)
    print(f"   Loss for identical 3D: {loss_same_3d.item():.4f}")
    print(f"   Loss for different 3D: {loss_diff_3d.item():.4f}")
    assert loss_same_3d.item() < loss_diff_3d.item(), "Identical volumes should have lower loss"
    
    # Test gradient
    print("\n4. Gradient computation...")
    img1 = torch.rand(1, 1, 64, 64, requires_grad=True)
    img2 = torch.rand(1, 1, 64, 64)
    loss = loss_fn_2d(img1, img2)
    loss.backward()
    assert img1.grad is not None
    print(f"   Gradient norm: {img1.grad.norm().item():.4f}")
    
    # GPU test
    if torch.cuda.is_available():
        print("\n5. GPU test...")
        img1_gpu = torch.rand(1, 1, 64, 64, device='cuda')
        img2_gpu = torch.rand(1, 1, 64, 64, device='cuda')
        loss_gpu = loss_fn_2d(img1_gpu, img2_gpu)
        print(f"   GPU loss: {loss_gpu.item():.4f} ✓")
    else:
        print("\n5. GPU test... SKIPPED (no CUDA)")
    
    print("\n   TEST PASSED: MonaiLocalNCCLoss")
    return True


def run_all_tests():
    """Run all loss function tests."""
    print("\n" + "#" * 60)
    print("# Loss Function Tests")
    print("#" * 60)
    
    results = {}
    all_passed = True
    
    # Test 1: Correlation
    try:
        success = test_correlation_loss()
        results['correlation'] = {'passed': success}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['correlation'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 2: MAE
    try:
        success = test_mae_loss()
        results['mae'] = {'passed': success}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mae'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 3: MI Histogram
    try:
        success = test_mi_histogram_loss()
        results['mi_histogram'] = {'passed': success}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['mi_histogram'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 4: get_loss_function
    try:
        success = test_get_loss_function()
        results['get_loss_function'] = {'passed': success}
        all_passed = all_passed and success
    except Exception as e:
        print(f"   ERROR: {e}")
        results['get_loss_function'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 5: MONAI Global MI (optional)
    try:
        result = test_monai_global_mi_loss()
        if result is None:
            results['monai_global_mi'] = {'passed': True, 'skipped': True}
        else:
            results['monai_global_mi'] = {'passed': result}
            all_passed = all_passed and result
    except Exception as e:
        print(f"   ERROR: {e}")
        results['monai_global_mi'] = {'passed': False, 'error': str(e)}
        all_passed = False
    
    # Test 6: MONAI Local NCC (optional)
    try:
        result = test_monai_local_ncc_loss()
        if result is None:
            results['monai_local_ncc'] = {'passed': True, 'skipped': True}
        else:
            results['monai_local_ncc'] = {'passed': result}
            all_passed = all_passed and result
    except Exception as e:
        print(f"   ERROR: {e}")
        results['monai_local_ncc'] = {'passed': False, 'error': str(e)}
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
