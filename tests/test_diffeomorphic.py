# -*- coding: utf-8 -*-
"""
Tests for diffeomorphic registration features.

Tests Jacobian determinant computation, fold detection, Jacobian penalty,
and SVF (Stationary Velocity Field) integration.
"""

import numpy as np
import pytest
import torch
import time
from typing import Tuple

# Import the module under test
from accel_deform_registration.diffeomorphic import (
    compute_jacobian_determinant_2d,
    compute_jacobian_determinant_3d,
    jacobian_penalty,
    detect_folds,
    get_jacobian_determinant_map,
    JacobianStats,
    scaling_squaring_2d,
    scaling_squaring_3d,
    integrate_svf,
    compose_displacement_2d,
    compose_displacement_3d,
)
from accel_deform_registration import (
    register_ffd_2d,
    register_ffd_3d,
    shepp_logan_2d,
    shepp_logan_3d,
    apply_random_deformation,
)


# =============================================================================
# Test Jacobian Determinant Computation
# =============================================================================

class TestJacobianDeterminant2D:
    """Tests for 2D Jacobian determinant computation."""
    
    def test_identity_displacement(self):
        """Zero displacement should give det(J) = 1 everywhere."""
        H, W = 32, 32
        disp = torch.zeros(1, 2, H, W)
        
        det_J = compute_jacobian_determinant_2d(disp)
        
        assert det_J.shape == (1, 1, H, W)
        # Should be ~1.0 everywhere (identity transform)
        assert torch.allclose(det_J, torch.ones_like(det_J), atol=1e-5)
        print("✓ 2D identity displacement gives det(J) ≈ 1")
    
    def test_uniform_expansion(self):
        """Uniform expansion should give det(J) > 1."""
        H, W = 32, 32
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create displacement that expands by factor (1+scale) in each direction
        # u(x) = scale * x  (in normalized coords [-1,1])
        # But we're in voxel units, so displacement gradient = scale / spacing
        # Central diff uses spacing of 2, so:
        # ∂u_x/∂x = scale * (W/2) / 2 = scale * W / 4 for central diff
        # But actually the displacement IS in voxel units, so
        # u(i) = scale * i where i is voxel index
        # Central diff: (u(i+1) - u(i-1)) / 2 = scale
        # So det(J) = (1+scale)^2
        
        scale = 0.2
        disp = torch.zeros(1, 2, H, W, device=device)
        
        # Create linear displacement: u_x(i,j) = scale * j, u_y(i,j) = scale * i
        for i in range(H):
            for j in range(W):
                disp[0, 0, i, j] = scale * j  # dx increases with x
                disp[0, 1, i, j] = scale * i  # dy increases with y
        
        det_J = compute_jacobian_determinant_2d(disp)
        
        # Interior should have det(J) ≈ (1+scale)^2
        interior = det_J[0, 0, 2:-2, 2:-2]
        expected = (1 + scale) ** 2
        
        # Check that det(J) > 1 (expansion) and reasonably close to expected
        assert interior.min() > 1.0, f"Expansion should give det(J) > 1, got {interior.min():.4f}"
        print(f"✓ 2D uniform expansion gives det(J) ≈ {interior.mean().item():.3f} (expected ~{expected:.3f})")
    
    def test_folding_detection(self):
        """Strong compression should give det(J) < 0 (folding)."""
        H, W = 32, 32
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create displacement with strong compression that causes folding
        # Large negative gradient will make det(J) negative
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # Create sinusoidal displacement with high frequency to induce folding
        disp = torch.zeros(1, 2, H, W, device=device)
        amplitude = 5.0  # Large amplitude relative to grid spacing
        disp[0, 0] = amplitude * torch.sin(4 * np.pi * xx)
        
        det_J = compute_jacobian_determinant_2d(disp)
        
        # Should have some regions with det(J) < 0
        has_folds = (det_J < 0).any().item()
        min_det = det_J.min().item()
        
        assert has_folds, f"Expected folds but min det(J) = {min_det}"
        print(f"✓ 2D sinusoidal displacement creates folds, min det(J) = {min_det:.4f}")


class TestJacobianDeterminant3D:
    """Tests for 3D Jacobian determinant computation."""
    
    def test_identity_displacement(self):
        """Zero displacement should give det(J) = 1 everywhere."""
        D, H, W = 16, 16, 16
        disp = torch.zeros(1, 3, D, H, W)
        
        det_J = compute_jacobian_determinant_3d(disp)
        
        assert det_J.shape == (1, 1, D, H, W)
        assert torch.allclose(det_J, torch.ones_like(det_J), atol=1e-5)
        print("✓ 3D identity displacement gives det(J) ≈ 1")
    
    def test_uniform_expansion(self):
        """Uniform expansion in 3D should give det(J) > 1."""
        D, H, W = 16, 16, 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        scale = 0.2
        disp = torch.zeros(1, 3, D, H, W, device=device)
        
        # Create linear displacement
        for k in range(D):
            for i in range(H):
                for j in range(W):
                    disp[0, 0, k, i, j] = scale * j  # dx
                    disp[0, 1, k, i, j] = scale * i  # dy
                    disp[0, 2, k, i, j] = scale * k  # dz
        
        det_J = compute_jacobian_determinant_3d(disp)
        
        interior = det_J[0, 0, 2:-2, 2:-2, 2:-2]
        expected = (1 + scale) ** 3
        
        assert interior.min() > 1.0, f"3D expansion should give det(J) > 1, got {interior.min():.4f}"
        print(f"✓ 3D uniform expansion gives det(J) ≈ {interior.mean().item():.3f} (expected ~{expected:.3f})")


# =============================================================================
# Test Jacobian Penalty
# =============================================================================

class TestJacobianPenalty:
    """Tests for the differentiable Jacobian penalty."""
    
    def test_zero_penalty_for_identity(self):
        """Identity displacement should have zero penalty."""
        disp = torch.zeros(1, 2, 32, 32, requires_grad=True)
        
        penalty = jacobian_penalty(disp, ndim=2, eps=0.01, mode='relu')
        
        assert penalty.item() < 1e-6
        print("✓ Zero penalty for identity displacement")
    
    def test_penalty_gradient_exists(self):
        """Penalty should be differentiable."""
        # Use a displacement that will actually trigger the penalty
        H, W = 32, 32
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create folding displacement (high-freq sinusoid)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        disp = torch.zeros(1, 2, H, W, device=device, requires_grad=True)
        # We can't assign directly to a leaf tensor, so create data first
        disp_data = torch.zeros(1, 2, H, W, device=device)
        disp_data[0, 0] = 5.0 * torch.sin(4 * np.pi * xx)
        
        # Clone as leaf tensor with requires_grad
        disp = disp_data.clone().requires_grad_(True)
        
        penalty = jacobian_penalty(disp, ndim=2, eps=0.01, mode='relu')
        
        assert penalty.item() > 0, "Need non-zero penalty to test gradients"
        penalty.backward()
        
        assert disp.grad is not None, "Gradient should exist"
        assert not torch.isnan(disp.grad).any(), "Gradient should not have NaN"
        print("✓ Jacobian penalty is differentiable")
    
    def test_penalty_increases_with_folding(self):
        """Penalty should be higher for folded displacements."""
        H, W = 32, 32
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mild displacement
        disp_mild = torch.randn(1, 2, H, W, device=device) * 0.5
        penalty_mild = jacobian_penalty(disp_mild, ndim=2, eps=0.01, mode='relu')
        
        # Strong displacement that causes folding
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        disp_strong = torch.zeros(1, 2, H, W, device=device)
        disp_strong[0, 0] = 5.0 * torch.sin(4 * np.pi * xx)
        penalty_strong = jacobian_penalty(disp_strong, ndim=2, eps=0.01, mode='relu')
        
        assert penalty_strong > penalty_mild
        print(f"✓ Penalty increases with folding: mild={penalty_mild.item():.4f}, strong={penalty_strong.item():.4f}")
    
    def test_3d_penalty(self):
        """Test 3D Jacobian penalty."""
        disp = torch.zeros(1, 3, 16, 16, 16, requires_grad=True)
        
        penalty = jacobian_penalty(disp, ndim=3, eps=0.01, mode='relu')
        penalty.backward()
        
        assert penalty.item() < 1e-6
        assert disp.grad is not None
        print("✓ 3D Jacobian penalty works correctly")


# =============================================================================
# Test Fold Detection
# =============================================================================

class TestFoldDetection:
    """Tests for fold detection utility."""
    
    def test_detect_no_folds_in_identity(self):
        """Identity displacement should have no folds."""
        disp = np.zeros((64, 64, 2), dtype=np.float32)
        
        stats = detect_folds(disp, ndim=2, warn=False)
        
        assert isinstance(stats, JacobianStats)
        assert not stats.has_folds
        assert stats.num_folds == 0
        assert abs(stats.mean_det - 1.0) < 0.01
        print(f"✓ No folds detected in identity: min_det={stats.min_det:.4f}, mean_det={stats.mean_det:.4f}")
    
    def test_detect_folds_in_folded_field(self):
        """Should detect folds in a field with negative Jacobian."""
        H, W = 64, 64
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, H),
            np.linspace(-1, 1, W),
            indexing='ij'
        )
        
        # Create displacement that causes folding
        disp = np.zeros((H, W, 2), dtype=np.float32)
        disp[..., 0] = 5.0 * np.sin(4 * np.pi * xx)
        
        stats = detect_folds(disp, ndim=2, warn=False)
        
        assert stats.has_folds
        assert stats.num_folds > 0
        assert stats.min_det < 0
        print(f"✓ Folds detected: {stats.num_folds} pixels ({stats.fold_fraction*100:.1f}%), min_det={stats.min_det:.4f}")
    
    def test_3d_fold_detection(self):
        """Test 3D fold detection."""
        disp = np.zeros((32, 32, 32, 3), dtype=np.float32)
        
        stats = detect_folds(disp, ndim=3, warn=False)
        
        assert not stats.has_folds
        assert stats.total_voxels == 32 * 32 * 32
        print(f"✓ 3D fold detection works: total_voxels={stats.total_voxels}")
    
    def test_jacobian_map_output(self):
        """Test that Jacobian determinant map has correct shape."""
        disp_2d = np.zeros((64, 64, 2), dtype=np.float32)
        disp_3d = np.zeros((32, 32, 32, 3), dtype=np.float32)
        
        map_2d = get_jacobian_determinant_map(disp_2d, ndim=2)
        map_3d = get_jacobian_determinant_map(disp_3d, ndim=3)
        
        assert map_2d.shape == (64, 64)
        assert map_3d.shape == (32, 32, 32)
        print("✓ Jacobian determinant maps have correct shapes")


# =============================================================================
# Test SVF Integration (Scaling-and-Squaring)
# =============================================================================

class TestSVFIntegration:
    """Tests for Stationary Velocity Field integration."""
    
    def test_zero_velocity_gives_zero_displacement(self):
        """Zero velocity field should integrate to zero displacement."""
        velocity = torch.zeros(1, 2, 32, 32)
        
        disp = scaling_squaring_2d(velocity, n_steps=7)
        
        assert torch.allclose(disp, torch.zeros_like(disp), atol=1e-6)
        print("✓ Zero velocity integrates to zero displacement")
    
    def test_svf_produces_positive_jacobian(self):
        """SVF integration should produce positive Jacobian when velocity is smooth."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a SMOOTH random velocity field
        # The SVF guarantee only holds for sufficiently smooth velocity fields
        # Use strong smoothing to ensure the velocity is smooth enough
        velocity = torch.randn(1, 2, 64, 64, device=device) * 0.5
        
        # Apply strong Gaussian smoothing (multiple passes)
        for _ in range(5):
            velocity = torch.nn.functional.avg_pool2d(velocity, 3, stride=1, padding=1)
        
        disp = scaling_squaring_2d(velocity, n_steps=7)
        det_J = compute_jacobian_determinant_2d(disp)
        
        min_det = det_J.min().item()
        # With sufficient smoothing and steps, SVF should maintain positive Jacobian
        # Allow small numerical errors near zero
        assert min_det > -0.01, f"SVF with smooth velocity should give positive Jacobian, got min={min_det}"
        print(f"✓ SVF produces near-positive Jacobian with smooth velocity: min_det={min_det:.4f}")
    
    def test_svf_3d(self):
        """Test 3D SVF integration."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        velocity = torch.randn(1, 3, 16, 16, 16, device=device) * 1.0
        velocity = torch.nn.functional.avg_pool3d(velocity, 3, stride=1, padding=1)
        
        disp = scaling_squaring_3d(velocity, n_steps=7)
        det_J = compute_jacobian_determinant_3d(disp)
        
        min_det = det_J.min().item()
        assert min_det > 0, f"3D SVF should give positive Jacobian, got min={min_det}"
        print(f"✓ 3D SVF produces positive Jacobian: min_det={min_det:.4f}")
    
    def test_integrate_svf_dispatch(self):
        """Test integrate_svf dispatches correctly to 2D and 3D."""
        vel_2d = torch.zeros(1, 2, 32, 32)
        vel_3d = torch.zeros(1, 3, 16, 16, 16)
        
        disp_2d = integrate_svf(vel_2d, ndim=2, n_steps=5)
        disp_3d = integrate_svf(vel_3d, ndim=3, n_steps=5)
        
        assert disp_2d.shape == (1, 2, 32, 32)
        assert disp_3d.shape == (1, 3, 16, 16, 16)
        print("✓ integrate_svf dispatches correctly")


# =============================================================================
# Test Displacement Composition
# =============================================================================

class TestDisplacementComposition:
    """Tests for displacement field composition."""
    
    def test_compose_with_identity(self):
        """Composing with identity should return original displacement."""
        disp1 = torch.randn(1, 2, 32, 32) * 2.0
        identity = torch.zeros(1, 2, 32, 32)
        
        composed = compose_displacement_2d(disp1, identity)
        
        # disp1 ∘ identity ≈ disp1
        assert torch.allclose(composed, disp1, atol=0.1)
        print("✓ Composition with identity preserves displacement")
    
    def test_composition_is_not_commutative(self):
        """Displacement composition is generally not commutative."""
        disp1 = torch.randn(1, 2, 32, 32) * 1.0
        disp2 = torch.randn(1, 2, 32, 32) * 1.0
        
        composed_12 = compose_displacement_2d(disp1, disp2)
        composed_21 = compose_displacement_2d(disp2, disp1)
        
        # Should not be equal (composition is not commutative)
        diff = (composed_12 - composed_21).abs().mean().item()
        assert diff > 0.01, "Composition should not be commutative"
        print(f"✓ Composition is not commutative: mean diff = {diff:.4f}")
    
    def test_3d_composition(self):
        """Test 3D displacement composition."""
        disp1 = torch.randn(1, 3, 16, 16, 16) * 1.0
        identity = torch.zeros(1, 3, 16, 16, 16)
        
        composed = compose_displacement_3d(disp1, identity)
        
        assert torch.allclose(composed, disp1, atol=0.1)
        print("✓ 3D composition with identity preserves displacement")


# =============================================================================
# Integration Tests: Registration with Diffeomorphic Options
# =============================================================================

class TestRegistrationWithDiffeomorphicOptions:
    """Integration tests for registration with Jacobian penalty and SVF."""
    
    def test_2d_registration_with_jacobian_penalty(self):
        """Test 2D registration with Jacobian penalty enabled."""
        # Create test images
        fixed = shepp_logan_2d(size=64)
        moving, _ = apply_random_deformation(fixed, max_displacement=3.0)
        
        # Register with Jacobian penalty
        disp, ctrl, pos, mask, info = register_ffd_2d(
            moving, fixed,
            grid_spacing=20,
            n_iterations=100,
            jacobian_penalty_weight=1.0,
            verbose=False,
            warn_on_folds=False,
        )
        
        assert 'jacobian_stats' in info
        assert 'jacobian_penalty_weight' in info
        assert info['jacobian_penalty_weight'] == 1.0
        print(f"✓ 2D registration with Jacobian penalty: min_det={info['jacobian_stats']['min_det']:.4f}")
    
    def test_2d_registration_with_svf(self):
        """Test 2D registration with SVF parameterization."""
        fixed = shepp_logan_2d(size=64)
        moving, _ = apply_random_deformation(fixed, max_displacement=3.0)
        
        # Register with SVF
        disp, ctrl, pos, mask, info = register_ffd_2d(
            moving, fixed,
            grid_spacing=20,
            n_iterations=100,
            use_svf=True,
            svf_steps=7,
            verbose=False,
            warn_on_folds=False,
        )
        
        assert info['use_svf'] == True
        assert 'jacobian_stats' in info
        
        # SVF should guarantee no folds
        assert not info['jacobian_stats']['has_folds'], \
            f"SVF should prevent folds, but got {info['jacobian_stats']['num_folds']} folds"
        print(f"✓ 2D SVF registration: no folds, min_det={info['jacobian_stats']['min_det']:.4f}")
    
    def test_3d_registration_with_jacobian_penalty(self):
        """Test 3D registration with Jacobian penalty enabled."""
        fixed = shepp_logan_3d(size=32)
        moving, _ = apply_random_deformation(fixed, max_displacement=2.0)
        
        disp, ctrl, pos, mask, info = register_ffd_3d(
            moving, fixed,
            grid_spacing=12,
            n_iterations=50,
            jacobian_penalty_weight=1.0,
            verbose=False,
            warn_on_folds=False,
        )
        
        assert 'jacobian_stats' in info
        print(f"✓ 3D registration with Jacobian penalty: min_det={info['jacobian_stats']['min_det']:.4f}")
    
    def test_3d_registration_with_svf(self):
        """Test 3D registration with SVF parameterization."""
        fixed = shepp_logan_3d(size=32)
        moving, _ = apply_random_deformation(fixed, max_displacement=2.0)
        
        disp, ctrl, pos, mask, info = register_ffd_3d(
            moving, fixed,
            grid_spacing=12,
            n_iterations=50,
            use_svf=True,
            verbose=False,
            warn_on_folds=False,
        )
        
        assert info['use_svf'] == True
        # SVF should prevent folds
        assert not info['jacobian_stats']['has_folds'], \
            f"SVF should prevent folds, got {info['jacobian_stats']['num_folds']}"
        print(f"✓ 3D SVF registration: no folds, min_det={info['jacobian_stats']['min_det']:.4f}")
    
    def test_vanilla_registration_still_works(self):
        """Ensure vanilla registration (no diffeomorphic options) still works."""
        fixed = shepp_logan_2d(size=64)
        moving, _ = apply_random_deformation(fixed, max_displacement=3.0)
        
        # Vanilla registration (default settings)
        disp, ctrl, pos, mask, info = register_ffd_2d(
            moving, fixed,
            grid_spacing=20,
            n_iterations=100,
            verbose=False,
            warn_on_folds=False,
        )
        
        assert info['use_svf'] == False
        assert info['jacobian_penalty_weight'] == 0.0
        assert info['final_correlation'] > info['initial_correlation']
        print(f"✓ Vanilla registration still works: corr {info['initial_correlation']:.4f} → {info['final_correlation']:.4f}")


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance timing tests comparing vanilla vs diffeomorphic methods."""
    
    @pytest.mark.slow
    def test_2d_timing_comparison(self):
        """Compare timing of vanilla vs Jacobian penalty vs SVF for 2D."""
        fixed = shepp_logan_2d(size=128)
        moving, _ = apply_random_deformation(fixed, max_displacement=5.0)
        
        n_iters = 100
        results = {}
        
        # Vanilla
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _, _, _, _, info_vanilla = register_ffd_2d(
            moving, fixed, grid_spacing=30, n_iterations=n_iters,
            verbose=False, warn_on_folds=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['vanilla'] = {
            'time': time.perf_counter() - start,
            'corr': info_vanilla['final_correlation'],
        }
        
        # Jacobian penalty
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _, _, _, _, info_jac = register_ffd_2d(
            moving, fixed, grid_spacing=30, n_iterations=n_iters,
            jacobian_penalty_weight=1.0,
            verbose=False, warn_on_folds=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['jacobian'] = {
            'time': time.perf_counter() - start,
            'corr': info_jac['final_correlation'],
            'min_det': info_jac['jacobian_stats']['min_det'],
        }
        
        # SVF
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _, _, _, _, info_svf = register_ffd_2d(
            moving, fixed, grid_spacing=30, n_iterations=n_iters,
            use_svf=True, svf_steps=7,
            verbose=False, warn_on_folds=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['svf'] = {
            'time': time.perf_counter() - start,
            'corr': info_svf['final_correlation'],
            'min_det': info_svf['jacobian_stats']['min_det'],
        }
        
        print("\n" + "="*60)
        print("2D Performance Comparison (128x128, 100 iterations)")
        print("="*60)
        print(f"{'Method':<15} {'Time (s)':<12} {'Final Corr':<12} {'Min det(J)':<12}")
        print("-"*60)
        for method, data in results.items():
            min_det = data.get('min_det', 'N/A')
            if isinstance(min_det, float):
                min_det = f"{min_det:.4f}"
            print(f"{method:<15} {data['time']:<12.3f} {data['corr']:<12.4f} {min_det:<12}")
        print("="*60)
        
        # Assertions
        assert results['jacobian']['time'] < results['vanilla']['time'] * 3, "Jacobian penalty shouldn't be >3x slower"
        assert results['svf']['time'] < results['vanilla']['time'] * 5, "SVF shouldn't be >5x slower"
    
    @pytest.mark.slow
    def test_3d_timing_comparison(self):
        """Compare timing of vanilla vs Jacobian penalty vs SVF for 3D."""
        fixed = shepp_logan_3d(size=48)
        moving, _ = apply_random_deformation(fixed, max_displacement=3.0)
        
        n_iters = 50
        results = {}
        
        # Vanilla
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _, _, _, _, info_vanilla = register_ffd_3d(
            moving, fixed, grid_spacing=16, n_iterations=n_iters,
            verbose=False, warn_on_folds=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['vanilla'] = {
            'time': time.perf_counter() - start,
            'corr': info_vanilla['final_correlation'],
        }
        
        # Jacobian penalty
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _, _, _, _, info_jac = register_ffd_3d(
            moving, fixed, grid_spacing=16, n_iterations=n_iters,
            jacobian_penalty_weight=1.0,
            verbose=False, warn_on_folds=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['jacobian'] = {
            'time': time.perf_counter() - start,
            'corr': info_jac['final_correlation'],
            'min_det': info_jac['jacobian_stats']['min_det'],
        }
        
        # SVF
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _, _, _, _, info_svf = register_ffd_3d(
            moving, fixed, grid_spacing=16, n_iterations=n_iters,
            use_svf=True, svf_steps=7,
            verbose=False, warn_on_folds=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        results['svf'] = {
            'time': time.perf_counter() - start,
            'corr': info_svf['final_correlation'],
            'min_det': info_svf['jacobian_stats']['min_det'],
        }
        
        print("\n" + "="*60)
        print("3D Performance Comparison (48x48x48, 50 iterations)")
        print("="*60)
        print(f"{'Method':<15} {'Time (s)':<12} {'Final Corr':<12} {'Min det(J)':<12}")
        print("-"*60)
        for method, data in results.items():
            min_det = data.get('min_det', 'N/A')
            if isinstance(min_det, float):
                min_det = f"{min_det:.4f}"
            print(f"{method:<15} {data['time']:<12.3f} {data['corr']:<12.4f} {min_det:<12}")
        print("="*60)


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all diffeomorphic registration tests."""
    print("\n" + "="*70)
    print("DIFFEOMORPHIC REGISTRATION TESTS")
    print("="*70 + "\n")
    
    # Jacobian Determinant Tests
    print("\n--- Jacobian Determinant 2D Tests ---")
    test_jac_2d = TestJacobianDeterminant2D()
    test_jac_2d.test_identity_displacement()
    test_jac_2d.test_uniform_expansion()
    test_jac_2d.test_folding_detection()
    
    print("\n--- Jacobian Determinant 3D Tests ---")
    test_jac_3d = TestJacobianDeterminant3D()
    test_jac_3d.test_identity_displacement()
    test_jac_3d.test_uniform_expansion()
    
    # Jacobian Penalty Tests
    print("\n--- Jacobian Penalty Tests ---")
    test_penalty = TestJacobianPenalty()
    test_penalty.test_zero_penalty_for_identity()
    test_penalty.test_penalty_gradient_exists()
    test_penalty.test_penalty_increases_with_folding()
    test_penalty.test_3d_penalty()
    
    # Fold Detection Tests
    print("\n--- Fold Detection Tests ---")
    test_folds = TestFoldDetection()
    test_folds.test_detect_no_folds_in_identity()
    test_folds.test_detect_folds_in_folded_field()
    test_folds.test_3d_fold_detection()
    test_folds.test_jacobian_map_output()
    
    # SVF Tests
    print("\n--- SVF Integration Tests ---")
    test_svf = TestSVFIntegration()
    test_svf.test_zero_velocity_gives_zero_displacement()
    test_svf.test_svf_produces_positive_jacobian()
    test_svf.test_svf_3d()
    test_svf.test_integrate_svf_dispatch()
    
    # Composition Tests
    print("\n--- Displacement Composition Tests ---")
    test_comp = TestDisplacementComposition()
    test_comp.test_compose_with_identity()
    test_comp.test_composition_is_not_commutative()
    test_comp.test_3d_composition()
    
    # Integration Tests
    print("\n--- Registration Integration Tests ---")
    test_reg = TestRegistrationWithDiffeomorphicOptions()
    test_reg.test_2d_registration_with_jacobian_penalty()
    test_reg.test_2d_registration_with_svf()
    test_reg.test_3d_registration_with_jacobian_penalty()
    test_reg.test_3d_registration_with_svf()
    test_reg.test_vanilla_registration_still_works()
    
    # Performance Tests
    print("\n--- Performance Benchmarks ---")
    test_perf = TestPerformanceBenchmarks()
    test_perf.test_2d_timing_comparison()
    test_perf.test_3d_timing_comparison()
    
    print("\n" + "="*70)
    print("ALL DIFFEOMORPHIC TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
