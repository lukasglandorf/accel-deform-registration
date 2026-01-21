# -*- coding: utf-8 -*-
"""
Demo: 3D Pyramid Registration for OCTA Volumes

This script demonstrates multi-resolution (pyramid) 3D registration
for OCTA volumes, using a coarse-to-fine approach.

Typical workflow:
1. (Optional) 2D MIP registration for initial alignment
2. 3D pyramid registration: 4x → 2x → 1x resolution

Usage:
    python demo_3d_pyramid.py [--data-dir DATA_DIR] [--download]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def download_test_data(data_dir: Path) -> None:
    """Download sample OCTA volume data."""
    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown")
        sys.exit(1)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Replace with actual Google Drive links
    print("Note: Using synthetic 3D data. Replace with actual download links.")
    
    from accel_deform_registration import shepp_logan_3d, apply_random_deformation
    
    # Create synthetic volumes
    print("Generating synthetic 3D phantom (this may take a moment)...")
    fixed = shepp_logan_3d(128)
    moving, _ = apply_random_deformation(fixed, max_displacement=10, seed=42)
    
    np.save(data_dir / "fixed_volume.npy", fixed)
    np.save(data_dir / "moving_volume.npy", moving)
    print(f"Saved synthetic 3D data to {data_dir}")


def load_data(data_dir: Path):
    """Load volumes from data directory."""
    fixed_path = data_dir / "fixed_volume.npy"
    moving_path = data_dir / "moving_volume.npy"
    
    if not fixed_path.exists() or not moving_path.exists():
        print(f"Data not found in {data_dir}")
        print("Generating synthetic data...")
        download_test_data(data_dir)
    
    fixed = np.load(fixed_path)
    moving = np.load(moving_path)
    
    return moving, fixed


def visualize_results_3d(
    moving: np.ndarray,
    fixed: np.ndarray,
    warped: np.ndarray,
    displacement: np.ndarray,
    output_dir: Path,
):
    """Create MIP visualization of 3D registration results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, skipping visualization")
        return
    
    # Compute MIPs along Z axis
    fixed_mip = fixed.max(axis=0)
    moving_mip = moving.max(axis=0)
    warped_mip = warped.max(axis=0)
    
    # Displacement magnitude MIP
    disp_mag = np.sqrt(
        displacement[:, :, :, 0]**2 + 
        displacement[:, :, :, 1]**2 + 
        displacement[:, :, :, 2]**2
    )
    disp_mip = disp_mag.max(axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: MIP images
    axes[0, 0].imshow(fixed_mip, cmap='gray')
    axes[0, 0].set_title('Fixed (MIP)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving_mip, cmap='gray')
    axes[0, 1].set_title('Moving (MIP)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(warped_mip, cmap='gray')
    axes[0, 2].set_title('Warped (MIP)')
    axes[0, 2].axis('off')
    
    # Row 2: Differences and displacement
    diff_before = np.abs(fixed_mip - moving_mip)
    diff_after = np.abs(fixed_mip - warped_mip)
    
    axes[1, 0].imshow(diff_before, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'|Fixed - Moving| MIP\nMAE={np.abs(fixed - moving).mean():.4f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_after, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title(f'|Fixed - Warped| MIP\nMAE={np.abs(fixed - warped).mean():.4f}')
    axes[1, 1].axis('off')
    
    im = axes[1, 2].imshow(disp_mip, cmap='jet')
    axes[1, 2].set_title(f'Displacement Magnitude MIP\nMax={disp_mag.max():.1f}vox')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    output_path = output_dir / "registration_3d_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="3D Pyramid Registration Demo")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download sample data",
    )
    parser.add_argument(
        "--loss",
        choices=["correlation", "mae", "mi"],
        default="correlation",
        help="Loss function to use",
    )
    parser.add_argument(
        "--skip-mip",
        action="store_true",
        help="Skip 2D MIP pre-registration",
    )
    args = parser.parse_args()
    
    # Setup directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.download:
        download_test_data(args.data_dir)
        return
    
    print("=" * 60)
    print("3D Pyramid Registration Demo")
    print("=" * 60)
    
    # Import registration functions
    from accel_deform_registration import (
        register_ffd_2d,
        apply_ffd_2d,
        pyramid_register_3d,
        apply_ffd_3d,
    )
    from accel_deform_registration.losses import get_loss_function
    
    # Load data
    print("\n1. Loading data...")
    moving, fixed = load_data(args.data_dir)
    print(f"   Moving shape: {moving.shape}")
    print(f"   Fixed shape: {fixed.shape}")
    
    # Initial metrics
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Get loss function
    loss_fn = get_loss_function(args.loss)
    
    # Optional: 2D MIP pre-registration
    displacement_2d = None
    if not args.skip_mip:
        print("\n2. Running 2D MIP pre-registration...")
        
        # Compute MIPs
        fixed_mip = fixed.max(axis=0)
        moving_mip = moving.max(axis=0)
        
        displacement_2d, _, _, _, info_2d = register_ffd_2d(
            moving_mip, fixed_mip,
            grid_spacing=40,
            n_iterations=500,
            loss_fn=loss_fn,
            verbose=True,
        )
        
        # Apply 2D displacement to each slice
        print("\n   Applying 2D displacement to volume...")
        moving_prealigned = np.zeros_like(moving)
        for z in range(moving.shape[0]):
            moving_prealigned[z] = apply_ffd_2d(moving[z], displacement_2d)
        
        moving = moving_prealigned
        prealign_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
        print(f"   Correlation after 2D pre-alignment: {prealign_corr:.4f}")
    else:
        print("\n2. Skipping 2D MIP pre-registration")
    
    # 3D Pyramid registration
    print(f"\n3. Running 3D pyramid registration (loss: {loss_fn.name})...")
    
    displacement, info = pyramid_register_3d(
        moving, fixed,
        levels=[2, 1],  # 2x downsampled, then full resolution
        grid_spacings=[30, 20],
        iterations_per_level=[150, 100],
        smooth_weight=0.01,
        bending_weight=0.01,
        loss_fn=loss_fn,
        verbose=True,
    )
    
    # Apply final displacement
    print("\n4. Applying displacement field...")
    warped = apply_ffd_3d(moving, displacement)
    
    # Final metrics
    final_corr = np.corrcoef(warped.flatten(), fixed.flatten())[0, 1]
    print(f"\n5. Results:")
    print(f"   Correlation: {initial_corr:.4f} → {final_corr:.4f}")
    print(f"   Max displacement: {info['max_displacement']:.1f} voxels")
    print(f"   Total iterations: {info['total_iterations']}")
    
    # Save displacement field
    np.save(args.output_dir / "displacement_3d.npy", displacement)
    print(f"\n   Saved displacement field to {args.output_dir / 'displacement_3d.npy'}")
    
    # Save warped volume
    np.save(args.output_dir / "warped_volume.npy", warped)
    print(f"   Saved warped volume to {args.output_dir / 'warped_volume.npy'}")
    
    # Visualize
    print("\n6. Creating visualization...")
    
    # Reload original moving for visualization
    original_moving, _ = load_data(args.data_dir)
    visualize_results_3d(original_moving, fixed, warped, displacement, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
