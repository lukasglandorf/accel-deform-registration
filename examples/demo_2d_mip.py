# -*- coding: utf-8 -*-
"""
Demo: 2D MIP Registration for OCTA Images

This script demonstrates 2D registration on MIP (Maximum Intensity Projection)
images, typically used as a pre-registration step for 3D OCTA volumes.

Usage:
    python demo_2d_mip.py [--data-dir DATA_DIR] [--download]

If --download is specified, sample data will be downloaded from Google Drive.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def download_test_data(data_dir: Path) -> None:
    """Download sample OCTA data from Google Drive."""
    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown")
        sys.exit(1)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Replace with actual Google Drive links
    # For now, generate synthetic data
    print("Note: Using synthetic data. Replace with actual download links.")
    
    from accel_deform_registration import shepp_logan_2d, apply_random_deformation
    
    # Create synthetic "OCTA-like" images
    fixed = shepp_logan_2d(512)
    moving, _ = apply_random_deformation(fixed, max_displacement=20, seed=42)
    
    np.save(data_dir / "fixed_mip.npy", fixed)
    np.save(data_dir / "moving_mip.npy", moving)
    print(f"Saved synthetic data to {data_dir}")


def load_data(data_dir: Path):
    """Load MIP images from data directory."""
    fixed_path = data_dir / "fixed_mip.npy"
    moving_path = data_dir / "moving_mip.npy"
    
    if not fixed_path.exists() or not moving_path.exists():
        print(f"Data not found in {data_dir}")
        print("Generating synthetic data...")
        download_test_data(data_dir)
    
    fixed = np.load(fixed_path)
    moving = np.load(moving_path)
    
    return moving, fixed


def visualize_results(
    moving: np.ndarray,
    fixed: np.ndarray,
    warped: np.ndarray,
    displacement: np.ndarray,
    output_dir: Path,
):
    """Create visualization of registration results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, skipping visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(fixed, cmap='gray')
    axes[0, 0].set_title('Fixed Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving, cmap='gray')
    axes[0, 1].set_title('Moving Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(warped, cmap='gray')
    axes[0, 2].set_title('Warped (Registered)')
    axes[0, 2].axis('off')
    
    # Row 2: Differences and displacement
    diff_before = np.abs(fixed - moving)
    diff_after = np.abs(fixed - warped)
    
    axes[1, 0].imshow(diff_before, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'|Fixed - Moving|\nMAE={diff_before.mean():.4f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_after, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title(f'|Fixed - Warped|\nMAE={diff_after.mean():.4f}')
    axes[1, 1].axis('off')
    
    # Displacement magnitude
    disp_mag = np.sqrt(displacement[:, :, 0]**2 + displacement[:, :, 1]**2)
    im = axes[1, 2].imshow(disp_mag, cmap='jet')
    axes[1, 2].set_title(f'Displacement Magnitude\nMax={disp_mag.max():.1f}px')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    output_path = output_dir / "registration_2d_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="2D MIP Registration Demo")
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
    args = parser.parse_args()
    
    # Setup directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.download:
        download_test_data(args.data_dir)
        return
    
    print("=" * 60)
    print("2D MIP Registration Demo")
    print("=" * 60)
    
    # Import registration functions
    from accel_deform_registration import (
        register_ffd_2d,
        apply_ffd_2d,
        create_2d_grid_image,
    )
    from accel_deform_registration.losses import get_loss_function
    
    # Load data
    print("\n1. Loading data...")
    moving, fixed = load_data(args.data_dir)
    print(f"   Moving shape: {moving.shape}")
    print(f"   Fixed shape: {fixed.shape}")
    
    # Initial correlation
    initial_corr = np.corrcoef(moving.flatten(), fixed.flatten())[0, 1]
    print(f"   Initial correlation: {initial_corr:.4f}")
    
    # Get loss function
    loss_fn = get_loss_function(args.loss)
    print(f"\n2. Using loss function: {loss_fn.name}")
    
    # Run registration
    print("\n3. Running 2D FFD registration...")
    displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_2d(
        moving, fixed,
        grid_spacing=80,
        smooth_weight=0.005,
        bending_weight=0.01,
        n_iterations=1000,
        loss_fn=loss_fn,
        verbose=True,
    )
    
    # Apply displacement
    print("\n4. Applying displacement field...")
    warped = apply_ffd_2d(moving, displacement)
    
    # Final metrics
    final_corr = np.corrcoef(warped.flatten(), fixed.flatten())[0, 1]
    print(f"\n5. Results:")
    print(f"   Correlation: {initial_corr:.4f} → {final_corr:.4f}")
    print(f"   Max displacement: {info['max_displacement']:.1f} pixels")
    print(f"   Iterations: {info['iterations']}")
    
    # Save displacement field
    np.save(args.output_dir / "displacement_2d.npy", displacement)
    print(f"\n   Saved displacement field to {args.output_dir / 'displacement_2d.npy'}")
    
    # Create grid visualization
    grid_image = create_2d_grid_image(ctrl_disps, ctrl_pos, fixed.shape)
    np.save(args.output_dir / "grid_2d.npy", grid_image)
    
    # Visualize
    print("\n6. Creating visualization...")
    visualize_results(moving, fixed, warped, displacement, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
