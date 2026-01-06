import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_data_and_phase(Pwm, Pgm, Pcsf, mask_wt, mask_tc, phi, z_slice=None, save_path=None):
    """Visualize input data and computed phase field."""
    if z_slice is None:
        z_slice = np.argmax(np.sum(mask_wt, axis=(0, 1)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(Pwm[:, :, z_slice], cmap='gray')
    axes[0, 0].set_title('White Matter Prob')

    axes[0, 1].imshow(Pgm[:, :, z_slice], cmap='gray')
    axes[0, 1].set_title('Gray Matter Prob')

    axes[0, 2].imshow(Pcsf[:, :, z_slice], cmap='gray')
    axes[0, 2].set_title('CSF Prob')

    axes[1, 0].imshow(mask_wt[:, :, z_slice], cmap='gray')
    axes[1, 0].set_title('Whole Tumor Mask')

    axes[1, 1].imshow(mask_tc[:, :, z_slice], cmap='gray')
    axes[1, 1].set_title('Tumor Core Mask')

    axes[1, 2].imshow(phi[:, :, z_slice], cmap='gray')
    axes[1, 2].set_title('Phase Field (Phi)')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved data visualization to {save_path}")
    
    # plt.show() # Commented out for script usage, or keep if interactive
    plt.close(fig)

def plot_prediction_contours(Pwm, seg, mask_tc, u_final, z_slice=None, levels=[0.01, 0.35, 0.6], save_path=None):
    """Visualize prediction contours against pre-op segmentation."""
    if z_slice is None:
        # Try to find slice with max tumor
        mask_wt = np.isin(seg, [1, 2, 3])
        z_slice = np.argmax(np.sum(mask_wt, axis=(0, 1)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Background
    ax.imshow(Pwm[:, :, z_slice], cmap='gray', alpha=0.4)

    # Color Edema (Label 2)
    mask_edema = (seg == 2).astype(float)
    # Color Core (Label 1+3)
    mask_core = mask_tc

    # Overlay Edema (Yellowish)
    ax.imshow(mask_edema[:, :, z_slice], cmap='YlOrBr', alpha=0.4 * mask_edema[:, :, z_slice])
    # Overlay Core (Reddish)
    ax.imshow(mask_core[:, :, z_slice], cmap='Reds', alpha=0.6 * mask_core[:, :, z_slice])

    # Contours
    colors = ['cyan', 'blue', 'green']
    # Ensure we have enough colors
    if len(levels) > len(colors):
        colors = plt.cm.jet(np.linspace(0, 1, len(levels)))
        
    contours = ax.contour(u_final[:, :, z_slice], levels=levels, colors=colors[:len(levels)], linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    ax.set_title('Pre-op Seg + Prediction Contours\n(Red: Core, Yellow: Edema)')
    ax.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved prediction contours to {save_path}")
    
    plt.close(fig)

def plot_evaluation(Pwm, standard_plan, model_plan, rec_seg_all, z_slice=None, tumor_threshold=0.0, save_path=None):
    """Visualize evaluation: Standard Plan vs Model Plan vs Recurrence."""
    if z_slice is None:
        z_slice = np.argmax(np.sum(rec_seg_all, axis=(0, 1)))

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Standard Plan
    ax[0].imshow(Pwm[:, :, z_slice], cmap='gray', alpha=0.4)
    ax[0].imshow(standard_plan[:, :, z_slice], cmap='Blues', alpha=0.5)
    ax[0].set_title('Standard Plan (CTV 15mm)')
    ax[0].axis('off')

    # Model Plan
    ax[1].imshow(Pwm[:, :, z_slice], cmap='gray', alpha=0.4)
    ax[1].imshow(model_plan[:, :, z_slice], cmap='Greens', alpha=0.5)
    ax[1].set_title(f'Model Plan (Vol Matched, th={tumor_threshold:.3f})')
    ax[1].axis('off')

    # Recurrence
    ax[2].imshow(Pwm[:, :, z_slice], cmap='gray', alpha=0.4)
    ax[2].imshow(rec_seg_all[:, :, z_slice], cmap='Reds', alpha=0.6)
    ax[2].set_title('Recurrence (All)')
    ax[2].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved evaluation plot to {save_path}")
        
    plt.close(fig)
