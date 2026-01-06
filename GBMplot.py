import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from scipy.interpolate import griddata
from util import error_logging_decorator
'''
Plotting utilities for GBM problems
Assuming X has columns [t, x, y, z] for unstructured data
'''


@error_logging_decorator
def plot_grid_imshow_panels(
    pred2d,
    ref2d=None,
    *,
    fname='fig_grid_panels.png',
    savedir=None,
    title='',
    cmap='viridis',
):
    """Plot grid prediction using imshow.

    Args:
        pred2d: (nx, ny) array
        ref2d:  (nx, ny) array or None
    Notes:
        Uses ndgrid-like orientation: x down, y right (invert y-axis).
    """
    pred2d = np.asarray(pred2d)
    if ref2d is not None:
        ref2d = np.asarray(ref2d)
        vmin = float(np.min([pred2d.min(), ref2d.min()]))
        vmax = float(np.max([pred2d.max(), ref2d.max()]))
        err2d = np.abs(pred2d - ref2d)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ims = []
        ims.append(axes[0].imshow(pred2d, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='equal'))
        axes[0].set_title('pred')
        ims.append(axes[1].imshow(ref2d, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='equal'))
        axes[1].set_title('data')
        ims.append(axes[2].imshow(err2d, cmap=cmap, origin='lower', aspect='equal'))
        axes[2].set_title('abs error')

        for ax in axes:
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            ax.invert_yaxis()
        fig.suptitle(title)

        # colorbars
        fig.colorbar(ims[0], ax=axes[:2], fraction=0.046, pad=0.04, label='u (shared)')
        fig.colorbar(ims[2], ax=axes[2], fraction=0.046, pad=0.04, label='|err|')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(pred2d, cmap=cmap, origin='lower', aspect='equal')
        ax.set_title('pred')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.invert_yaxis()
        fig.suptitle(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='u')

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig


@error_logging_decorator
def plot_grid_contour_overlay(
    pred2d,
    ref2d=None,
    *,
    levels=None,
    fname='fig_grid_contour.png',
    savedir=None,
    title='',
):
    """Contour overlay for grid fields.

    Args:
        pred2d: (nx, ny) array
        ref2d:  (nx, ny) array or None
        levels: contour levels (shared)
    Notes:
        Uses ndgrid-like orientation: x down, y right (invert y-axis).
    """
    pred2d = np.asarray(pred2d)
    if levels is None:
        levels = np.array([0.01, 0.1, 0.3, 0.6])

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.contour(pred2d, levels=levels, linestyles='solid')

    if ref2d is not None:
        ref2d = np.asarray(ref2d)
        ax.contour(ref2d, levels=levels, linestyles='dashed')

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig, ax


@error_logging_decorator
def plot_contour_over_seg(
    pred2d,
    *,
    thresholds=None,
    seg=None,
    bg=None,
    phi=None,
    fname='fig_contour_over_seg.png',
    savedir=None,
    title='',
    gt_alpha=0.5,
):
    """Overlay predicted segmentation contours onto filled GT segmentations.

    Args:
        pred2d: (nx, ny) array of predicted density
        thresholds: list of scalar thresholds
        seg: (nx, ny) array (segmentation mask)
        bg: (nx, ny) array (background image)
        phi: (nx, ny) array (phase field)
    """
    pred2d = np.asarray(pred2d)
    if thresholds is None:
        thresholds = []
    
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # 1. Background
    if bg is not None:
        bg = np.asarray(bg)
        ax.imshow(bg, cmap='gray', origin='lower', aspect='equal')
    
    # 2. Phi contour
    if phi is not None:
        phi = np.asarray(phi)
        ax.contour(phi, levels=[0.5], colors=['white'], linestyles='--', linewidths=1.0)

    handles = []
    labels = []

    # 3. Filled GT masks (seg)
    if seg is not None:
        seg = np.asarray(seg)
        unique_labels = np.unique(seg)
        unique_labels = unique_labels[unique_labels > 0] # Ignore 0 (background)
        
        # Define colors for segmentation labels
        # 1: Necrotic (Red), 2: Edema (Green), 3: Enhancing (Blue), 4: Recurrence (Orange)
        seg_colors = {
            1: 'tab:red',
            2: 'tab:green',
            3: 'tab:blue',
            4: 'tab:orange'
        }
        seg_names = {
            1: 'Necrotic',
            2: 'Edema',
            3: 'Enhancing',
            4: 'Recurrence'
        }
        
        for label in unique_labels:
            mask = seg == label
            color = seg_colors.get(int(label), 'tab:purple') # Default color
            name = seg_names.get(int(label), f'Seg {int(label)}')
            
            ax.contourf(mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=gt_alpha)
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=gt_alpha))
            labels.append(name)

    # 4. Predicted contours
    # 0.01
    # if pred2d.max() > 0.01:
    #     ax.contour(pred2d, levels=[0.01], colors=['yellow'], linestyles='dotted', linewidths=2.0)
    #     handles.append(Line2D([0], [0], color='yellow', linestyle='dotted', linewidth=2.0))
    #     labels.append('u=0.01')

    # Thresholds
    colors = ['cyan', 'magenta', 'lime', 'pink', 'white']
    linestyles = ['dashed', 'solid', 'dashdot', 'dotted']
    
    for i, th in enumerate(thresholds):
        th = float(th)
        if th < pred2d.max():
            color = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            ax.contour(pred2d, levels=[th], colors=[color], linestyles=ls, linewidths=2.0)
            handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2.0))
            labels.append(f'u={th:.2f}')

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if handles:
        ax.legend(handles, labels, loc='best', frameon=True)

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig, ax


@error_logging_decorator
def plot_plan_comparison(
    standard_plan,
    personal_plan,
    *,
    seg_pre=None,
    seg_post=None,
    bg=None,
    fname='fig_plan_comparison.png',
    savedir=None,
    title='',
    gt_alpha=0.5,
):
    """Compare standard and personalized plans with pre/post-op segmentations.

    Args:
        standard_plan: (nx, ny) binary mask
        personal_plan: (nx, ny) binary mask
        seg_pre: (nx, ny) pre-op segmentation
        seg_post: (nx, ny) post-op segmentation (recurrence)
        bg: (nx, ny) background image
    """
    standard_plan = np.asarray(standard_plan)
    personal_plan = np.asarray(personal_plan)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    plt.subplots_adjust(wspace=0.05)
    
    # Helper to plot one panel
    def plot_panel(ax, seg, panel_title):
        # Background
        if bg is not None:
            ax.imshow(bg, cmap='gray', origin='lower', aspect='equal')
            
        handles = []
        labels = []
        
        # Segmentation
        if seg is not None:
            seg = np.asarray(seg)
            unique_labels = np.unique(seg)
            unique_labels = unique_labels[unique_labels > 0]
            
            seg_colors = {
                1: 'tab:red', 2: 'tab:green', 3: 'tab:blue', 4: 'tab:orange'
            }
            seg_names = {
                1: 'Necrotic', 2: 'Edema', 3: 'Enhancing', 4: 'Recurrence'
            }
            
            for label in unique_labels:
                mask = seg == label
                color = seg_colors.get(int(label), 'tab:purple')
                name = seg_names.get(int(label), f'Seg {int(label)}')
                
                ax.contourf(mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=gt_alpha)
                # Only add to legend if not already there (for shared legend across panels if needed, 
                # but here we do per panel or just rely on colors)
                # handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=gt_alpha))
                # labels.append(name)

        # Plans
        if np.any(standard_plan):
            ax.contour(standard_plan, levels=[0.5], colors=['blue'], linewidths=2, linestyles='--')
            handles.append(Line2D([0], [0], color='blue', linestyle='--', lw=2))
            labels.append('Standard Plan')
            
        if np.any(personal_plan):
            ax.contour(personal_plan, levels=[0.5], colors=['red'], linewidths=2, linestyles='-')
            handles.append(Line2D([0], [0], color='red', linestyle='-', lw=2))
            labels.append('Personal Plan')
            
        ax.set_title(panel_title)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.invert_yaxis()
        if handles:
            ax.legend(handles, labels, loc='upper right', frameon=True)

    # Left Panel: Pre-op
    plot_panel(axes[0], seg_pre, 'Pre-op Segmentation')
    
    # Right Panel: Post-op
    plot_panel(axes[1], seg_post, 'Post-op Segmentation (Recurrence)')
    
    if title:
        fig.suptitle(title)

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig, axes


@error_logging_decorator
def plot_density_over_seg(
    pred2d,
    *,
    u1=None,
    u2=None,
    bg=None,
    phi=None,
    fname='fig_density_over_seg.png',
    savedir=None,
    title='',
):
    """Overlay predicted density onto GT segmentation contours.

    Style 2: Background, phi=0.5, u density (>0.01), u1/u2 contours.
    """
    pred2d = np.asarray(pred2d)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # 1. Background
    if bg is not None:
        bg = np.asarray(bg)
        ax.imshow(bg, cmap='gray', origin='lower', aspect='equal')

    # 2. Phi contour
    if phi is not None:
        phi = np.asarray(phi)
        ax.contour(phi, levels=[0.5], colors=['white'], linestyles='--', linewidths=1.0)

    # 3. Density u > 0.01
    masked_u = np.ma.masked_where(pred2d < 0.01, pred2d)
    im = ax.imshow(masked_u, cmap='jet', origin='lower', aspect='equal', alpha=0.6, vmin=0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='u')

    handles = []
    labels = []

    # 4. GT contours (u1, u2)
    if u1 is not None:
        u1 = np.asarray(u1)
        if np.any(u1 > 0.5):
            ax.contour(u1, levels=[0.5], colors=['tab:green'], linestyles='solid', linewidths=2.0)
            handles.append(Line2D([0], [0], color='tab:green', linestyle='solid', linewidth=2.0))
            labels.append('WT')

    if u2 is not None:
        u2 = np.asarray(u2)
        if np.any(u2 > 0.5):
            ax.contour(u2, levels=[0.5], colors=['tab:red'], linestyles='solid', linewidths=2.0)
            handles.append(Line2D([0], [0], color='tab:red', linestyle='solid', linewidth=2.0))
            labels.append('TC')

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if handles:
        ax.legend(handles, labels, loc='upper right', frameon=True)

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig, ax

