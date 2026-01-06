import numpy as np

def gen_samples(Pwm, Pgm, Pcsf, segwt, segtc, phi, factor, x0, rmax_threshold, DW, RHO, L, u_all, tgrid):
    """
    Generate training samples from patient data and characteristic model solution.
    Mimics Sampler.sampleFromPatientModel in MATLAB.

    Args:
        Pwm, Pgm, Pcsf, seg, phi: 3D arrays (or 2D).
        x0: Center coordinates (1-based, matching MATLAB convention).
        L, dw, rho, tfinal: Scalars.
        u_all: Solution array (x, y, z, t).
        rmax_threshold: Threshold for defining the mask.

    Returns:
        dict: Sampled data including subgrid features and coordinates.
    """
    
    # --- 1. Compute Geometry and Derivatives ---
    P = Pwm + Pgm/factor
    
    # u1 = whole tumor 
    u1 = segwt
    # u2 = tumor core
    u2 = segtc
    
    Pphi = P * phi
    grad_Pphi = np.gradient(Pphi, edge_order=1)
    DxPphi = grad_Pphi[0]
    DyPphi = grad_Pphi[1]
    DzPphi = grad_Pphi[2] if len(grad_Pphi) > 2 else np.zeros_like(P)
    
    # --- 2. Create Grid (1-based) ---
    shape = Pwm.shape
    ndim = len(shape)
    
    # Create 1-based coordinate vectors
    # MATLAB: 1:size(i)
    coords = [np.arange(1, s + 1) for s in shape]
    
    # --- 3. Compute rmax and Define Subgrid ---
    # Mask based on final solution
    u_final = u_all[..., -1]
    mask = u_final > rmax_threshold
    
    if not np.any(mask):
        print("Warning: No points found above threshold.")
        return {}

    # Get indices of mask (0-based)
    mask_indices = np.argwhere(mask)
    
    # Convert to 1-based coordinates
    mask_coords = mask_indices + 1
    
    # Compute Linf distance to x0
    # Assuming x0 is 1-based coordinates
    x0_arr = np.array(x0)
    dists = np.abs(mask_coords - x0_arr)
    rmax = np.min([np.max(dists), 64])  # Cap at 64 to avoid too large subgrids
    
    # Define subgrid range
    # [x0 - rmax, x0 + rmax]
    subgrid_slices = []
    subgrid_coords = []
    
    for i in range(ndim):
        c_min = x0_arr[i] - rmax
        c_max = x0_arr[i] + rmax
        
        # Find indices in the 1-based coordinate vector
        # coords[i] are 1, 2, ...
        # We want indices where coords[i] is in range
        
        # Boolean mask on the coordinate vector
        dim_mask = (coords[i] >= c_min) & (coords[i] <= c_max)
        
        # Get 0-based indices for array slicing
        dim_indices = np.where(dim_mask)[0]
        
        subgrid_slices.append(dim_indices)
        subgrid_coords.append(coords[i][dim_indices])
        
    # Create open mesh for indexing
    ix = np.ix_(*subgrid_slices)
    
    # --- 4. Extract Data ---
    def extract(vol):
        # vol is (Nx, Ny, Nz) or (Nx, Ny)
        # ix matches dimensions
        return vol[ix]

    data = {
        'phi': extract(phi),
        'P': extract(P),
        'Pwm': extract(Pwm),
        'Pgm': extract(Pgm),
        'Pcsf': extract(Pcsf),
        'DxPphi': extract(DxPphi),
        'DyPphi': extract(DyPphi),
        'DzPphi': extract(DzPphi),
        'u1': extract(u1),
        'u2': extract(u2),
        'uchar': extract(u_all), # (Nx_sub, Ny_sub, Nz_sub, Nt)
        'x0': x0_arr,
        'rmax': rmax,
        'DW': DW,
        'RHO': RHO,
        'L': L,
        'xdim': ndim,
        'tgrid': tgrid
    }
    
    # Add grid coordinates (gx, gy, gz)
    alias = ['gx', 'gy', 'gz']
    for i in range(ndim):
        data[alias[i]] = subgrid_coords[i]
    
    return data