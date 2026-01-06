import numpy as np
import multiprocessing
from solve_fisher_kpp_sph import solve_fisher_kpp_sph
from functools import partial
import sys
import os

def get_char(dwc, rhoc, L, T=None):
    """
    Implement GliomaSolver.getChar
    
    Args:
        dwc (float): Characteristic diffusion coefficient
        rhoc (float): Characteristic proliferation rate
        L (float): Characteristic length
        T (float, optional): Characteristic time. If None, calculated from L.
        
    Returns:
        dict: Dictionary containing DW, RHO, vc, L, T
    """
    vc = np.sqrt(dwc * rhoc)
    
    if T is None and L is not None:
        T = L / vc
    elif L is None and T is not None:
        L = vc * T
    else:
        raise ValueError("Either L or T should be None")
        
    DW = dwc * T / (L**2)
    RHO = rhoc * T
    
    return {'DW': DW, 'RHO': RHO, 'vc': vc, 'L': L, 'T': T}

def get_max_radius(mask, center=None):
    """
    Calculate the maximum radius of a binary mask from its centroid.
    """
    if not np.any(mask):
        return 0.0
    
    # Get indices of True values
    coords = np.argwhere(mask)
        
    # Calculate centroid
    if center is None:
        centroid = np.mean(coords, axis=0)
    else:
        centroid = np.array(center) - 1.0
    
    # Calculate distances
    dists = np.linalg.norm(coords - centroid, axis=1)
    
    return np.max(dists)

def get_vol_radius(mask, xdim=3):
    """
    Calculate the radius of a sphere/circle with equivalent volume/area.
    """
    if not np.any(mask):
        return 0.0
    
    vol = np.sum(mask) # Assuming dx=1
    
    if xdim == 2:
        return np.sqrt(vol / np.pi)
    else:
        return (3 * vol / (4 * np.pi))**(1/3)

def _sim_worker(params):
    """
    Worker function for parallel simulation.
    """
    if len(params) == 8:
        D, rho, L, wt_th, tc_th, h_init, r_init, xdim = params
    else:
        D, rho, L, wt_th, tc_th, h_init, r_init = params
        xdim = 3
    
    # Calculate characteristic parameters to get T
    char_params = get_char(D, rho, L)
    T = char_params['T']
    
    res = solve_fisher_kpp_sph(
        h_init=h_init, 
        r_init=r_init, 
        BD=180.0, 
        dx=1.0, 
        tfinal=T, 
        Nt=2, 
        D=D, 
        RHO=rho,
        xdim=xdim
    )
    
    u_end = res['u'][:,-1]
    xgrid = res['xgrid']
    
    # Find radii
    # Find first r where u < threshold
    def find_r(u, x, th):
        # Find indices where u < th
        idx = np.where(u < th)[0]
        if len(idx) > 0:
            # The first index is where it drops below threshold
            # We can interpolate for better precision, but ParamEstimator uses simple lookup
            return x[idx[0]]
        return np.nan
        
    r_wt_sim = find_r(u_end, xgrid, wt_th)
    r_tc_sim = find_r(u_end, xgrid, tc_th)
    
    return {
        'Dc': D,
        'rhoc': rho,
        'L': L,
        'T': T,
        'r_wt_sim': r_wt_sim,
        'r_tc_sim': r_tc_sim,
        'error': None
    }
    

def est_char_param(mask_wt, mask_tc, x0, wt_threshold_prior, tc_threshold_prior, h_init, r_init, parallel=True, xdim=3, radius_method='rmax'):
    """
    Estimate characteristic parameters based on segmentation.
    
    Args:
        tumor_seg: 3D numpy array (0: bg, 1: necrotic, 2: edema, 3: enhancing)
        x0: tumor centroid (1-based)
        wt_threshold_prior: Threshold for Whole Tumor in simulation (default 0.35)
        tc_threshold_prior: Threshold for Tumor Core in simulation (default 0.6)
        parallel: Boolean to enable parallel processing (default True)
        xdim: Spatial dimension (2 or 3) for spherical/circular symmetry assumption.
        radius_method: 'rmax' (default) or 'rvol'
        
    Returns:
        dict: Estimated parameters (Dq, rhoq, Lq, Tq, r_wt_sim, r_tc_sim)
    """
    
    # Radius estimation
    # WT: labels 1, 2, 3 (Necrotic + Edema + Enhancing)
    # TC: labels 1, 3 (Necrotic + Enhancing)
    if radius_method == 'rvol':
        r_wt_data = get_vol_radius(mask_wt, xdim)
        r_tc_data = get_vol_radius(mask_tc, xdim)
    else:
        r_wt_data = get_max_radius(mask_wt, center=x0)
        r_tc_data = get_max_radius(mask_tc, center=x0)
        
    
    # Length scale L is r_wt_data
    L = int(np.ceil(r_wt_data))
    
    # 2. Create grid
    # Matching ParamEstimator.m: Ds = linspace(0.1,1,10); rhos = linspace(0.01,0.1,10);
    Ds = np.linspace(0.1, 1.0, 10)
    rhos = np.linspace(0.01, 0.1, 10)
    
    param_list = []
    for D in Ds:
        for rho in rhos:
            param_list.append((D, rho, L, wt_threshold_prior, tc_threshold_prior, h_init, r_init, xdim))
            
    # 3. Run simulations
    results = []
    if parallel:
        # Use multiprocessing
        # Note: When called from MATLAB, multiprocessing might have issues.
        # If it fails, fallback to serial.
        try:
            with multiprocessing.Pool() as pool:
                results = pool.map(_sim_worker, param_list)
        except Exception as e:
            print(f"Parallel execution failed: {e}. Falling back to serial execution.")
            parallel = False
            
    if not parallel:
        for params in param_list:
            results.append(_sim_worker(params))
        
    # 4. Find best fit
    best_dist = float('inf')
    best_res = None
    
    for res in results:
        if res['error']:
            continue
            
        r_wt_sim = res['r_wt_sim']
        r_tc_sim = res['r_tc_sim']
        
        if np.isnan(r_wt_sim) or np.isnan(r_tc_sim):
            continue
            
        # Relative L1 distance
        # d = abs(fseg.r1-fpolar.r1)./fseg.r1 + abs(fseg.r2-fpolar.r2)./fseg.r2;
        # Add small constant to avoid division by zero
        dist = abs(r_wt_data - r_wt_sim)/(r_wt_data + 1e-3) + abs(r_tc_data - r_tc_sim)/(r_tc_data + 1e-3)
        
        if dist < best_dist:
            best_dist = dist
            best_res = res
            
    if best_res is None:
        # If no valid result found, return None or raise error
        # For robustness, return the first result or a default
        raise RuntimeError("Could not find valid parameters (all simulations failed or produced NaNs)")
        
    # Add data radii to result
    best_res['r_wt_data'] = r_wt_data
    best_res['r_tc_data'] = r_tc_data
    best_res['score'] = best_dist
    
    return best_res
