import numpy as np
def solve_fisher_kpp_sph(h_init, r_init, BD, dx, tfinal, Nt, D, RHO, xdim=3):
    """
    Finite difference solver for Fisher-KPP in spherically symmetric geometry.
    
    Equation:
        partial_t u = D * (u_rr + ((xdim-1)/r) * u_r) + RHO * u * (1 - u)
        where xdim is the spatial dimension (2 or 3).
        
    Initial Condition:
        u(r, 0) = h_init * exp(-r_init * r^2)
    
    Inputs:
        h_init (float): Amplitude of Gaussian initial condition.
        r_init (float): Decay coefficient of Gaussian initial condition.
        BD (float): Domain radius [0, BD].
        dx (float): Spatial grid step.
        tfinal (float): Final simulation time.
        Nt (int): Number of time points to save.
        D (float): Diffusion coefficient.
        RHO (float): Proliferation rate.
        xdim (int): Spatial dimension for spherical symmetry (2 or 3).
        
    Returns:
        dict: Dictionary containing 'xgrid', 'tgrid', and 'u' (solution matrix [space, time]).
    """

    if xdim not in (2, 3):
        raise ValueError(f"xdim must be 2 or 3 for spherical symmetry, got {xdim!r}")
    
    # --- 1. Grid Generation ---
    # Create grid including 0 and BD
    N = int(np.round(BD / dx)) + 1
    xgrid = np.linspace(0, BD, N)
    dx = xgrid[1] - xgrid[0] # Recompute exact dx
    
    # --- 2. Initial Condition ---
    # u0 = h_init * exp(-r_init * r^2)
    u0 = h_init * np.exp(-r_init * xgrid**2)
    
    # --- 3. CFL Condition ---
    # For xdim-dimensional spherical symmetry, the operator at r=0 behaves like xdim * D * d^2/dr^2
    # Stability condition: dt <= dx^2 / (2 * effective_D)
    effective_D_origin = xdim * D
    cfl_dt = 0.9 * dx**2 / (2.0 * effective_D_origin)
    
    
    # --- 4. Initialization ---
    t_save = np.linspace(0, tfinal, Nt)
    u_all = np.zeros((N, Nt))
    u_all[:, 0] = u0
    
    u = u0.copy().astype(np.float64)
    t = 0.0
    save_idx = 1
    
    # Precompute constants
    inv_dx2 = 1.0 / (dx**2)
    inv_2dx = 1.0 / (2.0 * dx)
    
    # --- 5. Time Stepping Loop ---
    while t < tfinal and save_idx < Nt:
        # Determine time step
        next_save_t = t_save[save_idx]
        dt = cfl_dt
        
        # Adjust dt to hit save points exactly
        if t + dt >= next_save_t:
            dt = next_save_t - t
            is_save_step = True
        else:
            is_save_step = False
            
        # --- Compute Spatial Derivatives ---
        
        # Shift arrays for vectorization
        # u_ip1 corresponds to u[i+1], u_im1 to u[i-1]
        u_ip1 = np.roll(u, -1)
        u_im1 = np.roll(u, 1)
        
        # Central Difference: Second Derivative d2u/dr2
        d2u = (u_ip1 - 2.0 * u + u_im1) * inv_dx2
        
        # Central Difference: First Derivative du/dr
        du = (u_ip1 - u_im1) * inv_2dx
        
        # Radial term: ((xdim-1)/r) * du/dr
        # Suppress division by zero warning at r=0 (handled by BC later)
        with np.errstate(divide='ignore', invalid='ignore'):
            radial_term = ((xdim - 1.0) / xgrid) * du
            
        # Combine to form Laplacian
        lap = d2u + radial_term
        
        # --- Apply Boundary Conditions to Laplacian ---
        
        # Left BC (r=0): Symmetry / Neumann -> du/dr = 0
        # At the origin, the operator limit is xdim * d2u/dr2
        # Using symmetry u[-1] = u[1], d2u at 0 becomes 2*(u[1]-u[0])/dx^2
        lap[0] = xdim * 2.0 * (u[1] - u[0]) * inv_dx2
        
        # Right BC (r=BD): Neumann -> du/dr = 0
        # Using ghost point u[N] = u[N-2], d2u at N-1 becomes 2*(u[N-2]-u[N-1])/dx^2
        # The radial term (2/r)*du/dr vanishes because du/dr = 0
        lap[-1] = 2.0 * (u[-2] - u[-1]) * inv_dx2
        
        # --- Update Solution ---
        # Forward Euler
        reaction = RHO * u * (1.0 - u)
        u = u + dt * (D * lap + reaction)
        
        t += dt
        
        # Save result
        if is_save_step:
            u_all[:, save_idx] = u
            save_idx += 1
            # Avoid precision drift
            t = next_save_t
            
    return {'xgrid': xgrid, 'tgrid': t_save, 'u': u_all}