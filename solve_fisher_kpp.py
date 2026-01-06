import numpy as np

def solve_fisher_kpp(Pwm, Pgm, dw, factor, phi, rho, tfinal, u0, Nt):
    """
    Solve Fisher-KPP equation using Finite Difference Method (Non-conservative form).

    Supports 2D or 3D based on the dimensionality of `Pwm`.
    
    Equation:
        du/dt = div(D*phi*grad(u)) + rho*phi*u*(1-u)
    Expanded (Non-conservative):
        du/dt = grad(D*phi).grad(u) + (D*phi)*lap(u) + rho*phi*u*(1-u)
    
    Args:
        Pwm (ndarray): 2D or 3D array, probability of white matter.
        Pgm (ndarray): 2D or 3D array, probability of gray matter.
        dw (float): Diffusion coefficient in white matter.
        factor (float): Ratio of diffusion coeff in white/gray matter (Dw/Dg).
        phi (ndarray): 2D or 3D array, phase field function.
        rho (float): Proliferation rate.
        tfinal (float): Final time.
        u0 (ndarray): 2D or 3D array, initial condition.
        Nt (int): Number of time points to save.
        
    Returns:
        u_all (ndarray): Array [x, y, (z), t] of solution history.
        t_all (ndarray): 1D array of time points.
    """

    if Pwm.ndim not in (2, 3):
        raise ValueError(f"Expected Pwm to be 2D or 3D, got ndim={Pwm.ndim}")
    if Pgm.shape != Pwm.shape or phi.shape != Pwm.shape or u0.shape != Pwm.shape:
        raise ValueError(
            "Pwm, Pgm, phi, and u0 must have the same shape; "
            f"got Pwm={Pwm.shape}, Pgm={Pgm.shape}, phi={phi.shape}, u0={u0.shape}"
        )
    
    # --- Grid Parameters ---
    h = 1.0
    h2 = h * h
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / h2
    
    # --- Diffusion Field Setup ---
    # Diffusion field: D = Dw * Pwm + Dg * Pgm = Dw * (Pwm + Pgm / factor)
    df = dw * (Pwm + Pgm / factor)
    
    # Effective diffusion coefficient term for the PDE expansion: D_eff = df * phi
    Dphi = df * phi

    # Precompute gradients of Dphi (Fixed in time) using central differences.
    # (Kept as explicit slicing for speed and to match original style.)
    if Pwm.ndim == 3:
        dx_Dphi = np.zeros_like(Dphi)
        dx_Dphi[1:-1, :, :] = (Dphi[2:, :, :] - Dphi[:-2, :, :]) * inv_2h

        dy_Dphi = np.zeros_like(Dphi)
        dy_Dphi[:, 1:-1, :] = (Dphi[:, 2:, :] - Dphi[:, :-2, :]) * inv_2h

        dz_Dphi = np.zeros_like(Dphi)
        dz_Dphi[:, :, 1:-1] = (Dphi[:, :, 2:] - Dphi[:, :, :-2]) * inv_2h
    else:
        dx_Dphi = np.zeros_like(Dphi)
        dx_Dphi[1:-1, :] = (Dphi[2:, :] - Dphi[:-2, :]) * inv_2h

        dy_Dphi = np.zeros_like(Dphi)
        dy_Dphi[:, 1:-1] = (Dphi[:, 2:] - Dphi[:, :-2]) * inv_2h
    
    # --- Time Stepping Setup ---
    # CFL condition: dt <= h^2 / (2 * d * max_diff)
    # d = number of spatial dimensions
    max_diff = np.max(Dphi)
    if max_diff == 0:
        max_diff = 1e-9 # Avoid division by zero

    d = Pwm.ndim
    cfl_dt = 0.99 * h2 / (2 * d * max_diff)
    
    # Timestamps to save
    t_all = np.linspace(0, tfinal, Nt)
    u_all = np.zeros(u0.shape + (Nt,), dtype=np.float64)
    
    # Initial condition
    u = u0.copy().astype(np.float64)
    u_all[..., 0] = u
    
    t = 0.0
    save_idx = 1 # Next index to save (0 is already saved)
    
    # Helper for Dirichlet BC (set boundaries to 0)
    def apply_dirichlet_bc(arr):
        if arr.ndim == 2:
            arr[0, :] = 0; arr[-1, :] = 0
            arr[:, 0] = 0; arr[:, -1] = 0
        else:
            arr[0, :, :] = 0; arr[-1, :, :] = 0
            arr[:, 0, :] = 0; arr[:, -1, :] = 0
            arr[:, :, 0] = 0; arr[:, :, -1] = 0
        return arr

    # Apply BC to initial u
    u = apply_dirichlet_bc(u)
    
    print(f"Starting simulation. tfinal={tfinal}, dt_cfl={cfl_dt:.4e}")
    
    # --- Main Loop ---
    while t < tfinal and save_idx < Nt:
        target_t = t_all[save_idx]
        
        # Determine time step to hit target exactly or use CFL step
        current_dt = cfl_dt
        if t + current_dt > target_t:
            current_dt = target_t - t
            hit_save_point = True
        else:
            hit_save_point = False
            
        # --- Finite Difference Operations (Slicing) ---
        
        if Pwm.ndim == 3:
            # Views for interior and neighbors
            u_center = u[1:-1, 1:-1, 1:-1]

            u_xp = u[2:, 1:-1, 1:-1]
            u_xm = u[:-2, 1:-1, 1:-1]
            u_yp = u[1:-1, 2:, 1:-1]
            u_ym = u[1:-1, :-2, 1:-1]
            u_zp = u[1:-1, 1:-1, 2:]
            u_zm = u[1:-1, 1:-1, :-2]

            # 1. Gradients of u (Central Difference)
            dx_u = (u_xp - u_xm) * inv_2h
            dy_u = (u_yp - u_ym) * inv_2h
            dz_u = (u_zp - u_zm) * inv_2h

            # 2. Laplacian of u (7-point stencil)
            lap_u = (u_xp + u_xm + u_yp + u_ym + u_zp + u_zm - 6.0 * u_center) * inv_h2

            # 3. Reaction term
            phi_in = phi[1:-1, 1:-1, 1:-1]
            reaction = rho * phi_in * u_center * (1.0 - u_center)

            # 4. Advection-like term: grad(Dphi) . grad(u)
            advection = (
                dx_Dphi[1:-1, 1:-1, 1:-1] * dx_u
                + dy_Dphi[1:-1, 1:-1, 1:-1] * dy_u
                + dz_Dphi[1:-1, 1:-1, 1:-1] * dz_u
            )

            # 5. Diffusion term: Dphi * lap(u)
            diffusion = Dphi[1:-1, 1:-1, 1:-1] * lap_u
        else:
            # 2D case
            u_center = u[1:-1, 1:-1]

            u_xp = u[2:, 1:-1]
            u_xm = u[:-2, 1:-1]
            u_yp = u[1:-1, 2:]
            u_ym = u[1:-1, :-2]

            dx_u = (u_xp - u_xm) * inv_2h
            dy_u = (u_yp - u_ym) * inv_2h

            lap_u = (u_xp + u_xm + u_yp + u_ym - 4.0 * u_center) * inv_h2

            phi_in = phi[1:-1, 1:-1]
            reaction = rho * phi_in * u_center * (1.0 - u_center)

            advection = dx_Dphi[1:-1, 1:-1] * dx_u + dy_Dphi[1:-1, 1:-1] * dy_u

            diffusion = Dphi[1:-1, 1:-1] * lap_u
        
        # Total change
        dudt = advection + diffusion + reaction
        
        # Update u (only interior)
        if Pwm.ndim == 3:
            u[1:-1, 1:-1, 1:-1] += current_dt * dudt
        else:
            u[1:-1, 1:-1] += current_dt * dudt
        
        # Threshold
        np.clip(u, 0, 1, out=u)
        
        t += current_dt
        
        if hit_save_point:
            u_all[..., save_idx] = u
            save_idx += 1
            
    return u_all, t_all