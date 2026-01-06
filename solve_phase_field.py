import numpy as np

def solve_phase_field_numpy(Pwm, Pgm, Pcsf, numiter=100, epsilon=3):
    """
    Pure NumPy implementation using explicit finite differences (np.roll).

    Supports 2D or 3D based on the dimensionality of `Pwm`.
    """
    if Pwm.ndim not in (2, 3):
        raise ValueError(f"Expected Pwm to be 2D or 3D, got ndim={Pwm.ndim}")

    h = 1.0
    h2 = h * h
    
    # --- Initialization ---
    phi_threshold = 0.1
    # Use float64 for stability or float32 for speed/memory
    phi = ((Pwm + Pgm) > np.maximum(phi_threshold, Pcsf)).astype(np.float64)
    
    dt = h**4 / (16.0 * epsilon)
    tau = 1e-3
    
    # Pre-calculate constants
    epsilon_sq = epsilon**2
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / h2

    # --- Periodic Finite Differences (np.roll) ---
    # np.roll(a, -1) shifts elements to the left (index i gets value from i+1) -> f(x+h)
    # np.roll(a, 1) shifts elements to the right (index i gets value from i-1) -> f(x-h)
    dim = Pwm.ndim
    
    for _ in range(numiter):
        # 1. Compute Laplacian of phi
        # Lap(phi) = (sum over axes (phi(x+h)+phi(x-h)) - 2*dim*phi) / h^2
        lap_phi = (-2.0 * dim) * phi
        for axis in range(dim):
            lap_phi += np.roll(phi, -1, axis=axis) + np.roll(phi, 1, axis=axis)
        lap_phi *= inv_h2
        
        # 2. Chemical potential u
        # u = 0.5*phi*(1-phi)*(1-2*phi) - epsilon^2*lap_phi
        u = 0.5 * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - epsilon_sq * lap_phi
        
        # 3. Mobility M
        M = phi * (1.0 - phi) + tau
        
        # 4. Gradients of u (Central Difference)
        # dx = (u(x+1) - u(x-1)) / 2h
        fluxes = []
        for axis in range(dim):
            d_u = (np.roll(u, -1, axis=axis) - np.roll(u, 1, axis=axis)) * inv_2h
            fluxes.append(M * d_u)

        # 5. Divergence of Fluxes
        # div(flux) = sum over axes d/dx_axis (flux_axis)
        dtphi = 0.0
        for axis, flux in enumerate(fluxes):
            dtphi += (np.roll(flux, -1, axis=axis) - np.roll(flux, 1, axis=axis)) * inv_2h
        
        # 6. Update
        phi += dt * dtphi
        np.clip(phi, 0, 1, out=phi)

    return phi