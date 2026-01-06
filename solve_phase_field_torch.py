import torch
import numpy as np

def solve_phase_field_torch(Pwm, Pgm, Pcsf, numiter=100, epsilon=3, device=None):
    """
    PyTorch implementation of Phase Field solver using explicit finite differences.
    Uses torch.compile for JIT acceleration (requires PyTorch 2.0+).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Phase Field Solver using device: {device}")

    dtype = torch.float32
    
    # Move to device
    Pwm_t = torch.as_tensor(Pwm, dtype=dtype, device=device)
    Pgm_t = torch.as_tensor(Pgm, dtype=dtype, device=device)
    Pcsf_t = torch.as_tensor(Pcsf, dtype=dtype, device=device)
    
    h = 1.0
    h2 = h * h
    
    # Initialization
    phi_threshold = 0.1
    phi = ((Pwm_t + Pgm_t) > torch.maximum(torch.tensor(phi_threshold, device=device), Pcsf_t)).to(dtype)
    
    dt = h**4 / (16.0 * epsilon)
    tau = 1e-3
    epsilon_sq = epsilon**2
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / h2
    
    dim = Pwm_t.ndim

    # Define the update step and compile it
    # @torch.compile
    def step_fn(phi_curr):
        # 1. Laplacian of phi
        # Lap(phi) = (sum over axes (phi(x+h)+phi(x-h)) - 2*dim*phi) / h^2
        lap_phi = (-2.0 * dim) * phi_curr
        for axis in range(dim):
            lap_phi += torch.roll(phi_curr, shifts=-1, dims=axis) + torch.roll(phi_curr, shifts=1, dims=axis)
        lap_phi *= inv_h2
        
        # 2. Chemical potential u
        u = 0.5 * phi_curr * (1.0 - phi_curr) * (1.0 - 2.0 * phi_curr) - epsilon_sq * lap_phi
        
        # 3. Mobility M
        M = phi_curr * (1.0 - phi_curr) + tau
        
        # 4. Divergence of Fluxes (M * grad(u))
        # div(flux) = sum over axes d/dx_axis (M * du/dx_axis)
        dtphi = torch.zeros_like(phi_curr)
        
        for axis in range(dim):
            # Gradient of u (Central Difference)
            d_u = (torch.roll(u, shifts=-1, dims=axis) - torch.roll(u, shifts=1, dims=axis)) * inv_2h
            flux = M * d_u
            # Divergence
            dtphi += (torch.roll(flux, shifts=-1, dims=axis) - torch.roll(flux, shifts=1, dims=axis)) * inv_2h
            
        return dtphi

    # Main loop
    for _ in range(numiter):
        dphi = step_fn(phi)
        phi += dt * dphi
        phi.clamp_(0.0, 1.0)
        
    return phi.cpu().numpy()
