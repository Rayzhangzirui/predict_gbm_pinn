import torch
import numpy as np

def solve_fisher_kpp_torch(Pwm, Pgm, dw, factor, phi, rho, tfinal, u0, Nt, device=None):
    """
    PyTorch implementation of Fisher-KPP solver.
    Automatically uses GPU if available and device is not specified.
    """
    
    # 1. Device selection
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Solve FKPP using device: {device}")

    # 2. Convert inputs to Tensor and move to device
    # Using float32 for GPU performance (usually sufficient precision)
    dtype = torch.float32 
    
    def to_tensor(x):
        return torch.as_tensor(x, dtype=dtype, device=device)

    Pwm_t = to_tensor(Pwm)
    Pgm_t = to_tensor(Pgm)
    phi_t = to_tensor(phi)
    u0_t = to_tensor(u0)
    
    # Scalars
    dw = float(dw)
    factor = float(factor)
    rho = float(rho)
    tfinal = float(tfinal)

    # Check shapes
    if Pwm_t.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D input, got {Pwm_t.ndim}")

    # --- Grid Parameters ---
    h = 1.0
    h2 = h * h
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / h2

    # --- Diffusion Field Setup ---
    # D = Dw * (Pwm + Pgm / factor)
    df = dw * (Pwm_t + Pgm_t / factor)
    Dphi = df * phi_t

    # Precompute gradients of Dphi
    # Using slicing for finite differences
    if Pwm_t.ndim == 3:
        dx_Dphi = torch.zeros_like(Dphi)
        dx_Dphi[1:-1, :, :] = (Dphi[2:, :, :] - Dphi[:-2, :, :]) * inv_2h

        dy_Dphi = torch.zeros_like(Dphi)
        dy_Dphi[:, 1:-1, :] = (Dphi[:, 2:, :] - Dphi[:, :-2, :]) * inv_2h

        dz_Dphi = torch.zeros_like(Dphi)
        dz_Dphi[:, :, 1:-1] = (Dphi[:, :, 2:] - Dphi[:, :, :-2]) * inv_2h
    else:
        dx_Dphi = torch.zeros_like(Dphi)
        dx_Dphi[1:-1, :] = (Dphi[2:, :] - Dphi[:-2, :]) * inv_2h

        dy_Dphi = torch.zeros_like(Dphi)
        dy_Dphi[:, 1:-1] = (Dphi[:, 2:] - Dphi[:, :-2]) * inv_2h

    # --- Time Stepping Setup ---
    max_diff = torch.max(Dphi).item()
    if max_diff == 0: max_diff = 1e-9

    d = Pwm_t.ndim
    # CFL condition
    cfl_dt = 0.99 * h2 / (2 * d * max_diff)
    
    print(f"Starting simulation (PyTorch). tfinal={tfinal}, dt={cfl_dt:.4e}")

    # Result storage (keep on CPU to save GPU memory if Nt is large)
    # But for small Nt, we can keep on GPU or move to CPU at the end.
    # Here we return numpy arrays, so we'll store in a CPU tensor or list.
    t_all = np.linspace(0, tfinal, Nt)
    
    # We will store results in a CPU tensor to avoid OOM on GPU for long histories
    u_all = torch.zeros(u0_t.shape + (Nt,), dtype=dtype, device='cpu')
    
    u = u0_t.clone()
    
    # Save initial condition
    u_all[..., 0] = u.cpu()
    
    t = 0.0
    save_idx = 1

    # Helper for Dirichlet BC
    def apply_bc(arr):
        if arr.ndim == 2:
            arr[0, :] = 0; arr[-1, :] = 0
            arr[:, 0] = 0; arr[:, -1] = 0
        else:
            arr[0, :, :] = 0; arr[-1, :, :] = 0
            arr[:, 0, :] = 0; arr[:, -1, :] = 0
            arr[:, :, 0] = 0; arr[:, :, -1] = 0
        return arr

    u = apply_bc(u)

    # --- JIT Compiled Step Functions ---
    
    # @torch.compile
    def step_3d(u_curr, phi_in, Dphi_in, dx_Dphi_in, dy_Dphi_in, dz_Dphi_in, rho_val):
        # Views
        u_center = u_curr[1:-1, 1:-1, 1:-1]
        
        # Gradients
        dx_u = (u_curr[2:, 1:-1, 1:-1] - u_curr[:-2, 1:-1, 1:-1]) * inv_2h
        dy_u = (u_curr[1:-1, 2:, 1:-1] - u_curr[1:-1, :-2, 1:-1]) * inv_2h
        dz_u = (u_curr[1:-1, 1:-1, 2:] - u_curr[1:-1, 1:-1, :-2]) * inv_2h
        
        # Laplacian
        lap_u = (u_curr[2:, 1:-1, 1:-1] + u_curr[:-2, 1:-1, 1:-1] +
                 u_curr[1:-1, 2:, 1:-1] + u_curr[1:-1, :-2, 1:-1] +
                 u_curr[1:-1, 1:-1, 2:] + u_curr[1:-1, 1:-1, :-2] - 
                 6.0 * u_center) * inv_h2
        
        # Reaction
        reaction = rho_val * phi_in[1:-1, 1:-1, 1:-1] * u_center * (1.0 - u_center)
        
        # Advection
        advection = (
            dx_Dphi_in[1:-1, 1:-1, 1:-1] * dx_u +
            dy_Dphi_in[1:-1, 1:-1, 1:-1] * dy_u +
            dz_Dphi_in[1:-1, 1:-1, 1:-1] * dz_u
        )
        
        # Diffusion
        diffusion = Dphi_in[1:-1, 1:-1, 1:-1] * lap_u
        
        return advection + diffusion + reaction

    # @torch.compile
    def step_2d(u_curr, phi_in, Dphi_in, dx_Dphi_in, dy_Dphi_in, rho_val):
        u_center = u_curr[1:-1, 1:-1]
        
        dx_u = (u_curr[2:, 1:-1] - u_curr[:-2, 1:-1]) * inv_2h
        dy_u = (u_curr[1:-1, 2:] - u_curr[1:-1, :-2]) * inv_2h
        
        lap_u = (u_curr[2:, 1:-1] + u_curr[:-2, 1:-1] +
                 u_curr[1:-1, 2:] + u_curr[1:-1, :-2] - 
                 4.0 * u_center) * inv_h2
        
        reaction = rho_val * phi_in[1:-1, 1:-1] * u_center * (1.0 - u_center)
        
        advection = dx_Dphi_in[1:-1, 1:-1] * dx_u + dy_Dphi_in[1:-1, 1:-1] * dy_u
        diffusion = Dphi_in[1:-1, 1:-1] * lap_u
        
        return advection + diffusion + reaction

    # --- Main Loop ---
    # JIT compilation via torch.compile could speed this up further in PyTorch 2.0+
    # but standard execution is already fast on GPU.
    
    while t < tfinal and save_idx < Nt:
        target_t = t_all[save_idx]
        
        current_dt = cfl_dt
        if t + current_dt > target_t:
            current_dt = target_t - t
            hit_save_point = True
        else:
            hit_save_point = False
            
        # --- Finite Difference Operations ---
        if Pwm_t.ndim == 3:
            dudt = step_3d(u, phi_t, Dphi, dx_Dphi, dy_Dphi, dz_Dphi, rho)
            u[1:-1, 1:-1, 1:-1] += current_dt * dudt
        else:
            dudt = step_2d(u, phi_t, Dphi, dx_Dphi, dy_Dphi, rho)
            u[1:-1, 1:-1] += current_dt * dudt

        # Clamp
        u.clamp_(0.0, 1.0)
        
        t += current_dt
        
        if hit_save_point:
            u_all[..., save_idx] = u.cpu()
            save_idx += 1
            
    return u_all.numpy(), t_all
