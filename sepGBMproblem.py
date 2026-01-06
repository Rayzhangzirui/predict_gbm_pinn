from torch.func import jacfwd, vmap
from Options import *
from util import *
from DenseNet import SeparableDenseNet
from GBMDataset import GBMDataset
from GBMplot import *
from GBMproblem import *


def build_lambda_transform(xdim, output_rank, h_init, r_init, L, r2max):
    transforms = []

    # 1) time transform: enforce h1(0)=1 and hi(0)=0 for i>=2
    def time_transform(t, u):
        # works whether u is (r,) (vmap-hidden batch) or (N,r)
        if output_rank == 1:
            one = torch.ones_like(u)
            return one + t * u

        u0 = u[..., 0:1]          # (1,) or (N,1)
        one = torch.ones_like(u0) # same shape as u0

        col0 = one + t * u0
        rest = t * u[..., 1:]     # (r-1,) or (N,r-1)
        return torch.cat([col0, rest], dim=-1)

    transforms.append(time_transform)

    # 2) space transforms: overwrite only first rank component with amp*exp(-r_init*x^2)
    def make_space_transform():
        def space_transform(x, u):
            # x is (1,) under vmap, or (N,1) normally
            h = torch.as_tensor(h_init, dtype=x.dtype, device=x.device)
            rr = torch.as_tensor(r_init, dtype=x.dtype, device=x.device)
            amp = h.pow(1.0 / float(xdim))

            val = amp * torch.exp(-rr * ((x * L)**2))  # (1,) or (N,1)

            if output_rank == 1:
                return val

            rest = u[..., 1:] * (r2max - x**2)  # (r-1,) or (N,r-1)
            return torch.cat([val, rest], dim=-1)

        return space_transform

    for _ in range(xdim):
        transforms.append(make_space_transform())

    return transforms


class sepGBMproblem(GBMproblem):
    def __init__(self, xdim, pde_opts: dict):
        super().__init__(xdim, pde_opts)

        self.loss_dict['ic'] = self.get_ic_loss
        self.loss_dict['fdmbc'] = self.fdmbcloss
        
        # finite difference method options
        self.use_fdm = pde_opts['use_fdm']  # default to automatic differentiation
        

    def setup_network(self, **kwargs):
        '''setup network for separable net'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        # Use the new helper to enforce IC
        list_of_transforms = build_lambda_transform(self.xdim, kwargs['output_rank'], self.h_init, self.r_init, self.L, self.r2max)

        net = SeparableDenseNet(**kwargs,
                        lambda_transform = list_of_transforms,
                        all_params_dict = self.all_params_dict,
                        pde_params = self.pde_params,
                        trainable_param = self.opts['trainable_param'])
        return net

    def setup_dataset(self, traindat_path, whichdata, device):
        ''' setup dataset for SPINN'''

        # configure data
        assert whichdata in ['char', 'gt', 'pat'], f'whichdata {whichdata} not recognized'
        self.whichdata = whichdata
        
        if self.whichdata == 'pat':
            ignore_fields = ['ugt','uchar']
        else:
            ignore_fields = []
        
        self.dataset = {}
        # make sure path exist
        if os.path.exists(traindat_path):
            self.dataset = GBMDataset(traindat_path, ignore = ignore_fields)
        else:
            raise FileNotFoundError(f'Training data file {traindat_path} not found.')

        # get physical parameters from dataset
        self.DW = self.dataset['DW']
        self.RHO = self.dataset['RHO']
        self.L = self.dataset['L']
        self.h_init = self.dataset['h_init']
        self.r_init = self.dataset['r_init']
        self.r2max = (self.dataset['rmax']/self.L)**2

        # no lambda transform needed for SPINN
        
        # ground truth parameters
        self.gt_param = {}
        if 'gt' == self.whichdata:
            self.gt_param['rD'] = self.dataset['rDe']
            self.gt_param['rRHO'] = self.dataset['rRHOe']
        else:
            self.gt_param['rD'] = 1.0
            self.gt_param['rRHO'] = 1.0


        # prepare data
        self.dataset.prepare_all_space_field()
        # put all variables to device
        self.dataset.to_device(device)

        self.Nt = self.opts['Nt']
        self.dataset.configure_dataset_spinn(self.Nt)

    def apply_loss_weighting(self, res, t):
        t = t.flatten()

        if self.opts.get('causal_weight', False):
            res_p2 = res.detach()**2
            # mean over spatial dimensions to get loss per time step
            spatial_dims = tuple(range(1, res.ndim))
            res_t = torch.mean(res_p2, dim=spatial_dims)
            res_cumsum = torch.cumsum(res_t, dim=0)
            # normalize so that min is 0
            res_cumsum = res_cumsum - torch.min(res_cumsum)
            # normalize max is 1
            res_cumsum = res_cumsum / (torch.max(res_cumsum) + 1e-10)
            w = torch.sqrt(torch.exp(-self.opts['texp_weight'] * res_cumsum))
        else:
            t_detach = t.detach()
            w = torch.sqrt(torch.exp(-self.opts['texp_weight'] * t_detach))  # time weighting exp(-w*t)

        # reshape w to match res shape (Nt, 1, 1, ...)
        res = res * w.reshape(-1, *([1] * (res.ndim - 1)))

        return res

    def residual_ad(self, nn, t, coords, phi, P, gradPphi):
        # t: (Nt, 1)
        # coords: list of [(Nx, 1), (Ny, 1), (Nz, 1)]
        
        coord_list = [t] + coords
        
        # Compute f and derivatives for each coordinate
        f_list = []
        df_list = [] # 1st derivative
        d2f_list = [] # 2nd derivative
        
        for i in range(self.input_dim):
            c = coord_list[i]
            # f
            f = nn.mlps[i](c)
            f_list.append(f)
            
            # df
            df = vmap(jacfwd(nn.mlps[i]))(c).view(-1, nn.output_rank)
            df_list.append(df)
            
            # d2f (only needed for spatial dims, i > 0)
            if i > 0:
                d2f = vmap(jacfwd(jacfwd(nn.mlps[i])))(c).view(-1, nn.output_rank)
                d2f_list.append(d2f)
            else:
                d2f_list.append(None)

        # Explicit residuals based on xdim
        if self.xdim == 1:
            # 1D space + time. Coords: t, x. Params: rD, rRHO
            # u = f_t * f_x * th_D * th_rho
            
            # Unpack
            f_t, f_x = f_list[0], f_list[1]
            df_t = df_list[0]
            d2f_x = d2f_list[1]
            
            # u
            u = torch.einsum(nn.einsum, f_t, f_x)
            
            # u_t
            u_t = torch.einsum(nn.einsum, df_t, f_x)
            
            # u_xx
            u_xx = torch.einsum(nn.einsum, f_t, d2f_x)
            
            # u_x (needed for gradPphi term)
            df_x = df_list[1]
            u_x = torch.einsum(nn.einsum, f_t, df_x)
            
            sum_u_xx = u_xx
            sum_grad_u = gradPphi[0].unsqueeze(0) * u_x
            
        elif self.xdim == 2:
            # 2D space + time. Coords: t, x, y. Params: rD, rRHO
            f_t, f_x, f_y = f_list[0], f_list[1], f_list[2]
            df_t = df_list[0]
            df_x, df_y = df_list[1], df_list[2]
            d2f_x, d2f_y = d2f_list[1], d2f_list[2]            
            # u
            u = torch.einsum(nn.einsum, f_t, f_x, f_y)
            # u_t
            u_t = torch.einsum(nn.einsum, df_t, f_x, f_y)
            # u_xx
            u_xx = torch.einsum(nn.einsum, f_t, d2f_x, f_y)
            # u_yy
            u_yy = torch.einsum(nn.einsum, f_t, f_x, d2f_y)
            # u_x
            u_x = torch.einsum(nn.einsum, f_t, df_x, f_y)
            # u_y
            u_y = torch.einsum(nn.einsum, f_t, f_x, df_y)
            sum_u_xx = u_xx + u_yy
            sum_grad_u = gradPphi[0].unsqueeze(0) * u_x + gradPphi[1].unsqueeze(0) * u_y
            
        elif self.xdim == 3:
            # 3D space + time. Coords: t, x, y, z. Params: rD, rRHO
            f_t, f_x, f_y, f_z = f_list[0], f_list[1], f_list[2], f_list[3]
            df_t = df_list[0]
            df_x, df_y, df_z = df_list[1], df_list[2], df_list[3]
            d2f_x, d2f_y, d2f_z = d2f_list[1], d2f_list[2], d2f_list[3]
            
            # u
            u = torch.einsum(nn.einsum, f_t, f_x, f_y, f_z)
            u_t = torch.einsum(nn.einsum, df_t, f_x, f_y, f_z)
            u_xx = torch.einsum(nn.einsum, f_t, d2f_x, f_y, f_z)
            u_yy = torch.einsum(nn.einsum, f_t, f_x, d2f_y, f_z)
            u_zz = torch.einsum(nn.einsum, f_t, f_x, f_y, d2f_z)
            u_x = torch.einsum(nn.einsum, f_t, df_x, f_y, f_z)
            u_y = torch.einsum(nn.einsum, f_t, f_x, df_y, f_z)
            u_z = torch.einsum(nn.einsum, f_t, f_x, f_y, df_z)
            sum_u_xx = u_xx + u_yy + u_zz
            sum_grad_u = gradPphi[0].unsqueeze(0) * u_x + gradPphi[1].unsqueeze(0) * u_y + gradPphi[2].unsqueeze(0) * u_z
            
        else:
            raise ValueError(f"Unsupported xdim: {self.xdim}")
            
        # Now assemble GBM residual
        rD = nn.all_params_dict['rD']
        rRHO = nn.all_params_dict['rRHO']
        
        # Broadcast phi and P
        phi_b = phi.unsqueeze(0)
        P_b = P.unsqueeze(0)
        
        prof = rRHO * self.RHO * phi_b * u * (1 - u)
        diff = rD * self.DW * (P_b * phi_b * sum_u_xx + self.L * sum_grad_u)
        
        res = phi_b * u_t - (prof + diff)

        # weighted residuals by time
        res = self.apply_loss_weighting(res, t)
        
        return res, u

    def resloss(self, net):
        data = self.dataset.batch['res']
        t = data['t']
        coords = data['coords']
        phi = data['phi']
        P = data['P']
        
        gradPphi = []
        if self.xdim >= 1:
            gradPphi.append(data['DxPphi'])
        if self.xdim >= 2:
            gradPphi.append(data['DyPphi'])
        if self.xdim >= 3:
            gradPphi.append(data['DzPphi'])
            
        if self.use_fdm:
            self.res, self.upred_res = self.residual_fdm(net, t, coords, phi, P, gradPphi)
        else:
            self.res, self.upred_res = self.residual_ad(net, t, coords, phi, P, gradPphi)
            
        val_loss_res = torch.mean(torch.square(self.res))
        return val_loss_res

    def get_res_pred(self, net):
        data = self.dataset.batch['res']
        t = data['t']
        coords = data['coords']
        phi = data['phi']
        P = data['P']
        
        gradPphi = []
        if self.xdim >= 1:
            gradPphi.append(data['DxPphi'])
        if self.xdim >= 2:
            gradPphi.append(data['DyPphi'])
        if self.xdim >= 3:
            gradPphi.append(data['DzPphi'])
            
        if self.use_fdm:
            res, u_pred = self.residual_fdm(net, t, coords, phi, P, gradPphi)
        else:
            res, u_pred = self.residual_ad(net, t, coords, phi, P, gradPphi)
            
        self.res = res
        self.upred_res = u_pred
        return res, u_pred
    
    def compute_spatial_derivatives_fd(self, u_grid):
        """
        Compute spatial derivatives using central difference with boundary handling
        
        Args:
            u_grid: (x, y ,z, t)
        """
        h = (self.dataset['gx_scaled'][1] - self.dataset['gx_scaled'][0]).item()
        
        def deriv_1d(u, dim):
            # Central difference for interior
            sl_mid = [slice(None)] * u.ndim
            sl_plus = [slice(None)] * u.ndim
            sl_minus = [slice(None)] * u.ndim
            
            sl_mid[dim] = slice(1, -1)
            sl_plus[dim] = slice(2, None)
            sl_minus[dim] = slice(None, -2)
            
            du = torch.zeros_like(u)
            du[tuple(sl_mid)] = (u[tuple(sl_plus)] - u[tuple(sl_minus)]) / (2 * h)
            
            # Boundaries - Forward/Backward
            sl_0 = [slice(None)] * u.ndim
            sl_0[dim] = 0
            sl_1 = [slice(None)] * u.ndim
            sl_1[dim] = 1
            
            sl_end = [slice(None)] * u.ndim
            sl_end[dim] = -1
            sl_end_m1 = [slice(None)] * u.ndim
            sl_end_m1[dim] = -2
            
            du[tuple(sl_0)] = (u[tuple(sl_1)] - u[tuple(sl_0)]) / h
            du[tuple(sl_end)] = (u[tuple(sl_end)] - u[tuple(sl_end_m1)]) / h
            
            return du

        def deriv2_1d(u, dim):
            # Central difference for interior
            sl_mid = [slice(None)] * u.ndim
            sl_plus = [slice(None)] * u.ndim
            sl_minus = [slice(None)] * u.ndim
            
            sl_mid[dim] = slice(1, -1)
            sl_plus[dim] = slice(2, None)
            sl_minus[dim] = slice(None, -2)
            
            d2u = torch.zeros_like(u)
            d2u[tuple(sl_mid)] = (u[tuple(sl_plus)] - 2*u[tuple(sl_mid)] + u[tuple(sl_minus)]) / (h**2)
            
            # Boundaries - Forward/Backward difference for 2nd derivative (1st order)
            sl_0 = [slice(None)] * u.ndim
            sl_0[dim] = 0
            sl_1 = [slice(None)] * u.ndim
            sl_1[dim] = 1
            sl_2 = [slice(None)] * u.ndim
            sl_2[dim] = 2
            
            d2u[tuple(sl_0)] = (u[tuple(sl_2)] - 2*u[tuple(sl_1)] + u[tuple(sl_0)]) / (h**2)
            
            sl_end = [slice(None)] * u.ndim
            sl_end[dim] = -1
            sl_end_m1 = [slice(None)] * u.ndim
            sl_end_m1[dim] = -2
            sl_end_m2 = [slice(None)] * u.ndim
            sl_end_m2[dim] = -3
            
            d2u[tuple(sl_end)] = (u[tuple(sl_end)] - 2*u[tuple(sl_end_m1)] + u[tuple(sl_end_m2)]) / (h**2)
            
            return d2u

        if self.xdim == 1:
            u_x = deriv_1d(u_grid, 0)
            u_xx = deriv2_1d(u_grid, 0)
            return u_x, u_xx
        elif self.xdim == 2:
            u_x = deriv_1d(u_grid, 0)
            u_y = deriv_1d(u_grid, 1)
            u_xx = deriv2_1d(u_grid, 0)
            u_yy = deriv2_1d(u_grid, 1)
            return u_x, u_y, u_xx, u_yy
        elif self.xdim == 3:
            u_x = deriv_1d(u_grid, 0)
            u_y = deriv_1d(u_grid, 1)
            u_z = deriv_1d(u_grid, 2)
            u_xx = deriv2_1d(u_grid, 0)
            u_yy = deriv2_1d(u_grid, 1)
            u_zz = deriv2_1d(u_grid, 2)
            return u_x, u_y, u_z, u_xx, u_yy, u_zz

    def residual_fdm(self, nn, t, coords, phi, P, gradPphi):
        """
        Compute residual using finite difference method
        Matches interface of residual_ad
        
        Args:
            nn: neural network
            t: time coordinates (Nt, 1)
            coords: list of spatial coordinates
            phi: spatial field
            P: spatial field
            gradPphi: gradients of P*phi
            
        Returns:
            res: residual tensor (Nt, ...)
            u: prediction tensor (Nt, ...)
        """
        # 1. Compute u (prediction) without derivatives
        coord_list = [t] + coords
        f_list = [nn.mlps[i](coord_list[i]) for i in range(self.input_dim)]
        u = torch.einsum(nn.einsum, *f_list) # Shape: (Nt, Nx, Ny, ...) usually
        
        # 2. Prepare for FDM (expecting Spatial dims first, Time last)
        # Permute u to (Nx, Ny, ..., Nt)
        if self.xdim == 1:
            u_grid = u.permute(1, 0) # (Nx, Nt)
        elif self.xdim == 2:
            u_grid = u.permute(1, 2, 0) # (Nx, Ny, Nt)
        elif self.xdim == 3:
            u_grid = u.permute(1, 2, 3, 0) # (Nx, Ny, Nz, Nt)
            
        # 3. Get grid parameters
        # dt from t (Nt, 1)
        dt = (t[1] - t[0]).item()
        
        # h is retrieved from dataset in compute_spatial_derivatives_fd
        
        # 4. Compute derivatives and residuals
        if self.xdim == 1:
            # u_grid: (Nx, Nt)
            u_x, u_xx = self.compute_spatial_derivatives_fd(u_grid)
            
            # Time derivative
            u_t = torch.zeros_like(u_grid)
            u_t[:, 1:] = (u_grid[:,1:] - u_grid[:, :-1]) / dt
            u_t[:, 0] = (u_grid[:, 1] - u_grid[:, 0]) / dt
            
            # Params
            # phi passed is (Nx, 1)
            
            prof = nn.all_params_dict['rRHO'] * self.RHO * phi * u_grid * (1 - u_grid)
            diff = nn.all_params_dict['rD'] * self.DW * (P * phi * u_xx + self.L * gradPphi[0] * u_x)
            
            res_grid = phi * u_t - (prof + diff)
            
        elif self.xdim == 2:
            # u_grid: (Nx, Ny, Nt)
            u_x, u_y, u_xx, u_yy = self.compute_spatial_derivatives_fd(u_grid)
            
            u_t = torch.zeros_like(u_grid)
            u_t[:, :, 1:] = (u_grid[:, :, 1:] - u_grid[:, :, :-1]) / dt
            u_t[:, :, 0] = (u_grid[:, :, 1] - u_grid[:, :, 0]) / dt
            
            # phi is (Nx, Ny). Need (Nx, Ny, 1)
            phi_s = phi.unsqueeze(-1)
            P_s = P.unsqueeze(-1)
            DxPphi_s = gradPphi[0].unsqueeze(-1)
            DyPphi_s = gradPphi[1].unsqueeze(-1)
            
            prof = nn.all_params_dict['rRHO'] * self.RHO * phi_s * u_grid * (1 - u_grid)
            
            laplacian = u_xx + u_yy
            gradient_term = DxPphi_s * u_x + DyPphi_s * u_y
            diff = nn.all_params_dict['rD'] * self.DW * (P_s * phi_s * laplacian + self.L * gradient_term)
            
            res_grid = phi_s * u_t - (prof + diff)
            
        elif self.xdim == 3:
            # u_grid: (Nx, Ny, Nz, Nt)
            u_x, u_y, u_z, u_xx, u_yy, u_zz = self.compute_spatial_derivatives_fd(u_grid)
            
            u_t = torch.zeros_like(u_grid)
            u_t[:, :, :, 1:] = (u_grid[:, :, :, 1:] - u_grid[:, :, :, :-1]) / dt
            u_t[:, :, :, 0] = (u_grid[:, :, :, 1] - u_grid[:, :, :, 0]) / dt
            
            phi_s = phi.unsqueeze(-1)
            P_s = P.unsqueeze(-1)
            DxPphi_s = gradPphi[0].unsqueeze(-1)
            DyPphi_s = gradPphi[1].unsqueeze(-1)
            DzPphi_s = gradPphi[2].unsqueeze(-1)
            
            prof = nn.all_params_dict['rRHO'] * self.RHO * phi_s * u_grid * (1 - u_grid)
            
            laplacian = u_xx + u_yy + u_zz
            gradient_term = DxPphi_s * u_x + DyPphi_s * u_y + DzPphi_s * u_z
            diff = nn.all_params_dict['rD'] * self.DW * (P_s * phi_s * laplacian + self.L * gradient_term)
            
            res_grid = phi_s * u_t - (prof + diff)

        # 5. Permute back to (Nt, ...)
        if self.xdim == 1:
            res = res_grid.permute(1, 0)
        elif self.xdim == 2:
            res = res_grid.permute(2, 0, 1)
        elif self.xdim == 3:
            res = res_grid.permute(3, 0, 1, 2)
            
        # 6. Apply weighting
        res = self.apply_loss_weighting(res, t)
        
        return res, u

    def fdmbcloss(self, net):
        # get dirichlet boundary condition loss for FDM
        # Zero boundary conditions - just square the boundary values
        
        # Get network predictions on grid
        # self.upred_res is (Nt, Nx, Ny, ...)
        u_grid = self.upred_res
        
        # Permute to (Nx, Ny, ..., Nt) for boundary slicing
        if self.xdim == 1:
            u_grid = u_grid.permute(1, 0) # (Nx, Nt)
        elif self.xdim == 2:
            u_grid = u_grid.permute(1, 2, 0) # (Nx, Ny, Nt)
        elif self.xdim == 3:
            u_grid = u_grid.permute(1, 2, 3, 0) # (Nx, Ny, Nz, Nt)
        
        if self.xdim == 1:
            # Collect boundary values: left and right edges for all time
            u_boundary = torch.cat([u_grid[0, :], u_grid[-1, :]])  # (2*nt,)
            
        elif self.xdim == 2:
            # Collect boundary values: all four edges for all time
            u_left  = u_grid[0, :, :].flatten()     # Left boundary
            u_right = u_grid[-1, :, :].flatten()   # Right boundary  
            u_bot   = u_grid[:, 0, :].flatten()   # Bottom boundary
            u_top   = u_grid[:, -1, :].flatten()     # Top boundary
            
            u_boundary = torch.cat([u_left, u_right, u_bot, u_top])
        elif self.xdim == 3:
            # Collect boundary values: all six faces for all time
            u_left  = u_grid[0, :, :, :].flatten()     # Left boundary
            u_right = u_grid[-1, :, :, :].flatten()   # Right boundary  
            u_bot   = u_grid[:, 0, :, :].flatten()   # Bottom boundary
            u_top   = u_grid[:, -1, :, :].flatten()     # Top boundary
            u_front = u_grid[:, :, 0, :].flatten()     # Front boundary
            u_back  = u_grid[:, :, -1, :].flatten()     # Back boundary
            
            u_boundary = torch.cat([u_left, u_right, u_bot, u_top, u_front, u_back])
        else:
            raise ValueError(f"Unsupported spatial dimension: {self.xdim}. Only 1D, 2D and 3D are supported.")
        
        # Zero boundary condition: just square the boundary values
        loss = torch.mean(u_boundary**2)
        
        return loss
    
    def get_seg1_loss(self, net):
        data = self.dataset.batch['dat']
        u1 = data['u1']
        phi = data['phi']
        
        # Prediction at t=1.0
        t_target = torch.ones(1, 1, device=self.dataset.device)
        coords = data['coords']
        coord_list = [t_target] + coords
        
        upred = net(coord_list)
        upred = upred.squeeze(0) # (Nx, Ny, Nz)
        
        pred_seg = sigmoid_binarize(upred, net.all_params_dict['th1'])
        loss = phimse(pred_seg, u1, phi)/torch.mean(u1**2)
        return loss

    def get_seg2_loss(self, net):
        data = self.dataset.batch['dat']
        u2 = data['u2']
        phi = data['phi']
        
        # Prediction at t=1.0
        t_target = torch.ones(1, 1, device=self.dataset.device)
        coords = data['coords']
        coord_list = [t_target] + coords
        
        upred = net(coord_list)
        upred = upred.squeeze(0) # (Nx, Ny, Nz)
        
        pred_seg = sigmoid_binarize(upred, net.all_params_dict['th2'])
        loss = phimse(pred_seg, u2, phi)/torch.mean(u2**2)
        return loss
    
    def get_uchar_res_loss(self, net):
        data = self.dataset.batch['dat']
        u = data['uchar']
        phi = data['phi']
        
        t = data['t']
        coords = data['coords']
        coord_list = [t] + coords
        
        upred = net(coord_list)
        
        return phimse(upred, u, phi.unsqueeze(0))

    def get_uchar_dat_loss(self, net):
        raise NotImplementedError('not implemented yet')
        

    def get_ugt_res_loss(self, net):
        data = self.dataset.batch['dat']
        u = data['ugt']
        phi = data['phi']
        
        t = data['t']
        coords = data['coords']
        coord_list = [t] + coords
        
        upred = net(coord_list)
        
        return phimse(upred, u, phi.unsqueeze(0))

    def get_ugt_dat_loss(self, net):
        raise NotImplementedError('not implemented yet')

    def get_ic_loss(self, net):
        # get initial condition loss
        # reuse u from residual calculation at t=0
        # u_pred at t=0 (assuming t starts at 0 in res batch)
        u0_pred = self.upred_res[0] 
        
        data = self.dataset.batch['res']
        coords = data['coords']
        phi = data['phi']
        
        # Compute r2 on grid
        L = self.L
        r2 = 0
        for i, c in enumerate(coords):
            # Reshape c to have 1s in other dimensions
            # coords[0] is x (dim 0 of u0_pred), coords[1] is y (dim 1), etc.
            shape = [1] * len(coords)
            shape[i] = -1
            c_reshaped = c.reshape(shape)
            r2 = r2 + (c_reshaped * L)**2
            
        u0_gt = self.h_init * torch.exp(-self.r_init * r2)
        
        return phimse(u0_pred, u0_gt, phi)

    def grid_forward(self, net, t_array):
        """
        Transform grid tensor to coordinate arrays for network forward pass
        """
        # SPINN logic
        t_tensor = t_array.to(self.dataset.device).reshape(-1, 1)
        coords = []
        if 'gx_scaled' in self.dataset:
            coords.append(self.dataset['gx_scaled'].clone().detach().to(self.dataset.device).reshape(-1, 1))
        if 'gy_scaled' in self.dataset:
            coords.append(self.dataset['gy_scaled'].clone().detach().to(self.dataset.device).reshape(-1, 1))
        if 'gz_scaled' in self.dataset:
            coords.append(self.dataset['gz_scaled'].clone().detach().to(self.dataset.device).reshape(-1, 1))
        
        coord_list = [t_tensor] + coords
        
        # Assuming pde_params_dict is empty or not needed for now
        u_grid = net(coord_list)
        
        # u_grid is (nt, nx, ny, nz)
        # grid_forward expects (nx, ny, nz, nt) (Matlab order)
        # So permute
        if self.xdim == 1:
            # (nt, nx) -> (nx, nt)
            u_grid = u_grid.permute(1, 0)
        elif self.xdim == 2:
            # (nt, nx, ny) -> (nx, ny, nt)
            u_grid = u_grid.permute(1, 2, 0)
        elif self.xdim == 3:
            # (nt, nx, ny, nz) -> (nx, ny, nz, nt)
            u_grid = u_grid.permute(1, 2, 3, 0)
            
        return u_grid
