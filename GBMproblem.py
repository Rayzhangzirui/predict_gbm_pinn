#!/usr/bin/env python
# define problems for PDE
import os
import torch

from Options import *
from util import *
from DenseNet import DenseNet
from GBMDataset import GBMDataset
from GBMplot import *

def sumcol(A):
    # sum along column
    return torch.sum(A, dim=1, keepdim=True)


def sigmoid_binarize(x, th):
    # smooth heaviside function using a sigmoid
    K = 20
    return torch.nn.functional.sigmoid(K*(x-th))

def phimse(x,y,phi):
    return torch.mean(((x-y)*phi)**2)

def segmseloss(upred, udat, phi, th):    
    '''spatial segmentation loss by mse'''
    uth = sigmoid_binarize(upred,th)
    return phimse(uth, udat, phi)

def dice(seg_pred, seg_gt):
    # dice
    tp = torch.sum(seg_pred * seg_gt)
    fp = torch.sum(seg_pred * (1-seg_gt))
    fn = torch.sum((1-seg_pred) * seg_gt)
    dice = 2*tp/(2*tp+fp+fn)
    return dice

def r2(X):
    # return r^2 in non-dimensional scale
    r2 = sumcol(torch.square((X[:, 1:])))
    return r2

def range_penalty(x, xmin, xmax):
    # relu loss
    return torch.nn.functional.relu(x - xmax)**2 + torch.nn.functional.relu(xmin - x)**2

def reg_penalty(x, ref):
    # relu loss
    return (x - ref)**2

class GBMproblem():
    def __init__(self, xdim, pde_opts: dict):
        
        self.opts = pde_opts
        

        self.prior_th1 = pde_opts.get('prior_th1', 0.3)
        self.prior_th2 = pde_opts.get('prior_th2', 0.6)
        
        ### GBM custom loss
        self.loss_dict = {'res': self.resloss}
        self.loss_dict['seg1'] = self.get_seg1_loss
        self.loss_dict['seg2'] = self.get_seg2_loss


        self.loss_dict['uchar_res'] = self.get_uchar_res_loss
        self.loss_dict['uchar_dat'] = self.get_uchar_dat_loss
        self.loss_dict['ugt_res'] = self.get_ugt_res_loss
        self.loss_dict['ugt_dat'] = self.get_ugt_dat_loss
        

        # regularization of the parameters
        self.loss_dict['rD_reg'] = self.regD_loss
        self.loss_dict['rRHO_reg'] = self.regRHO_loss
        self.loss_dict['th1_reg'] = self.regth1_loss
        self.loss_dict['th2_reg'] = self.regth2_loss


        self.loss_dict['thrange'] = self.range_constraint

        
        
        # finite difference method options
        self.use_fdm = self.opts.get('use_fdm', False)  # default to automatic differentiation

        # range of th1 and th2
        self.th1_range = self.opts['th1_range']
        self.th2_range = self.opts['th2_range']
        self.rD_range = self.opts['rD_range']
        self.rRHO_range = self.opts['rRHO_range']

        ### dimension
        self.xdim = xdim
        self.input_dim = self.xdim + 1 # add time dimension
        self.output_dim = 1


        # get parameter from mat file
        # check empty string
        # inititalize parameters 
        self.init_param = {}
        self.init_param['rD'] = self.opts['init_param']['rD'] if 'rD' in self.opts['init_param'] else 1.0
        self.init_param['rRHO'] = self.opts['init_param']['rRHO'] if 'rRHO' in self.opts['init_param'] else 1.0
        self.init_param['th1'] = self.opts['init_param']['th1'] if 'th1' in self.opts['init_param'] else self.prior_th1
        self.init_param['th2'] = self.opts['init_param']['th2'] if 'th2' in self.opts['init_param'] else self.prior_th2

        self.all_params_dict = self.init_param.copy()

        self.pde_params = ['rD', 'rRHO']

    def config_traintype(self, traintype):
        pass

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        # first copy self.pde.param, which include all pde-param in network
        # then update by init_param if provided
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        net = DenseNet(**kwargs,
                        lambda_transform = self.lambda_transform,
                        all_params_dict = self.all_params_dict,
                        trainable_param = self.opts['trainable_param'])
        return net
    
    # for testing purpose, still need to set whichdata
    def get_uchar_res_loss(self, net):
        # mse of uchar_res
        data = self.dataset.batch['st']
        X = data['X_st']
        u = data['uchar_st']
        phi = data['phi_st']
        upred = net(X)
        return phimse(upred, u, phi)
    
    def get_uchar_dat_loss(self, net):
        # mse of uchar_dat
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        u = data['uchar_dat_train']
        phi = data['phi_dat_train']
        upred = net(X)
        return phimse(upred, u, phi)
    
    def get_ugt_res_loss(self, net):
        # mse of ugt_res
        data = self.dataset.batch['st']
        X = data['X_st']
        u = data['ugt_st']
        phi = data['phi_st']
        upred = net(X)
        return phimse(upred, u, phi)
    
    def get_ugt_dat_loss(self, net):
        # mse of ugt_dat
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        u = data['ugt_dat_train']
        phi = data['phi_dat_train']
        upred = net(X)
        return phimse(upred, u, phi)

    def regD_loss(self, net):
        # regularization loss for rD
        rD = net.all_params_dict['rD'].squeeze()
        return reg_penalty(rD, 1.0)

    def regRHO_loss(self, net):
        # regularization loss for rRHO
        rRHO = net.all_params_dict['rRHO'].squeeze()
        return reg_penalty(rRHO, 1.0)
    
    def regth1_loss(self, net):
        # regularization loss for th1
        th1 = net.all_params_dict['th1'].squeeze()
        return reg_penalty(th1, self.prior_th1)

    def regth2_loss(self, net):
        # regularization loss for th2
        th2 = net.all_params_dict['th2'].squeeze()
        return reg_penalty(th2, self.prior_th2)

    # compute validation statistics
    @torch.no_grad()
    def validate(self, nn):
        '''compute err '''
        v_dict = {}
        for vname in nn.trainable_param:
            v_dict[vname] = nn.all_params_dict[vname]
            # if vname in self.gt_param and vname in nn.all_params_dict:
            #     err = torch.abs(nn.all_params_dict[vname] - self.gt_param[vname])
            #     v_dict[f'abserr_{vname}'] = err
        return v_dict

    def ic(self, X, L):
        # initial condition
        r2 = sumcol(torch.square((X[:, 1:self.input_dim])*L)) # this is in pixel scale, unit mm, 
        return self.h_init*torch.exp(-self.r_init*r2)


    def residual(self, nn, X, phi, P, gradPphi):
        
        # Get the number of dimensions
        n = X.shape[0]

        # split each column of X into a separate tensor of size (n, 1)
        vars = [X[:, d:d+1] for d in range(self.input_dim)]
        
        t = vars[0].detach()  # time variable detached for weighting if needed
        
        
        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat(vars, dim=1)
       
        # Forward pass through the network
        u = nn(nn_input)
        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u, vars[0], grad_outputs=v, create_graph=True)[0]

        # n by d matrix
        u_x = torch.zeros(n, self.xdim, device=X.device)
        u_xx = torch.zeros(n, self.xdim, device=X.device)

        for d in range(0, self.xdim):
            u_x[:,d:d+1] = torch.autograd.grad(u, vars[d+1], grad_outputs=v, create_graph=True)[0]
            u_xx[:,d:d+1] = torch.autograd.grad(u_x[:,d:d+1], vars[d+1], grad_outputs=v, create_graph=True)[0]
        
        prof = nn.all_params_dict['rRHO'] * self.RHO * phi * u * ( 1 - u)
        diff = nn.all_params_dict['rD'] * self.DW * (P * phi * sumcol(u_xx) + self.L * sumcol(gradPphi * u_x))
        res = phi * u_t - (prof + diff)

        w = 1.0
        if self.opts['causal_weight']:
            # res_grid_sqr = res_grid**2
            res_p2 = res.detach()**2
            # cumulative sum along t, this is okay because t is sorted
            res_cumsum = torch.cumsum(res_p2, dim=0)
            # normalize so that min is 0
            res_cumsum = res_cumsum - torch.min(res_cumsum)
            w_t = torch.sqrt(torch.exp(-self.opts['texp_weight'] * res_cumsum))

        else:
            w_t = torch.sqrt(torch.exp(-self.opts['texp_weight'] * t))  # time weighting exp(-w*t), sqrt because res is squared later
        

        r = torch.sqrt(r2(X[:, 1:])).detach()  # r detached for weighting if needed
        r = r/torch.max(r)
        w_r = 1e-3 + torch.pow(r, self.opts['rpow_weight'])  # radial weighting 1/(1e-3 + r^p)
        w_r = torch.pow(w_r, -0.5)
        
        res = res * w_t * w_r

        return res, u

    def resloss(self, net):
        self.res, self.upred_res = self.get_res_pred(net)
        val_loss_res = torch.mean(torch.square(self.res))
        return val_loss_res
    
    def get_res_pred(self, net):
        # Use automatic differentiation (original method)
        data = self.dataset.batch['res']
        X = data['X_res_train']
        X.requires_grad_(True)
        
        phi = data['phi_res_train']
        P = data['P_res_train']
        gradPphi = data['gradPphi_res_train']
        res, u_pred = self.residual(net, X, phi, P, gradPphi)
        self.res = res
        self.upred_res = u_pred
        return res, u_pred


    def get_seg1_loss(self, net):
        # get segmentation loss of u1
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u1 = data['u1_dat_train']
        upred = net(X)
        pred_seg = sigmoid_binarize(upred, net.all_params_dict['th1'])
        loss = phimse(pred_seg, u1, phi)/torch.mean(u1**2)
        return loss
    
    def get_seg2_loss(self, net):
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u2 = data['u2_dat_train']
        upred = net(X)
        pred_seg = sigmoid_binarize(upred, net.all_params_dict['th2'])
        loss = phimse(pred_seg, u2, phi)/torch.mean(u2**2)
        return loss
    
    def bcloss(self, net):
        # get dirichlet boundary condition loss
        raise NotImplementedError
    

    def range_constraint(self, net):
        # range constraint for th1 and th2

        th1 = net.all_params_dict['th1'].squeeze()
        th2 = net.all_params_dict['th2'].squeeze()

        loss = range_penalty(th1, self.th1_range[0], self.th1_range[1]) + range_penalty(th2, self.th2_range[0], self.th2_range[1])
        return loss
    
    def print_info(self):
        pass

    
    
    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        self.dataset.to_device(self.dataset.device)
        net.to(self.dataset.device)

        upred_grid = self.grid_forward(net, self.dataset['t_scaled'])

        self.names2save = ['upred_final', 'rD_pred', 'rRHO_pred', 'th1_pred', 'th2_pred']
        self.dataset['upred'] = upred_grid  # only save the last time point
        self.dataset['upred_final'] = upred_grid[..., -1]  # only save the last time point
        # prediction of parameters
        self.dataset['rD_pred'] = net.all_params_dict['rD']
        self.dataset['rRHO_pred'] = net.all_params_dict['rRHO']
        self.dataset['th1_pred'] = net.all_params_dict['th1']
        self.dataset['th2_pred'] = net.all_params_dict['th2']


    def grid_forward(self, net, t_array):
        """
        Transform grid tensor to coordinate arrays for network forward pass
        
        Args:
            t_array: nt x 1 time points
            net: neural network
            
        Returns:
            u_grid: (nx, nt) for 1D, (nx, ny, nt) for 2D, (nx, ny, nz, nt) for 3D
                    Dimension order follows MATLAB ndgrid convention: (x, y, z, t)
        """
        t_array = t_array.view(-1, 1)  # ensure shape (nt, 1)
        nt = int(t_array.shape[0])

        # Spatial axes
        gx = self.dataset['gx_scaled']
        nx = int(gx.shape[0])

        if self.xdim >= 2:
            gy = self.dataset['gy_scaled']
            ny = int(gy.shape[0])
        if self.xdim >= 3:
            gz = self.dataset['gz_scaled']
            nz = int(gz.shape[0])

        # Precompute flattened spatial coordinates 
        # gx/gy/gz_mesh defiend in GBMDataset following MATLAB ndgrid (Fortran order).
        # note that reshape(-1,1) here give order that is different from matlab, but okay when we do the same reshap back
        if self.xdim == 1:
            space_coords = self.dataset['gx_mesh'].reshape(-1, 1)  # (nx, 1)
            n_space = nx
        elif self.xdim == 2:
            Xg = self.dataset['gx_mesh']  # (nx, ny)
            Yg = self.dataset['gy_mesh']  # (nx, ny)
            # Fortran-order flattening: permute dims then flatten in C-order
            perm = (1, 0)
            x_flat = Xg.reshape(-1, 1)
            y_flat = Yg.reshape(-1, 1)
            space_coords = torch.cat([x_flat, y_flat], dim=1)  # (nx*ny, 2)
            n_space = nx * ny
        elif self.xdim == 3:
            Xg = self.dataset['gx_mesh']  # (nx, ny, nz)
            Yg = self.dataset['gy_mesh']  # (nx, ny, nz)
            Zg = self.dataset['gz_mesh']  # (nx, ny, nz)
            perm = (2, 1, 0)
            x_flat = Xg.reshape(-1, 1)
            y_flat = Yg.reshape(-1, 1)
            z_flat = Zg.reshape(-1, 1)
            space_coords = torch.cat([x_flat, y_flat, z_flat], dim=1)  # (nx*ny*nz, 3)
            n_space = nx * ny * nz
        else:
            raise ValueError(f"Unsupported spatial dimension: {self.xdim}. Only 1D/2D/3D are supported.")

    
        # Order preserved: for each time, list all space points.
        # Avoid materializing the full X_input (and even full X_input_t) to reduce peak memory.
        batch_size = 1_000_000
        u_chunks = []
        for it in range(nt):
            t_val = t_array[it:it + 1]  # (1, 1)
            for s in range(0, n_space, batch_size):
                e = min(s + batch_size, n_space)
                xs = space_coords[s:e]
                t_rep = t_val.expand(int(xs.shape[0]), 1)
                X = torch.cat([t_rep, xs], dim=1)
                u_chunks.append(net(X))

        u_flat = torch.cat(u_chunks, dim=0)

        if self.opts.get('force_bc', False):
            # Mask values outside radius 1.0
            r_sq = torch.sum(space_coords**2, dim=1)
            mask = r_sq > self.r2max
            # Repeat mask for all time steps
            mask = mask.repeat(nt)
            u_flat[mask] = 0.0
        
        if self.xdim == 1:
            u_grid = u_flat.view(nt,nx).permute(1, 0)  # (nx, nt)
            # set boundary values to zero
            u_grid[0, :] = 0.0
            u_grid[-1, :] = 0.0
        elif self.xdim == 2:
            u_grid = u_flat.view(nt,nx,ny).permute(1, 2, 0)  # (nx, ny, nt)
            u_grid[0, :, :] = 0.0
            u_grid[-1, :, :] = 0.0
            u_grid[:, 0, :] = 0.0
            u_grid[:, -1, :] = 0.0
        else:
            u_grid = u_flat.view(nt,nx,ny,nz).permute(1, 2, 3, 0)  # (nx, ny, nz, nt)
            u_grid[0, :, :, :] = 0.0
            u_grid[-1, :, :, :] = 0.0
            u_grid[:, 0, :, :] = 0.0
            u_grid[:, -1, :, :] = 0.0
            u_grid[:, :, 0, :] = 0.0
            u_grid[:, :, -1, :] = 0.0

        return u_grid

    
    @error_logging_decorator
    def visualize(self, savedir=None, prefix=''):
        # visualize the results
        self.dataset.to_np()

        th1_pred = self.dataset.get('th1_pred', None)
        th2_pred = self.dataset.get('th2_pred', None)

        # Prediction grid uses ndgrid convention with dims (x[,y[,z]], t)
        ugrid = self.dataset['upred']  # shape (nx[,ny[,nz]], nt)
        phiugrid = ugrid * self.dataset['phi'][..., np.newaxis]  # apply phi mask
        nt_pred = int(phiugrid.shape[-1])
        t_indices = np.linspace(0, nt_pred - 1, 5).astype(int)

        # Optional reference on the same grid (gt/char only)
        ref_key = None
        uref = None
        if 'gt' in self.whichdata and 'ugt' in self.dataset:
            uref = self.dataset['ugt']
            phiuref = uref * self.dataset['phi'][..., np.newaxis]
        elif 'char' in self.whichdata and 'uchar' in self.dataset:
            uref = self.dataset['uchar']
            phiuref = uref * self.dataset['phi'][..., np.newaxis]
        else:
            pass
        

        # scatter plot of solution at t_indices vs radius
        space_mesh = ("gx_mesh", "gy_mesh", "gz_mesh")[:self.xdim]
        X_space_full = np.stack([self.dataset[m].ravel(order="F") for m in space_mesh], axis=1)
        r_full = np.linalg.norm(X_space_full, axis=1)
        # if too many points, randomly sample 10000 points for scatter plot
        
        if r_full.shape[0] > 10000:
            idx = np.random.choice(r_full.shape[0], size=10000, replace=False)
        else:
            idx = np.arange(r_full.shape[0])

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('viridis')
        for it in t_indices:
            phiu_it = phiugrid[..., it].ravel(order="F")
            phiuref_it = phiuref[..., it].ravel(order="F") if uref is not None else None
            t = self.dataset['t_scaled'][it]
            color = cmap(t)
            ax.scatter(r_full[idx], phiu_it[idx], color=color, marker='o',label=f'pred t={t:.2f}')
            if phiuref_it is not None:
                ax.scatter(r_full[idx], phiuref_it[idx], color=color, marker='x',label=f'ref t={t:.2f}')
            ax.legend(loc='upper right')
            # add grid
            ax.grid(True)
            ax.set_xlabel('Radius r')
            ax.set_ylabel('u')
            min_u = min(np.min(phiu_it), np.min(phiuref_it) if phiuref_it is not None else 0.0)
            ax.set_ylim([min_u, 1.0])
            ax.set_title(f'{prefix} Scatter Plot of u vs Radius')
        
        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_{prefix}_scatter_u_vs_r.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        
        # contour and imshow plotting functions
        if self.xdim == 1:
            # For 1D, plot line profiles at different times
            xgrid = self.dataset['gx'].squeeze()
                
            fig, ax = plt.subplots(figsize=(8, 6))
            for it in t_indices:
                t_val = it / max(1, (nt_pred - 1))  # normalize time to [0,1]
                ax.plot(xgrid, phiugrid[:, it], label=f't={t_val:.2f}')
                if phiuref is not None:
                    ax.plot(xgrid, phiuref[:, it], '--', label=f't={t_val:.2f} (ref)')
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title(f'{prefix} Grid Solution at Different Times')
            ax.legend()
            ax.grid(True)
            
            if savedir is not None:
                fpath = os.path.join(savedir, f'fig_{prefix}_1d.png')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')
            plt.close(fig)
        
        # 2D and 3D visualization moved to Engine.py

        plt.close('all')
    
    def setup_dataset(self, traindat_path, whichdata, device):
        ''' downsample for training'''

        # configure data
        assert whichdata in ['char', 'gt', 'pat'], f'whichdata {whichdata} not recognized'
        # for patient simulation, no need to load ugt and uchar into memory
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

        # setup transformation
        self.lambda_transform = torch.nn.Module()
        self.lambda_transform.register_buffer('L', torch.tensor(self.L))
        if self.opts['force_bc'] == True:
            # force bc same as ic at rmax
            print(f'force bc at r2max = {self.r2max}')
            self.lambda_transform.register_buffer('r2max', torch.tensor(self.r2max))
            self.lambda_transform.forward = lambda X, u: (self.ic(X, self.lambda_transform.L) + u * X[:,0:1]) * (self.lambda_transform.r2max - r2(X))/self.lambda_transform.r2max
        else:
            self.lambda_transform.forward = lambda X, u: self.ic(X, self.lambda_transform.L) + u * X[:,0:1]
        
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
        if self.whichdata in ['gt', 'char']:
            include_st = True
            self.dataset.prepare_all_spacetime_field()
        else:
            include_st = False        
        # put all variables to device (iterable samplers sample indices on GPU)
        self.dataset.to_device(device)


        res_batch_size = int(self.opts['res_batch_size'])
        dat_batch_size = int(self.opts['dat_batch_size'])
        
        # for residual batch size, the larger the better, no limit
        # for data loss batch size, cannot be larger than dataset size
        if dat_batch_size > self.dataset['X_space_flat'].shape[0]:
            dat_batch_size = self.dataset['X_space_flat'].shape[0]
            self.opts['dat_batch_size'] = dat_batch_size
            print(f'Adjusted dat_batch_size to {dat_batch_size} due to dataset size.')

        # Iterable samplers already randomize; keep workers=0 since tensors live on GPU.
        self.dataset.configure_dataloader(res_batch_size, dat_batch_size, include_st)