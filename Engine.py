#!/usr/bin/env python
import os
from torchinfo import summary
import nibabel as nib
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from gbmutil import est_char_param, get_char
from solve_phase_field_torch import solve_phase_field_torch
from solve_fisher_kpp_torch import solve_fisher_kpp_torch
from gen_samples import gen_samples
from GBMplot import plot_contour_over_seg, plot_density_over_seg, plot_plan_comparison, plot_grid_imshow_panels, plot_grid_contour_overlay
from evaluate import evaluate_personalized_plan, create_standard_plan, evaluate_prediction

from Options import *
from util import *
from GBMproblem import GBMproblem
from sepGBMproblem import sepGBMproblem

from Logger import Logger
from Trainer import Trainer

class Engine:
    def __init__(self, opts=None) -> None:

        self.device = set_device(opts['device'])
        self.opts = opts
        self.restore_artifacts = {}
        self.logger = None

        self.logger = None
        self.trainer = None
        self.pdata = {} # patient related data
        self.z_slice = None # For 3D visualization

        # for controlling logging of standard plan metrics
        self.std_metrics_logged = False

        # dictionary for storing variables for visualization
        self.vdict = {}

        set_seed(self.opts['seed'])

        self.logger = Logger(self.opts['logger_opts'])
        self.logger.log_options(self.opts)


    def restore_opts(self, restore_opts):
        ''' only restore neural network options, other options are from command line or default
        '''
        for key in restore_opts['nn_opts']:
            self.opts['nn_opts'][key] = restore_opts['nn_opts'][key]
        
    def restore_run(self):
        # if restore is empty, do nothing
        if self.opts['restore'] != '':
            # if is director
            if os.path.isdir(self.opts['restore']):
                path = self.opts['restore']
                opts_path = os.path.join(path, 'options.json')
                restore_opts = read_json(opts_path)
                self.restore_artifacts = {fname: os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.pth')}
                self.restore_artifacts['artifacts_dir'] = path
                print(f'restore from directory {path}')

            else:
                #  restore from exp_name:run_name
                self.restore_artifacts = self.logger.load_artifact(name_str=self.opts['restore'])
                restore_opts = read_json(self.restore_artifacts['options.json'])
                print(f'restore from {self.opts["restore"]}')
        
            self.restore_opts(restore_opts)
        else:
            print('no restore')
        
    
    def _load_raw_nifti(self):
        def convert_to_int(z):
            return np.rint(z).astype(np.int32)
        
        dopts = self.opts['dataset_opts']
        input_dir = Path(dopts['input_dir']) if dopts['input_dir'] else None

        # Helper to resolve path: use explicit option if present, else fallback to input_dir/default_name
        def get_path(opt_key, default_name):
            if dopts.get(opt_key):
                return Path(dopts[opt_key])
            if input_dir:
                return input_dir / default_name
            return None

        path_csf = get_path('csf_file', 'csf_pbmap.nii.gz')
        path_gm = get_path('gm_file', 'gm_pbmap.nii.gz')
        path_wm = get_path('wm_file', 'wm_pbmap.nii.gz')
        path_seg = get_path('segpre_file', 'tumor_seg.nii.gz')
        path_rec = get_path('segrec_file', 'recurrence_preop.nii.gz')
        path_mask = get_path('brain_mask_file', 't1c_bet_mask.nii.gz')

        if not all([path_csf, path_gm, path_wm, path_seg]):
             raise ValueError("Missing required input files (CSF, GM, WM, TumorSeg). Please provide paths or input_dir.")

        Pcsf = nib.load(path_csf).get_fdata()
        Pgm = nib.load(path_gm).get_fdata()
        Pwm = nib.load(path_wm).get_fdata()
        
        seg = nib.load(path_seg).get_fdata()
        seg = convert_to_int(seg)

        if path_rec and path_rec.exists():
            rec = nib.load(path_rec).get_fdata()
            rec = convert_to_int(rec)
        else:
            rec = None 
            
        if path_mask and path_mask.exists():
            brain_mask = nib.load(path_mask).get_fdata()
            brain_mask = convert_to_int(brain_mask)
        else:
            brain_mask = None

        return Pcsf, Pgm, Pwm, seg, rec, brain_mask

    def _setup_data(self, Pcsf, Pgm, Pwm, seg, rec, brain_mask):
        dopts = self.opts['dataset_opts']
        xdim = dopts.get('xdim', 3)
        
        # if  z-slice is -1, set the slice with largest WT area
        self.z_slice = dopts['z_slice']
        if dopts['z_slice'] == -1:
            mask_wt = np.isin(seg, [1, 2, 3])
            self.z_slice = np.argmax(np.sum(mask_wt, axis=(0, 1)))
        
        # slice data for 2D
        if xdim == 2:            
            self.pdata = {
                'Pcsf': Pcsf[:, :, self.z_slice],
                'Pgm': Pgm[:, :, self.z_slice],
                'Pwm': Pwm[:, :, self.z_slice],
            }
            
            self.pdata['mask_wt'] = np.isin(seg[:, :, self.z_slice], [1, 2, 3]).astype(float)
            self.pdata['mask_tc'] = np.isin(seg[:, :, self.z_slice], [1, 3]).astype(float)
            self.pdata['seg'] = seg[:, :, self.z_slice]
            
            if brain_mask is not None:
                self.pdata['brain_mask'] = brain_mask[:, :, self.z_slice]
            else:
                # Fallback if no brain mask
                self.pdata['brain_mask'] = (self.pdata['Pwm'] + self.pdata['Pgm'] + self.pdata['Pcsf'] > 0.01).astype(float)

            if rec is not None:
                self.pdata['mask_rec'] = (rec[:, :, self.z_slice] > 0).astype(float)
            
            self.slices_to_plot = [(None, '2d')]
                
        else:
            # use full 3D data
            self.pdata = {
                'Pcsf': Pcsf,
                'Pgm': Pgm,
                'Pwm': Pwm,
            }
            self.pdata['mask_wt'] = np.isin(seg, [1, 2, 3]).astype(float)
            self.pdata['mask_tc'] = np.isin(seg, [1, 3]).astype(float)
            self.pdata['seg'] = seg
            
            if brain_mask is not None:
                self.pdata['brain_mask'] = brain_mask
            else:
                self.pdata['brain_mask'] = (self.pdata['Pwm'] > 0).astype(float)

            if rec is not None:
                self.pdata['mask_rec'] = (rec > 0).astype(float)

            # for tumor region, rescale Pwm and Pgm to 1
            tumor_region = self.pdata['mask_wt'] > 0
            total_pbmap = self.pdata['Pwm'] + self.pdata['Pgm']
            pos_region = tumor_region & (total_pbmap > 1e-6)
            self.pdata['Pwm'][pos_region] /= total_pbmap[pos_region]
            self.pdata['Pgm'][pos_region] /= total_pbmap[pos_region]
            # for zero pbmap region, set Pwm=Pgm=0.5
            zero_region = tumor_region & (total_pbmap <= 1e-6)
            self.pdata['Pwm'][zero_region] = 0.5
            self.pdata['Pgm'][zero_region] = 0.5
            
        # Calculate tumor center (center_tc logic)
        # TC: labels 1, 3. mask_tc is already computed as np.isin(seg, [1, 3])
        if np.any(self.pdata['mask_tc']):
            center_region = self.pdata['mask_tc']
        else:
            print("TC mask is empty, using WT mask for centroid estimation.")
            center_region = self.pdata['mask_wt']
            
        coords = np.argwhere(center_region)
        # (1,1,1)-based indexing
        self.pdata['x0'] = np.mean(coords + 1.0, axis=0)
            
        # Calculate interesting z-slices for visualization
        self.slices_to_plot = []
        
        # 1. Max Pre-op TC (seg in [1, 3])
        mask_tc_pre = np.isin(seg, [1, 3])
        z_tc_pre = np.argmax(np.sum(mask_tc_pre, axis=(0, 1)))
        self.slices_to_plot.append((z_tc_pre, 'max_pre_tc'))
        
        # 2. Max Pre-op WT (seg in [1, 2, 3])
        mask_wt_pre = np.isin(seg, [1, 2, 3])
        z_wt_pre = np.argmax(np.sum(mask_wt_pre, axis=(0, 1)))
        self.slices_to_plot.append((z_wt_pre, 'max_pre_wt'))
        
        # 3. Max Post-op TC/WT (rec)
        if rec is not None:
            if np.max(rec) > 1:
                mask_tc_post = np.isin(rec, [1, 3])
                z_tc_post = np.argmax(np.sum(mask_tc_post, axis=(0, 1)))
                self.slices_to_plot.append((z_tc_post, 'max_post_tc'))
                
                mask_wt_post = np.isin(rec, [1, 2, 3])
                z_wt_post = np.argmax(np.sum(mask_wt_post, axis=(0, 1)))
                self.slices_to_plot.append((z_wt_post, 'max_post_wt'))
            else:
                z_rec = np.argmax(np.sum(rec > 0, axis=(0, 1)))
                self.slices_to_plot.append((z_rec, 'max_post_rec'))

    def _ensure_phi(self):
        # for 3D, save and reload if exit,
        # for 2d, always compute
        dopts = self.opts['dataset_opts']
        output_dir = Path(self.logger.get_dir())
        xdim = dopts.get('xdim', 3)
        
        phi = None
        # If 3D, try to load first
        if xdim == 3:
            phi_path = output_dir / "phi.mat"
            if phi_path.exists():
                print(f"Loading phi from {phi_path}")
                phi = sio.loadmat(phi_path)['phi']
        
        # If not loaded (or 2D), compute it
        if phi is None:
            phi = solve_phase_field_torch(
                self.pdata['Pwm'], self.pdata['Pgm'], self.pdata['Pcsf'], 
                numiter=100, epsilon=3, device=self.device
            )
            
            # If 3D, save it
            if xdim == 3:
                output_dir.mkdir(parents=True, exist_ok=True)
                sio.savemat(output_dir / "phi.mat", {'phi': phi})
                print(f"Saved phase field to {output_dir / 'phi.mat'}")
        
        self.pdata['phi'] = phi

    def load_geometry(self):
        """Ensure geometry (Pwm, phi, etc.) is loaded for FDM simulation."""
        if 'Pwm' in self.pdata and 'phi' in self.pdata:
            return

        print("Loading geometry")
        Pcsf, Pgm, Pwm, seg, rec, brain_mask = self._load_raw_nifti()
        self._setup_data(Pcsf, Pgm, Pwm, seg, rec, brain_mask)
        self._ensure_phi()

    def visualize_prediction(self, u_pred, thresholds, out_dir, prefix=''):
        """
        Visualize prediction (contour and density) with automatic 3D slicing.
        """
        if self.opts.get('skip_fig', False):
            return

        xdim = self.opts['dataset_opts']['xdim']
        
        if hasattr(self, 'slices_to_plot'):
            slices_to_plot = self.slices_to_plot
        else:
            slices_to_plot = []
            if xdim == 3:
                if self.z_slice is not None and self.z_slice != -1:
                     slices_to_plot.append((self.z_slice, 'default'))
                else:
                     slices_to_plot.append((u_pred.shape[2]//2, 'mid'))
            else:
                slices_to_plot.append((None, '2d'))

        # Unique slices handling
        unique_slices = {}
        for z, lbl in slices_to_plot:
            if z not in unique_slices:
                unique_slices[z] = lbl
            else:
                unique_slices[z] += f"_{lbl}"

        for z_slice, label in unique_slices.items():
            # Prepare 2D slices
            u_viz = u_pred
            bg_viz = self.pdata['Pwm']
            phi_viz = self.pdata['phi']
            wt_viz = self.pdata['mask_wt']
            tc_viz = self.pdata['mask_tc']
            seg_viz = self.pdata['seg']
            rec_viz = self.pdata.get('mask_rec', None)
            
            if xdim == 3:                
                u_viz = u_viz[:, :, z_slice]
                bg_viz = bg_viz[:, :, z_slice]
                phi_viz = phi_viz[:, :, z_slice]
                wt_viz = wt_viz[:, :, z_slice]
                tc_viz = tc_viz[:, :, z_slice]
                seg_viz = seg_viz[:, :, z_slice]
                if rec_viz is not None:
                    rec_viz = rec_viz[:, :, z_slice]
                        
            plot_contour_over_seg(
                u_viz,
                thresholds=thresholds,
                seg=seg_viz,
                bg=bg_viz,
                phi=phi_viz,
                fname=f'fig_{prefix}_z{z_slice}_contour.png',
                savedir=str(out_dir),
                title=f'{prefix} {label} (Contour)'
            )
            
            plot_density_over_seg(
                u_viz,
                u1=wt_viz,
                u2=tc_viz,
                bg=bg_viz,
                phi=phi_viz,
                fname=f'fig_{prefix}_z{z_slice}_density.png',
                savedir=str(out_dir),
                title=f'{prefix} {label} (Density)'
            )

    def evaluate_and_log(self, u_pred, prefix, th1=None, th2=None):
        if 'mask_rec' not in self.pdata or self.pdata['mask_rec'] is None:
            print("No recurrence segmentation, skipping evaluation")
            return

        mask_tc = self.pdata['mask_tc']
        brain_mask = self.pdata['brain_mask']
        mask_rec = self.pdata['mask_rec']
        seg = self.pdata['seg']
        
        metrics, plans = evaluate_personalized_plan(
            segmentation=mask_tc,
            brain_mask=brain_mask,
            predicted_density=u_pred,
            recurrence_mask=mask_rec,
            ctv_margin=15
        )
        
        # Evaluate Prediction (Dice)
        if th1 is not None and th2 is not None:
            dice_metrics = evaluate_prediction(u_pred, seg, th_wt=th1, th_tc=th2)
            metrics.update(dice_metrics)
        
        print(f"Evaluation {prefix}: {metrics}")
        if self.logger:
            # Log with prefix
            params_to_log = {}
            for k, v in metrics.items():
                if 'std_ctv' in k:
                    # Only log standard metrics once (no prefix)
                    if not self.std_metrics_logged:
                        params_to_log[k] = v
                else:
                    # Log model-specific metrics with prefix
                    params_to_log[f"{k}_{prefix}"] = v
            self.logger.log_params(params_to_log)
            
            # Mark standard metrics as logged
            if not self.std_metrics_logged:
                self.std_metrics_logged = True
            
        # Visualization
        ctv_threshold = metrics['threshold']
        
        thresholds = [th1, th2, ctv_threshold]
        
        out_dir = Path(self.logger.get_dir())
        self.visualize_prediction(u_pred, thresholds, out_dir, prefix=prefix)

        # Standard vs Personalized CTV Comparison
        standard_plan = plans['standard_plan']
        personal_plan = plans['personal_plan']
        
        # Prepare for plotting (slice if 3D)
        xdim = self.opts['dataset_opts']['xdim']
        
        if hasattr(self, 'slices_to_plot'):
            slices_to_plot = self.slices_to_plot
        else:
            slices_to_plot = [(None, '2d')] if xdim == 2 else [(u_pred.shape[2]//2, 'mid')]

        # Unique slices handling
        unique_slices = {}
        for z, lbl in slices_to_plot:
            if z not in unique_slices:
                unique_slices[z] = lbl
            else:
                unique_slices[z] += f"_{lbl}"

        for z_slice, label in unique_slices.items():
            std_viz = standard_plan
            pers_viz = personal_plan
            bg_viz = self.pdata['Pwm']
            seg_viz = self.pdata['seg']
            rec_viz = self.pdata.get('mask_rec', None)
            
            if xdim == 3:
                 std_viz = std_viz[:, :, z_slice]
                 pers_viz = pers_viz[:, :, z_slice]
                 bg_viz = bg_viz[:, :, z_slice]
                 seg_viz = seg_viz[:, :, z_slice]
                 if rec_viz is not None:
                     rec_viz = rec_viz[:, :, z_slice]
            
            
            if self.opts['skip_fig'] == False:
                plot_plan_comparison(
                    standard_plan=std_viz,
                    personal_plan=pers_viz,
                    seg_pre=seg_viz,
                    seg_post=rec_viz,
                    bg=bg_viz,
                    fname=f'fig_{prefix}_z{z_slice}_ctv.png',
                    savedir=str(out_dir),
                    title=f'{prefix} z={z_slice} {label} CTV Comparison'
                )


    def step_preprocess(self):
        print("Preprocessing")
        Pcsf, Pgm, Pwm, seg, rec, brain_mask = self._load_raw_nifti()
        self._setup_data(Pcsf, Pgm, Pwm, seg, rec, brain_mask)
        self._ensure_phi()

    def step_estimate_params(self):
        print("Estimating Char Params")
        xdim = self.opts['dataset_opts']['xdim']
        h_init = self.opts['pde_opts'].get('h_init', 0.5)
        r_init = self.opts['pde_opts'].get('r_init', 0.1)
        prior_th1 = self.opts['pde_opts']['prior_th1']
        prior_th2 = self.opts['pde_opts']['prior_th2']
        
        radius_method = self.opts['pde_opts'].get('radius_method', 'rmax')

        self.char_params = est_char_param(
            self.pdata['mask_wt'], self.pdata['mask_tc'], self.pdata['x0'],
            prior_th1, prior_th2, h_init, r_init, parallel=False, xdim=xdim,
            radius_method=radius_method
        )
        # log char params
        self.char_params['x0'] = self.pdata['x0']
        self.logger.log_params(self.char_params)
        self.char_params.update({'h_init': h_init, 'r_init': r_init, 'th1': prior_th1, 'th2': prior_th2})

    def compute_u0(self, x0, h_init, r_init):
        """Compute initial condition u0 based on estimated parameters."""
        x0 = x0.squeeze()
        center = x0 - 1.0
        xdim = self.opts['dataset_opts']['xdim']
        
        if xdim == 2:
            x = np.arange(self.pdata['Pwm'].shape[0])
            y = np.arange(self.pdata['Pwm'].shape[1])
            xx, yy = np.meshgrid(x, y, indexing='ij')
            dist_sq = (xx - center[0])**2 + (yy - center[1])**2
        else:
            x = np.arange(self.pdata['Pwm'].shape[0])
            y = np.arange(self.pdata['Pwm'].shape[1])
            z = np.arange(self.pdata['Pwm'].shape[2])
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            dist_sq = (xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2

        u0 = h_init * np.exp(-r_init * dist_sq)
        return u0

    def step_generate_dataset(self):
        
        Dc, rhoc, T, x0, L = self.char_params['Dc'], self.char_params['rhoc'], self.char_params['T'], self.char_params['x0'], self.char_params['L']
        
        # Initial Condition
        u0 = self.compute_u0(x0, self.char_params['h_init'], self.char_params['r_init'])
        
        factor = self.opts['pde_opts']['factor']
        Nt = self.opts['pde_opts']['Nt']
        threshold_rmax = 0.01

        # Simulation
        print("Solve FKPP with char param")
        u_all, t_all = solve_fisher_kpp_torch(
            self.pdata['Pwm'], self.pdata['Pgm'], dw=Dc, factor=factor, 
            phi=self.pdata['phi'], rho=rhoc, tfinal=T, u0=u0, Nt=Nt, device=self.device
        )

        # Evaluation and Visualization
        # Downsample and save for comparison
        phiu_all = u_all * self.pdata['phi'][..., np.newaxis]
        Nt = u_all.shape[-1]
        t_indices = np.linspace(0, Nt - 1, 5).astype(int)
        self.vdict['char_fdm'] = phiu_all[..., t_indices]

        phiu_final = phiu_all[..., -1]

        self.evaluate_and_log(
            phiu_final, 
            prefix='char',
            th1=self.char_params['th1'],
            th2=self.char_params['th2']
        )
        
        if self.opts['skip_dataset'] == True:
            print("Skipping dataset generation")
            return
    
        print("Generating Dataset for training")
        # generate training data
        char_param = get_char(Dc, rhoc, L)
        train_data = gen_samples(
            self.pdata['Pwm'], self.pdata['Pgm'], self.pdata['Pcsf'],
            self.pdata['mask_wt'], self.pdata['mask_tc'], self.pdata['phi'],
            factor, x0, threshold_rmax, char_param['DW'], char_param['RHO'], L, u_all, t_all
        )
        train_data.update({'h_init': self.char_params['h_init'], 'r_init': self.char_params['r_init']})
        train_data.update({'factor': 10.0, 'Nt': Nt})
        
        out_dir = Path(self.logger.get_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        pdataset_path = out_dir / "dat.mat"
        sio.savemat(pdataset_path, train_data)
        self.opts['pde_opts']['datafile'] = str(pdataset_path)
        print(f"Save dataset to {pdataset_path}")        

    def visualize_comparison(self, pred, ref, prefix=''):
        if self.opts.get('skip_fig', False):
            return 
        # pred and ref are (nx, ny, [nz], nt)
        # apply phi mask
        
        # assert same shape and same nt
        assert pred.shape == ref.shape, "Prediction and Reference shapes do not match"
        savedir = self.logger.get_dir()

        # if pred is 3D+time, slice at closed to x0[3]

        if pred.ndim == 4:
            x0 = self.pdata['x0']
            z0 = np.rint(x0[2] - 1).astype(int)
            z_indices = np.arange(pred.shape[2])
            z_slice = np.argmin(np.abs(z_indices - z0))
            pred = pred[:, :, z_slice, :]
            ref = ref[:, :, z_slice, :]
        
        levels = np.array([0.01, 0.1, 0.3, 0.6])
        for it in range(pred.shape[-1]):
            pred2d = pred[:, :, it]
            ref2d = ref[:, :, it] if ref is not None else None
            
            plot_grid_imshow_panels(
                pred2d,
                ref2d,
                fname=f'fig_{prefix}_density_t{it}.png',
                savedir=savedir,
                title=f'{prefix} Grid t={it}',
            )
            
            plot_grid_contour_overlay(
                pred2d,
                ref2d,
                levels=levels,
                fname=f'fig_{prefix}_contour_t{it}.png',
                savedir=savedir,
                title=f'{prefix} Contours t={it}',
            )


    def step_final_prediction(self):
        print("Final Prediction")
        self.load_geometry() # Ensure geometry is loaded
        
        # Load prediction from trainer's output
        # If running full, trainer is active. If running inv standalone, trainer is active.
        # We can rely on the file on disk which is safer for standalone
        pred_path = self.logger.gen_path("inv_dataset.mat")
        if not os.path.exists(pred_path):
             print(f"Prediction file {pred_path} not found.")
             return

        pred_data = sio.loadmat(pred_path)
        
        rD = pred_data['rD_pred']
        rRHO = pred_data['rRHO_pred']
        th1 = pred_data['th1_pred']
        th2 = pred_data['th2_pred']
        
        # Need DW and RHO from training dat.mat
        train_dat = self.pde.dataset
        dw_pred = rD * train_dat['DW'] * train_dat['L']**2
        rho_pred = rRHO *  train_dat['RHO']
        
        # Need u0. If not in memory, reconstruct
        if 'Pwm' not in self.pdata:
            self.load_geometry()
        
        u0 = self.compute_u0(train_dat['x0'], train_dat['h_init'], train_dat['r_init'])
        
         # FDM simulation with predicted parameters
        print("Solve FKPP with estimated param")
        u_all, _ = solve_fisher_kpp_torch(
            self.pdata['Pwm'], self.pdata['Pgm'], dw=dw_pred, factor=10.0,
            phi=self.pdata['phi'], rho=rho_pred, tfinal=1.0, 
            u0=u0, Nt=self.opts['pde_opts']['Nt'], device=self.device
        )
        
        Nt = u_all.shape[-1]
        phiu_all = u_all * self.pdata['phi'][..., np.newaxis]
        t_indices = np.linspace(0, Nt - 1, 5).astype(int)
        self.vdict['upred_fdm'] = phiu_all[..., t_indices]
        
        phiu_final = phiu_all[..., -1]
        
        final_path = Path(self.logger.get_dir()) / "final_prediction.mat"
        sio.savemat(final_path, {'upred_fdm': phiu_final})
        print(f"Saved final prediction to {final_path}")

        # Visualization        
        self.evaluate_and_log(
            phiu_final, 
            prefix='upredfdm',
            th1=th1,
            th2=th2
        )
        
        return phiu_final
        

    def configure_stage(self, stage):
        assert stage in ['init', 'inv'], f'stage {stage} not recognized'

        t_opts = self.opts['train_opts']
        p_opts = self.opts['pde_opts']
        
        suffix = f'_{stage}' # _init or _inv
        if stage == 'init':
            self.pde.setup_dataset(
                self.opts['pde_opts']['datafile'],
                whichdata = 'char',
                device=self.device
            )
        else:
            self.pde.setup_dataset(
                self.opts['pde_opts']['datafile'],
                whichdata = p_opts.get('whichdata', 'pat'),
                device=self.device
            )
        
        # the network require the dataset to be setup first
        if not hasattr(self, 'net'):
            self.net = self.pde.setup_network(**self.opts['nn_opts'])

        # set trainable parameters
        for name in self.opts['pde_opts']['trainable_param']:
            if name in self.net.all_params_dict:
                self.net.all_params_dict[name].requires_grad = True if stage == 'inv' else False
        
        if self.trainer is None:
            self.trainer = Trainer(self.opts['train_opts'], self.net, self.pde, self.device, self.logger)
        
        # restore if needed from init
        # always restore from init stage
        if self.restore_artifacts:
            self.trainer.restore(self.restore_artifacts['artifacts_dir'], prefix='init')

        # configure trainer according to stage
        if stage == 'init':
            self.trainer.set_loss_weight(
                loss=self.opts['train_opts']['loss_init'],
                loss_test=self.opts['train_opts']['loss_test'],
                loss_weight=self.opts['train_opts']['weights_init']
            )
            self.trainer.set_lr(self.opts['train_opts']['lr_init'])
            self.trainer.stage = 'init'
            self.trainer.set_steps(0, self.opts['train_opts']['iter_init'])
        else:
            self.trainer.set_loss_weight(
                loss=self.opts['train_opts']['loss_inv'],
                loss_test=self.opts['train_opts']['loss_test'],
                loss_weight=self.opts['train_opts']['weights_inv']
            )
            self.trainer.set_lr(self.opts['train_opts']['lr_inv'])
            self.trainer.stage = 'inv'
            
            start_step = 0
            if self.opts['traintype'] == 'full':
                start_step = self.opts['train_opts']['iter_init']
            
            self.trainer.set_steps(start_step, self.opts['train_opts']['iter_inv'])

            

    def run_pre(self):
        self.step_preprocess()
        self.step_estimate_params()
        self.step_generate_dataset()
        

    def run_init(self):
        # Ensure data is set
        if not self.opts['pde_opts']['datafile']:
            #  get the file from restore
            # restore artificts should contain the path to dat.mat
            if 'dat.mat' in self.restore_artifacts:
                self.opts['pde_opts']['datafile'] = self.restore_artifacts['dat.mat']
            else:
                print('datafile not find, run preprocess')
                self.run_pre()

        
        if self.opts['nn_opts'].get('separable', False):
            self.pde = sepGBMproblem(self.opts['dataset_opts']['xdim'], self.opts['pde_opts'])
        else:
            self.pde = GBMproblem(self.opts['dataset_opts']['xdim'], self.opts['pde_opts'])
    
        self.configure_stage('init')
        
        self.trainer.train()
        self.trainer.save(prefix='init')
        if not self.opts.get('skip_fig', False):
            self.pde.visualize(savedir=self.logger.get_dir(), prefix='init')
            self.pde.dataset.visualize_sampling(self.logger.get_dir())
        
        # Compare NN vs Char
        self.load_geometry()
        upred = self.pde.dataset['upred']
        Nt = upred.shape[-1]
        t_indices = np.linspace(0, Nt - 1, 5).astype(int)
        upred = upred[..., t_indices]
        full_shape = self.pdata['Pwm'].shape
        upred_full = self.pde.dataset.restore_to_full_grid(full_shape, upred)
        self.vdict['char_nn'] = upred_full * self.pdata['phi'][..., np.newaxis]
        
        
        self.visualize_comparison(self.vdict['char_nn'], self.vdict['char_fdm'], prefix='init_compare')

        
    def run_inv(self):
        
        if not self.opts['pde_opts']['datafile']:
            #  get the file from restore
            # restore artificts should contain the path to dat.mat
            if 'dat.mat' in self.restore_artifacts:
                self.opts['pde_opts']['datafile'] = self.restore_artifacts['dat.mat']
            else:
                raise ValueError("Datafile not foun")
        print(f"Using datafile: {self.opts['pde_opts']['datafile']}")

        # If running standalone, setup problem/logger
        if not hasattr(self, 'pde'):
            if self.opts['nn_opts'].get('separable', False):
                self.pde = sepGBMproblem(self.opts['dataset_opts']['xdim'], self.opts['pde_opts'])
            else:
                self.pde = GBMproblem(self.opts['dataset_opts']['xdim'], self.opts['pde_opts'])
        
        self.configure_stage('inv')
        # setup logger
        self.trainer.train()
        self.trainer.save(prefix='inv')
        
        if not self.opts.get('skip_fig', False):
            self.pde.visualize(savedir=self.logger.get_dir(), prefix='inv')

        if self.pde.whichdata != 'pat':
            print("Not patient data, skipping final prediction.")
            return

        # Visualize NN Solution
        self.load_geometry()
        
        upred = self.pde.dataset['upred']
        Nt = upred.shape[-1]
        t_indices = np.linspace(0, Nt - 1, 5).astype(int)
        full_shape = self.pdata['Pwm'].shape
        upred = upred[..., t_indices]
        upred_full = self.pde.dataset.restore_to_full_grid(full_shape, upred)
        phiupred_full = upred_full * self.pdata['phi'][..., np.newaxis]
        self.vdict['upred_nn'] = phiupred_full
        phiu_final = self.vdict['upred_nn'][..., -1]
        th1_pred = self.pde.dataset['th1_pred']
        th2_pred = self.pde.dataset['th2_pred']
        
        self.evaluate_and_log(
            phiu_final, 
            prefix='uprednn',
            th1=th1_pred,
            th2=th2_pred
        )

        self.step_final_prediction()

        self.visualize_comparison(pred=self.vdict['upred_nn'], ref=self.vdict['upred_fdm'], prefix='final_compare')

    def run_full(self):
        self.run_pre()
        self.run_init()
        self.run_inv()

    def run(self):
        traintype = self.opts['traintype']
        self.restore_run()
        
        if traintype == 'pre':
            self.run_pre()
        elif traintype == 'init':
            self.run_init()
        elif traintype == 'inv':
            self.run_inv()
        elif traintype == 'full':
            self.run_full()
        else:
            raise ValueError(f"traintype {traintype} not recognized")