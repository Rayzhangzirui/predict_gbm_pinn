#!/usr/bin/env python
import sys
import json
import copy
from BaseOption import BaseOption

# pde_opts that 

# for BilO and PINN
default_opts = {
    'logger_opts': {'use_mlflow':True,
                    'use_stdout':False,
                    'experiment_name':'dev',
                    'run_name':'test',
                    'use_tmp': False,
                    'output_dir': ''},
    'dataset_opts': {
        'input_dir': '',
        'segpre_file': '',
        'gm_file': '',
        'wm_file': '',
        'csf_file': '',
        'segrec_file': '',
        'brain_mask_file': '',
        'z_slice': -1,
        'xdim': 3,
    },
    'restore': '',
    'traintype': 'full',
    'skip_dataset': False,
    'skip_fig': False, 
    'flags': '', 
    'device': 'cuda',
    'seed': 0,
    'pde_opts': {
        'factor': 10.0, # factor = D_wm / D_gm
        'datafile': '',
        'h_init': 0.5, # initial guess
        'r_init': 0.01, # initial guess
        'trainable_param': 'rD,rRHO,th1,th2', # list of trainable parameters, e.g. 'D,rho'
        'init_param': '', # nn initial parameter as string, e.g. 'D,1.0'
        'whichdata':'pat', # which data to use for data loss
        'force_bc':True, # force boundary condition
        'th1_range':[0.0,1.0], # range of theta1
        'th2_range':[0.0,1.0], # range of theta2
        'rD_range':[0.0,2.0], # range of D
        'rRHO_range':[0.0,2.0], # range of RHO
        'texp_weight': 10.0, # exp(- w t) weight for time
        'rpow_weight': 0.0, # 1/(r^p) weight for radial coordinate
        'causal_weight': False, # use causal weight for time
        'ksigmoid': 50, # larger k for sharper heaviside function
        'dat_batch_size': 100000, # batch size for data loss
        'res_batch_size': 200000, # batch size for residual loss
        'm_times': 100, # number of time snapshots for pretraining data
        'prior_th1': 0.35, # prior for th1
        'prior_th2': 0.7, # prior for th2
        'radius_method': 'rvol', # 'rmax' or 'rvol'
        'Nt': 101, # number of time points for residual loss in SPINN
        'use_fdm': False, # use fdm to compute residual
    },
    'nn_opts': {
        'depth': 4,
        'width': 128,
        'output_rank': 64,
        'fourier':True,
        'sigma': 1.0,
        'arch': 'mmlp',
        'separable': True,
    },
    'train_opts': {
        'print_every': 10,
        
        # Stage-specific configurations
        'lr_init': 1e-3,
        'iter_init': 4000,
        'loss_init': 'res,uchar_res',
        'weights_init': 'res,1.0,uchar_res,10.0',
        
        'lr_inv': 1e-3,
        'iter_inv': 1000,
        'loss_inv': 'res,seg1,seg2,rD_reg,rRHO_reg,th1_reg,th2_reg',
        'weights_inv': 'res,1.0,seg1,1e-5,seg2,1e-5,rD_reg,1e-3,rRHO_reg,1e-3,th1_reg,1e-3,th2_reg,1e-3',
        'reset_optim': True, # reset optimizer state

        'loss_test': '', # loss for testing
        # string for optimizer name
        'optim':'Adam',
        # string for optimizer options
        'optim_opts':'amsgrad,True',
        # scheduler options
        'sch':'ExponentialLR',
        'schopt':'gamma,1.0',
        'acc_iter': 1, # accumulate n-iteration of gradient
        
    },
}

class Options(BaseOption):
    def __init__(self):
        self.opts = copy.deepcopy(default_opts)
    

    def parse_args(self, *args):
        # first parse args and update dictionary
        # then process dependent options
        self.parse_nest_args(*args)
    
    def process_options(self):
        self.process_traintype()
        self.process_flags()
    
    def process_flags(self):

        if self.opts['flags'] != '':
            # remove white space and split
            self.opts['flags'] = self.opts['flags'].replace(' ','').split(',')

            assert all([flag in ['small','local'] for flag in self.opts['flags']]), 'invalid flag'
        else:
            self.opts['flags'] = []

        if 'small' in self.opts['flags']:
            # use small network for testing
            self.opts['nn_opts']['depth'] = 4
            self.opts['nn_opts']['width'] = 2
            self.opts['train_opts']['iter_init'] = 10
            self.opts['train_opts']['iter_inv'] = 10
            self.opts['train_opts']['print_every'] = 1
            
        if 'local' in self.opts['flags']:
            # use local logger
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            self.opts['logger_opts']['use_csv'] = False
        

    def process_traintype(self):
        # process options related to training
        # split traintype
        traintype = self.opts['traintype']

        assert traintype in ['init','inv','pre','full'], f'Invalid traintype {traintype}'
        
        self.opts['pde_opts']['trainable_param'] = [] if self.opts['pde_opts']['trainable_param'] == '' else self.opts['pde_opts']['trainable_param'].split(',')
        
        self.opts['train_opts']['loss_test'] = self.opts['train_opts']['loss_test'].split(',') if self.opts['train_opts']['loss_test'] != '' else []

        
        self.opts['train_opts']['schopt'] = self.convert_to_dict(self.opts['train_opts']['schopt'])
        self.opts['pde_opts']['init_param'] = self.convert_to_dict(self.opts['pde_opts']['init_param'])
        self.opts['train_opts']['optim_opts'] = self.convert_to_dict(self.opts['train_opts']['optim_opts'])
        # convert weight to dict
        for suffix in ['_init', '_inv']:
            w_key = f'weights{suffix}'
            l_key = f'loss{suffix}'
            
            self.opts['train_opts'][w_key] = self.convert_to_dict(self.opts['train_opts'][w_key])
            self.opts['train_opts'][l_key] = self.opts['train_opts'][l_key].split(',')
            

        # convert scheduler option to dict

    def print(self):
        print(json.dumps(self.opts, indent=2, sort_keys=True))



if __name__ == "__main__":

    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))