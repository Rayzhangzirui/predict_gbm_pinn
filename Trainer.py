#!/usr/bin/env python

# for training the network
# need: options, network, pde, dataset, lossCollection
import os
import time
import torch
import torch.optim as optim
from Logger import Logger
from util import get_mem_stats, set_device, set_seed, print_dict, flatten

import torch.profiler

import copy

optimizer_dictionary = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}

class Trainer():
    def __init__(self, opts, net, pde, device, logger:Logger):
        self.opts = opts
        self.logger = logger
        self.net = net
        self.pde = pde
        self.device = device
        self.info = {}
        self.stage = ''

        

        optimizer = optimizer_dictionary[self.opts['optim']]
        self.optimizer = optimizer(self.net.parameters(), lr = 1e-3, **self.opts['optim_opts'])
        
        scheduler_net = getattr(optim.lr_scheduler, self.opts['sch'])
        self.scheduler = scheduler_net(self.optimizer, **self.opts['schopt'])
        
        self.start_step = 0
        self.num_iter = 0
       
    
    def set_loss_weight(self, loss, loss_test, loss_weight:dict):
        # configure loss weight dictionary
        self.loss = loss
        self.loss_test = loss_test
        self.loss_weight = loss_weight
        # if loss not in loss_weight, set weight to 1.0
        for key in self.loss + self.loss_test:
            if key not in self.loss_weight:
                self.loss_weight[key] = 1.0

         # dict during training
        self.weighted_loss_comp = {} # component of each loss, weighted
        self.unweighted_loss_comp = {} # component of each loss, unweighted
        self.wtotal = None # total loss for backprop
    
    def set_lr(self, lr):
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def set_steps(self, start_step, num_iter):
        self.start_step = start_step
        self.num_iter = num_iter
    
    def get_wloss_sum_comp(self, list_of_loss: list, yes_grad: bool):
        # for bilevel optimization
        # list_of_loss can be empty list
        weighted_sum = 0.0 if list_of_loss else None
        weighted_loss_comp = {}
        unweighted_loss_comp = {}
        with torch.set_grad_enabled(yes_grad):
            for key in list_of_loss:
                unweighted_loss_comp[key] = self.pde.loss_dict[key](self.net)
                weighted_loss_comp[key] = self.loss_weight[key] * unweighted_loss_comp[key]
                weighted_sum += weighted_loss_comp[key]
        
        return weighted_sum, weighted_loss_comp, unweighted_loss_comp


    def train_loop(self):
        '''
        vanilla training of network, update all parameters simultaneously
        '''
        for step in range(self.start_step, self.start_step + self.num_iter):

            self.optimizer.zero_grad()
            
            wtotal, wloss_comp, uwloss_comp = self.get_wloss_sum_comp(self.loss, True)

            # print statistics at interval or at stop
            self.log_stat(step, **uwloss_comp, total=wtotal)
            self.validate(step)
            
            # take gradient of residual loss w.r.t all parameters
            # do not use setgrad here. if vanilla init, pde_param requires grad = False
            wtotal.backward()
            self.optimizer.step()
            self.scheduler.step()
            # next cycle
            self.pde.dataset.next_batch()
            

    def log_trainable_params(self, step):
        '''
        log trainable parameters
        '''
        for key in self.net.trainable_param:
            self.logger.log_metrics({key:self.net.all_params_dict[key].item()}, step=step)
        
    def log_stat(self, step, **kwargs):
        # kwargs is key value pair
        if step % self.opts['print_every'] == 0:
            for key, val in kwargs.items():
                if val is not None:
                    self.logger.log_metrics({key:val}, step=step)
        
    @torch.no_grad
    def validate(self, epoch):
        if epoch % self.opts['print_every'] == 0:
            val = self.pde.validate(self.net)
            self.logger.log_metrics(val, step=epoch)
            
            # log testing loss
            wtotal, wloss_comp, uwloss_comp = self.get_wloss_sum_comp(self.loss_test, False)
            self.logger.log_metrics(uwloss_comp, step=epoch)
    
    def set_grad(self, params, loss):
        '''
        set gradient of loss w.r.t params
        '''
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for param, grad in zip(params, grads):
            param.grad = grad
    
    def acc_grad(self, params, loss):
        '''
        accumulate gradient of loss w.r.t params
        '''
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for param, grad in zip(params, grads):
            # skip if grad is None
            if grad is None:
                continue
            # if param.grad is None, then initial grad, otherwise accumulate
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    def train(self):
        # move to device

        self.info[f'{self.stage}_num_params'] = sum(p.numel() for p in self.net.parameters())
        self.info[f'{self.stage}_num_train_params'] = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        self.net.to(self.device)

        start = time.time()

        # reset memeory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        
        # for running
        try:    
            self.train_loop()
        except KeyboardInterrupt:
            print('Interrupted by user')
            self.info.update({f'{self.stage}_error':'interrupted by user'})
            self.logger.set_tags('status', 'FAILED')
        except Exception as e:
            self.info.update({f'{self.stage}_error':str(e)})
            # mlflow set state to failed
            print(f'Error: {str(e)}')
            self.logger.set_tags('status', 'FAILED')
            raise e
            

        # for profiling
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir='./log/tmp')

        # with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA],
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=False,
        #         ) as prof:
        #     self.train_loop()
        # # prof.export_chrome_trace("profiler_trace.json")
        # # writer.close()
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        
        # log memory usage
        mem =  get_mem_stats(self.device)
        self.info.update({f'{self.stage}_' + k: v for k, v in mem.items()})

        # log info
        self.logger.log_params(flatten(self.info))

    def save_optimizer(self, prefix=''):
        # save optimizer
        fpath = self.logger.gen_path(f"{prefix}_optimizer.pth")
        torch.save(self.optimizer.state_dict(), fpath)
        print(f'save optimizer to {fpath}')
    

    def load_optim(self, fpath):
        if not os.path.exists(fpath):
            print(f'optimizer file {fpath} not found, use default optimizer')
            return
        
        print(f'restore optimizer from {fpath}')
        state_dict = torch.load(fpath, map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(state_dict)

        # if GPU changed, need to move optimizer state to device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    
    def restore_optimizer(self, dirname, prefix=''):
        # restore optimizer, need dirname
        if self.opts['reset_optim']:
            print('do not restore optimizer, reset optimizer to default')
            return

        fname = f"{prefix}_optimizer.pth"
        fpath = os.path.join(dirname, fname)
        self.load_optim(fpath)
        

    def save_net(self, prefix=''):
        # save network
        net_path = self.logger.gen_path(f"{prefix}_net.pth")
        torch.save(self.net.state_dict(), net_path)
        print(f'save model to {net_path}')

    def restore_net(self, net_path):
        
        state_dict = torch.load(net_path,map_location=self.device, weights_only=True)

        # set strict to False, to allow loading partial model for loRA
        self.net.load_state_dict(state_dict, strict=False)

        print(f'restore model from {net_path}')

    def save_dataset(self, prefix=''):
        ''' make prediction and save dataset'''
        self.pde.make_prediction(self.net)

        # if self.pde has attribute names2save, save only those variables, mainly for GBMproblem
        dataset_path = self.logger.gen_path(f"{prefix}_dataset.mat")
        names2save = []
        if hasattr(self.pde, 'names2save'):
            names2save = self.pde.names2save
        self.pde.dataset.save(dataset_path, names2save=names2save)
        
    def restore(self, dirname, prefix=''):
        # restore optimizer and network
        self.restore_optimizer(dirname, prefix=prefix)

        fnet = os.path.join(dirname, f'{prefix}_net.pth')
        self.restore_net(fnet)


    def save(self, prefix=''):
        '''saving dir from logger'''
        # save prediction
        self.save_dataset(prefix=prefix)

        # if num_iter is 0, do not save optimizer and net
        if self.num_iter > 0:
            self.save_optimizer(prefix=prefix)
            self.save_net(prefix=prefix)

    


