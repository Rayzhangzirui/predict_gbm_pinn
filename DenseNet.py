#!/usr/bin/env python
import sys
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from util import *

from Options import *
    

class RandomFourierFeatureLayer(nn.Module):
    def __init__(self, input_dim, output_dim, sigma):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(input_dim, output_dim)
        # Initialize weights with normal distribution (standard for RFF)
        nn.init.normal_(self.linear.weight, std=sigma)
        # Initialize bias with uniform distribution [0, 2*pi]
        nn.init.uniform_(self.linear.bias, 0, 2 * torch.pi)
        # Freeze parameters
        for param in self.linear.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return torch.sin(self.linear(x))

# simple mlp
class MLP(nn.Module):
    def __init__(self, 
                  input_size:int,
                  width:int, 
                  depth:int, 
                  output_dim:int, 
                  activation:str='tanh',
                  resnet:bool=False,
                  rff:bool=False,
                  sigma:float=1.0,
                  output_transform=lambda x,u: u):
        super(MLP, self).__init__()
        
        # tanh activation function
        string_to_activation = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'softplus': torch.nn.functional.softplus,
        }
            
        self.activation = string_to_activation[activation]
        # if rff is True, then use random fourier feature
        self.rff = rff
        self.resnet = resnet
        self.sigma = sigma

        # input layer
        if self.rff:
            self.input_layer = RandomFourierFeatureLayer(input_size, width, sigma)
        else:
            self.input_layer = nn.Linear(input_size, width)

        # hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )
        # output layer
        self.output_layer = nn.Linear(width, output_dim)
        self.output_transform = output_transform

    def forward(self, x):
        
        if self.rff:
            # first layer is random fourier feature
            u = self.input_layer(x)
        else:
            # normal input layer
            u = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            if self.resnet:
                u = u + self.activation(layer(u))
            else:
                u = self.activation(layer(u))
    
        u = self.output_layer(u)
        u = self.output_transform(x, u)
        return u


class PirateBlock(nn.Module):
    def __init__(self, width, activation):
        super(PirateBlock, self).__init__()
        self.activation = activation
        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, U, V):
        f = self.activation(self.layer1(x))
        z1 = f * U + (1 - f) * V
        
        g = self.activation(self.layer2(z1))
        z2 = g * U + (1 - g) * V
        
        h = self.activation(self.layer3(z2))
        
        return self.alpha * h + (1 - self.alpha) * x

class PirateNet(nn.Module):
    def __init__(self, 
                  input_size:int,
                  width:int, 
                  depth:int, 
                  output_dim:int, 
                  activation:str='tanh',
                  sigma:float=1.0,
                  output_transform=lambda x,u: u,
                  **kwargs):
        super(PirateNet, self).__init__()
        
        string_to_activation = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'softplus': torch.nn.functional.softplus,
        }
            
        self.activation = string_to_activation[activation]
        self.sigma = sigma
        self.output_transform = output_transform

        # Random Fourier Feature layer (fixed)
        self.rff_layer = RandomFourierFeatureLayer(input_size, width, sigma)
        
        # Encoding maps U and V
        self.U_layer = nn.Linear(width, width)
        self.V_layer = nn.Linear(width, width)
        
        # Residual blocks
        self.blocks = nn.ModuleList(
            [PirateBlock(width, self.activation) for _ in range(depth)]
        )
            
        # Output layer
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x):
        # RFF embedding
        phi = self.rff_layer(x)
        
        # Encoding maps
        U = self.activation(self.U_layer(phi))
        V = self.activation(self.V_layer(phi))
        
        curr_x = phi
        
        for block in self.blocks:
            curr_x = block(curr_x, U, V)
            
        u = self.output_layer(curr_x)
        u = self.output_transform(x, u)
        return u

class ModifiedMLP(nn.Module):
    def __init__(self, 
                  input_size:int,
                  width:int, 
                  depth:int, 
                  output_dim:int, 
                  activation:str='tanh',
                  rff:bool=False,
                  sigma:float=1.0,
                  output_transform=lambda x,u: u):
        super(ModifiedMLP, self).__init__()
        
        string_to_activation = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'softplus': torch.nn.functional.softplus,
        }
            
        self.activation = string_to_activation[activation]
        self.rff = rff
        self.sigma = sigma
        self.output_transform = output_transform

        # input layer
        if self.rff:
            self.input_layer = RandomFourierFeatureLayer(input_size, width, sigma)
        else:
            self.input_layer = nn.Linear(input_size, width)

        # Encoding maps U and V
        self.U_layer = nn.Linear(width, width)
        self.V_layer = nn.Linear(width, width)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )
            
        # Output layer
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x):
        if self.rff:
            H = self.input_layer(x)
        else:
            H = self.activation(self.input_layer(x))
        
        # Encoding maps
        U = self.activation(self.U_layer(H))
        V = self.activation(self.V_layer(H))
        
        for layer in self.hidden_layers:
            Z = layer(H)
            A = self.activation(Z)
            H = A * U + (1 - A) * V
            
        u = self.output_layer(H)
        u = self.output_transform(x, u)
        return u

class DenseNet(nn.Module):
    def __init__(self, depth=4, width=8, input_dim=1, output_dim=1, 
                lambda_transform=lambda x, u: u,
                all_params_dict: Dict = {},
                trainable_param:List[str] = [],
                # architecture control
                arch: str = 'mlp',
                 # residual connection
                resnet: bool = False, 
                # random fourier feature
                fourier: bool = False,
                # scale for fourier feature
                sigma: float = 1.0,
                act: str = 'tanh',
                modifiedmlp: bool = False,
                **kwargs):
        super().__init__()
        
        self.trainable_param = trainable_param

        tmp = {}
        for k, v in all_params_dict.items():
            tmp[k] = nn.Parameter(torch.tensor(v, dtype=torch.float32))
        self.all_params_dict = nn.ParameterDict(tmp)

        for name, param in self.all_params_dict.items():
            if name not in self.trainable_param:
                param.requires_grad = False

        # Backward compatibility
        if modifiedmlp:
            arch = 'pirate'

        if arch == 'pirate':
            self.net = PirateNet(input_dim, width, depth, output_dim, 
                                 activation=act, sigma=sigma, 
                                 output_transform=lambda_transform, **kwargs)
        elif arch == 'mmlp':
            self.net = ModifiedMLP(input_dim, width, depth, output_dim,
                                   activation=act, rff=fourier, sigma=sigma,
                                   output_transform=lambda_transform)
        elif arch == 'mlp':
            self.net = MLP(input_dim, width, depth, output_dim, 
                           activation=act, resnet=resnet, rff=fourier, 
                           sigma=sigma, output_transform=lambda_transform)
        else:
            raise ValueError(f"Architecture {arch} not supported")

    def forward(self, x):
        return self.net(x)

class SeparableDenseNet(nn.Module):
    def __init__(self, 
                 depth=4, 
                 width=8, 
                 input_dim=1, 
                 output_dim=1, 
                 output_rank=16,
                #  list of transform for each coordinate
                lambda_transform: List[Callable] = None,
                all_params_dict = None,
                trainable_param:List[str] = [],
                # architecture control
                arch: str = 'mlp',
                # residual connection
                resnet: bool = False, 
                # random fourier feature
                fourier: bool = False,
                # scale for fourier feature
                sigma: float = 1.0,
                act: str = 'tanh',
                modifiedmlp: bool = False,
                **kwargs):
        super().__init__()
        
        # for separable net
        self.depth = depth
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_rank = output_rank

        if lambda_transform is None:
            self.lambda_transform = [lambda x, u: u] * input_dim
        else:
            self.lambda_transform = lambda_transform

        # random fourier feature, smaller sigma, smoother features.
        self.resnet = resnet
        self.fourier = fourier
        self.sigma = sigma
        # activation function
        self.act = act
        
        self.trainable_param = trainable_param

        tmp = {}
        if all_params_dict is not None:
            for k, v in all_params_dict.items():
                tmp[k] = nn.Parameter(torch.tensor(v, dtype=torch.float32))
        self.all_params_dict = nn.ParameterDict(tmp)

        for name, param in self.all_params_dict.items():
            if name not in self.trainable_param:
                param.requires_grad = False
        
        self.arch = arch
        self.setup_basic_architecture()

    def setup_basic_architecture(self):
        
        input_dim = self.input_dim
        width = self.width
        depth = self.depth
        output_rank = self.output_rank
        output_transform = self.lambda_transform
        # set einsum depending on input dim
        # Generate einsum string for arbitrary dimensions
        # Pattern: 'ai,bi,ci,...->abc...' for input_dim dimensions
        tensor_dim = input_dim # total tensor dimensions including PDE params
        input_indices = ','.join([f'{chr(97+i)}i' for i in range(tensor_dim)])  # 'ai,bi,ci,...'
        output_indices = ''.join([chr(97+i) for i in range(tensor_dim)])        # 'abc...'
        self.einsum = f'{input_indices}->{output_indices}'

        # create tuple of dimensions, for 2d, (0,1) is coord, (2,3, 4...) is pde param
        self.coord_dim_tuple = tuple(range(input_dim))

        # list of MLP for each coord
        if self.arch == 'pirate':
            self.mlps = nn.ModuleList(
                [PirateNet(1, width, depth, output_dim=output_rank, activation=self.act,
                     sigma=self.sigma,
                     output_transform=output_transform[i]) for i in range(input_dim)]
            )
        elif self.arch == 'mmlp':
            self.mlps = nn.ModuleList(
                [ModifiedMLP(1, width, depth, output_dim=output_rank, activation=self.act,
                     rff=self.fourier, sigma=self.sigma,
                     output_transform=output_transform[i]) for i in range(input_dim)]
            )
        elif self.arch == 'mlp':
            self.mlps = nn.ModuleList(
                [MLP(1, width, depth, output_dim=output_rank, activation=self.act,
                     resnet=self.resnet, rff=self.fourier, sigma=self.sigma,
                     output_transform=output_transform[i]) for i in range(input_dim)]
            )
        else:
            raise ValueError(f"Architecture {self.arch} not supported")
        
    def forward(self, coord_list:list):
        list_output = []
        for i, coord in enumerate(coord_list):
            output = self.mlps[i](coord)
            list_output.append(output)
        
        u = torch.einsum(self.einsum, *list_output)

        if self.input_dim == 1:
            u = u.unsqueeze(-1)  # keep N-by-1 shape for consistency
            
        return u