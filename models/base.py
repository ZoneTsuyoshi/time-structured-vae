# Author: Tsuyoshi Ishizone <tsuyoshi.ishizone@gmail.com>
# Contributors: Kosuke Matsunaga,
#               Soichiro Fuchigami,
#               Kazuyuki Nakamura
# Copyright (c) 2022, Meiji University and the Authors
# All rights reserved.

import os
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from msmbuilder.base import BaseEstimator

from .utils import construct_dense_network


class BaseModel(nn.Module, BaseEstimator):
    def __init__(self, input_dim:int, lag_time:int=1, n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True, cuda:Union[bool, int]=True, save_dir=None,
                 encoder_stochastic:bool=False, decoder_stochastic:bool=False):
        super(BaseModel, self).__init__()
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lag_time = lag_time
        
        self.encoder = construct_dense_network(input_dim, (1 + encoder_stochastic) * latent_dim, hidden_dim, n_layers, activation, dropout_ratio)
        self.decoder = construct_dense_network(latent_dim, (1 + decoder_stochastic) * input_dim, hidden_dim, n_layers, activation, dropout_ratio)
        
        if cuda:
            cuda = 0 # set device 0
        if type(cuda)==int:
            self.device = torch.device("cuda", cuda)
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        
        self.save = save_dir is not None
        if self.save:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.path.exists(save_dir)

        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=learning_rate)
        self.loss_fn = getattr(nn, loss)
        
        self.is_fitted = False
        
        
        
    def _create_dataset(self, data:np.array):
        """
        create dataset from data
            data:[n_timesteps, n_dim]
        """
        [data[j::self.lag_time] for j in range(self.lag_time)]
        t0 = np.concatenate([d[j::self.lag_time][:-1] for d in data
                             for j in range(slide)], axis=0)
        t1 = np.concatenate([d[j::self.lag_time][1:] for d in data
                             for j in range(slide)], axis=0)
        
        
    def fit(self, X):
        train_data = self._create_dataset(X)

        for i in range(self.n_epochs):
            if self.verbose:
                print('Epoch: %s' % i)
            self._train(train_data)

        self.is_fitted = True


    def _reparameterized_sample(self, mean, logstd):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps.mul(logstd.exp()).add_(mean)