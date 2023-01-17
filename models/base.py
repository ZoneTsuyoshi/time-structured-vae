# Author: Tsuyoshi Ishizone <tsuyoshi.ishizone@gmail.com>
# Contributors: Kosuke Matsunaga,
#               Soichiro Fuchigami,
#               Kazuyuki Nakamura
# Copyright (c) 2022, Meiji University and the Authors
# All rights reserved.

import os
from typing import Union

import numpy as np
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from msmbuilder.base import BaseEstimator

from .utils import construct_dense_network


class BaseModel(nn.Module, BaseEstimator):
    def __init__(self, input_dim:int, lagtime:int=1, n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True, print_every:int=10, cuda:Union[bool, int]=True, save_dir=None,
                 encoder_stochastic:bool=False, decoder_stochastic:bool=False):
        super(BaseModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lagtime = lagtime
        self.sliding_window = sliding_window
        self.verbose = verbose
        self.print_every = print_every
        
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
        self.loss_fn = getattr(nn, loss)()
        self.loss_name_list = ["main", loss]
        
        self.is_fitted = False
        
        
        
    def _create_dataset(self, data:np.ndarray, valid_ratio:float=0.) -> DataLoader:
        """
        create dataset from data
        
        Args:
            data:[n_timesteps, n_dim]
        """
        slide = self.lagtime if self.sliding_window else 1
        t0 = np.concatenate([data[j::self.lagtime][:-1] for j in range(slide)], axis=0) #(slide,T//lag,dx)
        t1 = np.concatenate([data[j::self.lagtime][1:] for j in range(slide)], axis=0) #(slide,T//lag,dx)
        t = np.concatenate((t0.reshape(-1, self.input_dim, 1), t1.reshape(-1, self.input_dim, 1)), axis=-1) #(ns,dx,2)
        
        if valid_ratio > 0:
            train_t, valid_t = model_selection.train_test_split(t, valid_ratio)
            return DataLoader(torch.from_numpy(train_t), batch_size=self.batch_size, shuffle=True), DataLoader(torch.from_numpy(valid_t), batch_size=self.batch_size, shuffle=False)
        else:
            return DataLoader(torch.from_numpy(t), batch_size=self.batch_size, shuffle=True), None
    
    
    # def compute_loss(self, X:np.array):
    #     pass
    
    
    def _train(self, data:DataLoader) -> np.ndarray:
        self.train()
        epoch_loss = np.zeros(len(self.loss_name_list))

        for t, X in enumerate(data):
            self.optimizer.zero_grad()
            loss = self._compute_loss(X)
            
            for i in range(len(loss)):
                epoch_loss[i] += loss[i].item() / len(data)
                
            loss[0].backward()
            self.optimizer.step()
        return epoch_loss
    
    
    def _valid(self, data:DataLoader) -> np.ndarray:
        self.eval()
        epoch_loss = np.zeros(len(self.loss_name_list))

        for t, X in enumerate(data):
            loss, _ = self(X)
            
            for i in range(len(loss)):
                epoch_loss[i] += loss[i].item() / len(data)
        return epoch_loss
        
        
    def fit(self, X:np.ndarray, valid_ratio:float=0.):
        train_data, valid_data = self._create_dataset(X, valid_ratio)

        for i in range(self.n_epochs):
            train_loss = self._train(train_data)
            if valid_data is not None:
                valid_loss = self._valid(valid_data)
            
            if (i + 1) % self.print_every == 0 and self.verbose:
                print_content = f"Train Epoch: [{i+1}/{self.n_epochs}]"
                for j, loss_name in enumerate(self.loss_name_list):
                    print_content += " {}:{:.4f}".format(loss_name, train_loss[j])
                    
                if valid_data is not None:
                    print_content += "\n==> Valid: "
                    for j, loss_name in enumerate(self.loss_name_list):
                        print_content += " {}:{:.4f}".format(loss_name, valid_loss[j])
                print(print_content)

        self.is_fitted = True
        
        
    def transform(self, X:np.ndarray) -> np.ndarray:
        self.eval()
        X = torch.from_numpy(X).to(self.device)
        Z = self.encoder(X)[:, :self.latent_dim]
        return Z.detach().cpu().numpy()
        
        
    def fit_transform(self, X:np.ndarray, valid_ratio:float=0.) -> np.ndarray:
        self.fit(X, valid_ratio)
        return self.transform(X)


    def _reparameterized_sample(self, mean:torch.tensor, logstd:torch.tensor) -> torch.tensor:
        """using std to sample"""
        eps = torch.FloatTensor(logstd.size()).normal_().to(self.device)
        return eps.mul(logstd.exp()).add_(mean)