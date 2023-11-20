from typing import Optional
import math
import torch
import torch.nn as nn


def construct_dense_network(input_dim:int, output_dim:int=None, hidden_dim:int=50, n_layers:int=1, activation:str="LeakyReLU", dropout_ratio:float=0., output_activation:str=None, use_batch_norm:bool=False, pre_batchnorm:bool=False, n_dropout_layers:Optional[int]=None):
    if type(hidden_dim)==int:
        hidden_dim = [hidden_dim] * max(n_layers - 1, 1)
    if output_dim is None:
        output_dim = hidden_dim[-1]
    if n_dropout_layers is None:
        n_dropout_layers = n_layers - 1
    if n_layers==1:
        net = [nn.Linear(input_dim, output_dim)]
    else:
        net = [nn.Linear(input_dim, hidden_dim[0])]
        if pre_batchnorm and use_batch_norm:
            net.append(nn.BatchNorm1d(hidden_dim[0]))
        net.append(getattr(nn, activation)())
        if dropout_ratio>0. and n_dropout_layers>0:
            net.append(nn.Dropout(dropout_ratio))
        if use_batch_norm and not pre_batchnorm:
            net.append(nn.BatchNorm1d(hidden_dim[0]))
        for i in range(n_layers-2):
            net.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            if use_batch_norm and pre_batchnorm:
                net.append(nn.BatchNorm1d(hidden_dim[i+1]))
            net.append(getattr(nn, activation)())
            if dropout_ratio>0. and i+1 < n_dropout_layers:
                net.append(nn.Dropout(dropout_ratio))
            if use_batch_norm and not pre_batchnorm:
                net.append(nn.BatchNorm1d(hidden_dim))
        net += [nn.Linear(hidden_dim[-1], output_dim)]
    if output_activation is not None:
        if output_activation=="Softmax":
            net.append(getattr(nn, output_activation)(dim=-1))
        else:
            net.append(getattr(nn, output_activation)())
    return nn.Sequential(*net)



def compute_sample_correlation(x, y):
    """Compute sample correlation"""
    x = x.view(-1)
    y = y.view(-1)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def compute_autocorrelation_loss(alpha, prior="Beta"):
    if prior=="standard":
        autocorrelation_loss = - alpha
    elif prior=="Beta":
        autocorrelation_loss = - (2 * torch.log(alpha) - math.log(3))
    elif prior=="exp":
        autocorrelation_loss = - torch.log(torch.exp(math.log(3)*alpha)-1)
    return autocorrelation_loss.sum()


def kld_between_two_Gaussians(mean_1, logstd_1, mean_2, logstd_2, dim=-1, keepdim=False):
    """Compute KLD b/w two Gaussians"""
    kld_element =  (2 * logstd_2 - 2 * logstd_1 + (logstd_1.exp().pow(2) + (mean_1 - mean_2).pow(2)) / logstd_2.exp().pow(2) - 1)
    return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)



def kld_gauss_normal(mean, logstd, dim=-1, keepdim=False):
    """Compute KLD w/ standard Gaussian"""
    kld_element =  (-2 * logstd + logstd.exp().pow(2) + mean.pow(2) - 1)
    return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)



def log_prob(mean, logstd, x, dim=-1, keepdim=False, logvar=False):
    """Negative log likelihood for Gaussian distribution"""
    return - 0.5 * torch.sum((x - mean).pow(2) / logstd.exp() + math.log(2 * math.pi) + logstd, dim=dim, keepdim=keepdim)