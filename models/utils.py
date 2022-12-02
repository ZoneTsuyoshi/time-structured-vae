import torch.nn as nn


def construct_dense_network(input_dim:int, output_dim:int=None, hidden_dim:int=50, n_layers:int=1, activation:str="LeakyReLU", dropout_ratio:float=0., output_activation:str=None,):
    if output_dim is None:
        output_dim = hidden_dim
    if n_layers==1:
        net = [nn.Linear(input_dim, output_dim)]
    else:
        net = [nn.Linear(input_dim, hidden_dim)]
        if dropout>0.:
            net.append(nn.Dropout(dropout_ratio))
        for i in range(n_layers-1):
            net += [getattr(nn, activation),
                    nn.Linear(hidden_dim, hidden_dim)]
            if dropout>0.:
                net.append(nn.Dropout(dropout_ratio))
        net += [getattr(nn, activation),
                nn.Linear(hidden_dim, output_dim)]
    if output_activation_class is not None:
        net.append(getattr(nn, output_activation))
    return nn.Sequential(*net)



def kld_between_two_gauss(mean_1, logstd_1, mean_2, logstd_2, dim=-1, keepdim=False):
    """Using std to compute KLD"""
    kld_element =  (2 * logstd_2 - 2 * logstd_1 + (logstd_1.exp().pow(2) + (mean_1 - mean_2).pow(2)) / logstd_2.exp().pow(2) - 1)
    return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)



def kld_gauss_normal(mean, logstd, dim=-1, keepdim=False):
    """Using std to compute KLD"""
    kld_element =  (-2 * logstd + logstd.exp().pow(2) + mean.pow(2) - 1)
    return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)



def log_prob(mean, logstd, x, dim=-1, keepdim=False, logvar=False):
    """Negative log likelihood for Gaussian distribution"""
    return - 0.5 * torch.sum((x - mean).pow(2) / logstd.exp() + math.log(2 * math.pi) + logstd, dim=dim, keepdim=keepdim)