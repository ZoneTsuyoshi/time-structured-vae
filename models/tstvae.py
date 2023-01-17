from .tsvae import *


class tsTVAE(tsVAE):
    def __init__(self, input_dim:int, lagtime:int=1, autocorrelation:float=0.999, autocorrelation_mode:str="fixed",
                 decaying_time:int=1000, autocorrelation_prior:str="Beta",
                 beta:float=0., n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True, print_every:int=10, cuda:Union[bool, int]=True, save_dir=None):
        super(tsTVAE, self).__init__(input_dim, lagtime, autocorrelation, autocorrelation_mode, decaying_time,
                                autocorrelation_prior, beta, n_epochs, batch_size, learning_rate, latent_dim, hidden_dim, 
                                n_layers, sliding_window, activation, dropout_ratio, optimizer, loss, verbose, print_every,
                                cuda, save_dir)
        
        
    def _compute_loss(self, X:torch.Tensor):
        """
        Args:
            X [batch_size, input_dim, 2]
        """
        X = X.permute(2,0,1).to(self.device)

        o, loss_list = self(X) #(2,ns,dx)
        rec_loss = self.loss_fn(o[0], X[1].detach())
        loss_list = [rec_loss] + loss_list

        return [sum(loss_list)] + loss_list