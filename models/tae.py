from .base import *


class TAE(BaseEncoderDecoder):
    def __init__(self, input_dim:int, lagtime:int=1, n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True, print_every:int=10, cuda:Union[bool, int]=True, save_dir=None):
        super(TAE, self).__init__(input_dim, lagtime, n_epochs, batch_size, learning_rate, latent_dim, hidden_dim, 
                                n_layers, sliding_window, activation, dropout_ratio, optimizer, loss, verbose, print_every,
                                cuda, save_dir, False, False)
        
        
    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return o


    def _compute_loss(self, X:torch.Tensor):
        """
        Args:
            X [batch_size, input_dim, 2]
        """
        X = X.to(self.device)
        x = X[:, :, 0]
        y = X[:, :, 1]

        o = self(x) #(ns,dx)
        rec_loss = self.loss_fn(o, y.detach()).sum(-1).mean()

        return [rec_loss, rec_loss]
        