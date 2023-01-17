from .base import *
from .utils import kld_gauss_normal


class TVAE(BaseModel):
    def __init__(self, input_dim:int, lagtime:int=1, beta:float=0., n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True, print_every:int=10, cuda:Union[bool, int]=True, save_dir=None):
        super(TVAE, self).__init__(input_dim, lagtime, n_epochs, batch_size, learning_rate, latent_dim, hidden_dim, 
                                n_layers, sliding_window, activation, dropout_ratio, optimizer, loss, verbose, print_every,
                                cuda, save_dir, True, False)
        self.beta = beta
        self.loss_name_list += ["KLD Loss"]
        
    def forward(self, x):
        enc = torch.moveaxis(self.encoder(x), -1, 0) #(2*dz,*)
        enc_mean, enc_logstd = torch.moveaxis(enc[:self.latent_dim], 0, -1), torch.moveaxis(enc[self.latent_dim:], 0, -1) #(*,dz)
        z = self._reparameterized_sample(enc_mean, enc_logstd)
        o = self.decoder(z) #(*,dx)
        kld_loss = self.beta * kld_gauss_normal(enc_mean, enc_logstd).mean()
        return o, z, kld_loss


    def _compute_loss(self, X:torch.Tensor):
        """
        Args:
            X [batch_size, input_dim, 2]
        """
        X = X.to(self.device)
        x = X[:, :, 0]
        y = X[:, :, 1]

        o, _, kld_loss = self(x) #(ns,dx),(ns,dz)
        rec_loss = self.loss_fn(o, y.detach()).sum(-1).mean()

        return [rec_loss + kld_loss, rec_loss, kld_loss]