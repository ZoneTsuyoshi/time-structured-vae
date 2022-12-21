from .tvae import *
from .utils import compute_sample_correlation


class VDE(TVAE):
    def __init__(self, input_dim:int, lag_time:int=1, beta:float=0., n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True,　print_every:int=10, cuda:Union[bool, int]=True, save_dir=None):
        super(VDE, self).__init__(input_dim, lag_time, beta, n_epochs, batch_size, learning_rate, latent_dim, hidden_dim, 
                                n_layers, sliding_window, activation, dropout_ratio, optimizer, loss, verbose, print_every,
                                cuda, save_dir)
        self.loss_name_list += ["Negative Sample Autocorrelation"]
            
            
        def compute_loss(self, X:np.ndarray):
            """
            Args:
                X [batch_size, input_dim, 2]
            """
            X = X.to(device)
            x = X[:, :, 0]
            y = X[:, :, 1]
            
            o, zx, kld_loss = self(x) #(ns,dx),(ns,dz)
            _, zy, _ = self(y) #(ns,dz)
            rec_loss = self.loss_fn(o, y.detach())
            sa_loss = - compute_sample_correlation(zx, zy)
            
            return [rec_loss + kld_loss + sa_loss, rec_loss, kld_loss, sa_loss]