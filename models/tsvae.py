from .tvae import *
from .utils import kld_gauss_normal, kld_between_two_Gaussians, compute_autocorrelation_loss


class tsVAE(TVAE):
    def __init__(self, input_dim:int, lag_time:int=1, autocorrelation:float=0.999, autocorrelation_mode:str="fixed",
                 decaying_time:int=1000, autocorrelation_prior:str="Beta",
                 beta:float=0., n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", loss:str="MSELoss",
                 verbose:bool=True,　print_every:int=10, cuda:Union[bool, int]=True, save_dir=None):
        super(tsVAE, self).__init__(input_dim, lag_time, beta, n_epochs, batch_size, learning_rate, latent_dim, hidden_dim, 
                                n_layers, sliding_window, activation, dropout_ratio, optimizer, loss, verbose, print_every,
                                cuda, save_dir)
        self.loss_name_list.append("KLD Transition Loss")
        
        if type(autocorrelation) == int:
            autocorrelation = [autocorrelation]

        if autocorrelation_mode=="adjusted":
            if type(decaying_time) == int:
                decaying_time = [decaying_time]
            autocorrelation = [np.sqrt(1 - lag_time / st) for st in decaying_time]

        self.alpha = torch.Tensor(autocorrelation * torch.ones(self.latent_dim)).to(self.device)
        
        self.learning_autocorrelation = autocorrelation_mode=="learned"
        if self.learning_autocorrelation:
            self.autocorrelation_prior = autocorrelation_prior
            self.alpha = nn.Parameter(torch.logit(2*self.alpha.clamp(0,1)-1), requires_grad=True)
            self.sigmoid = lambda x:0.5*(F.sigmoid(x)+1)
            self.loss_name_list += ["Negative Autocorrelation Log-Prior"]
        else:
            self.sigmoid = nn.Identity()
            
        
        def forward(self, x):
            """
            Args:
                x:[2,*,input_dim]
            """
            enc = torch.moveaxis(self.encoder(x), -1, 0)
            z = self._reparameterized_sample(enc[:self.latent_dim], enc[self.latent_dim:])
            o = self.decoder(z)
            alpha = self.sigmoid(self.alpha)
            loss_list = [self.beta * kld_gauss_normal(enc[:self.latent_dim], enc[self.latent_dim:], z),
                   kld_between_two_Gaussians(enc[:self.latent_dim,1], enc[self.latent_dim:,1], alpha*z[0], torch.sqrt(1-alpha**2))]
            if self.learning_autocorrelation:
                loss_list.append(compute_autocorrelation_loss(alpha, self.autocorrelation_prior))
            return o, loss_list
            
            
        def compute_loss(self, X:np.ndarray):
            """
            Args:
                X [batch_size, input_dim, 2]
            """
            X = X.permute(2,0,1).to(device)
            
            o, loss_list = self(X) #(2,ns,dx)
            rec_loss = self.loss_fn(o, X.detach())
            loss_list = [rec_loss] + loss_list
            
            return [sum(loss_list)] + loss_list