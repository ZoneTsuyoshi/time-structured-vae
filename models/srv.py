from typing import Optional
from scipy import linalg
from .base import *
from .vamploss import VAMPLoss


class SRV(BaseModel):
    def __init__(self, input_dim:int, lagtime:int=1, n_epochs:int=100, batch_size:int=256, learning_rate:float=1e-3,
                 latent_dim:int=1, hidden_dim:int=50, n_layers:int=2, sliding_window:bool=True,
                 activation:str="LeakyReLU", dropout_ratio:float=0., optimizer:str="Adam", vamp_r: int = 2,
                 vamp_mode: str = "trunc", vamp_epsilon : float = 1e-6, vamp_rank: Optional[int] = None, vamp_reversible: bool = False,
                 verbose:bool=True, print_every:int=10, cuda:Union[bool, int]=True, save_dir=None):
        super(SRV, self).__init__(input_dim, lagtime, n_epochs, batch_size, latent_dim, sliding_window, "VAMP", verbose, print_every, cuda, save_dir)

        self.reversible = vamp_reversible

        self.encoder = construct_dense_network(input_dim, latent_dim, hidden_dim, n_layers, activation, dropout_ratio, use_batch_norm=True)
        self.to(self.device)
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=learning_rate)
        self.loss_fn = VAMPLoss(vamp_r, vamp_mode, vamp_epsilon, vamp_rank, vamp_reversible)
        
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.encoder(X)
        return Z


    def _compute_loss(self, X:torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            X [batch_size, input_dim, 2]
        """
        X = X.to(self.device)
        x0 = X[:, :, 0]
        x1 = X[:, :, 1]

        z0 = self.encoder(x0) #(ns,dz)
        z1 = self.encoder(x1) #(ns,dz)

        # compute the vamp loss
        vamp_loss = self.loss_fn(z0, z1)

        return [vamp_loss, vamp_loss]


    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform
        Args:
            X (np.ndarray): input variables [n_samples, *, input_dim]

        Returns:
            np.ndarray: transformed latent variables [n_samples, *, latent_dim]
        """
        self.eval()

        # encode
        X = torch.from_numpy(X).to(self.device)
        shape = X.shape
        X = X.reshape(-1, shape[-1]) #(ns,dx)
        Z = self.encoder(X) #(ns,dz)
        Z = Z.reshape(shape[:-1] + (-1,))
        Z = Z.detach().cpu().numpy()

        # compute means
        self._means = Z.mean(0)

        # replicate by time-lag
        z0 = Z[:-self.lagtime]
        zt = Z[self.lagtime:]

        # compute covariance matrix
        C00 = z0.T @ z0 / len(z0)
        C01 = z0.T @ zt / len(z0)
        C10 = zt.T @ z0 / len(z0)
        C11 = zt.T @ zt / len(z0)

        # compute eigenpairs
        self._compute_vac_basis_from_covariances(Z, C00, C01, C10, C11)

        # vac encode
        Z_vac = self._vac_encoder(Z)

        return Z_vac


    def _compute_vac_basis_from_covariances(self, z: np.ndarray, C00: np.ndarray, C01: np.ndarray, C10: np.ndarray, C11: np.ndarray) -> None:
        if self.reversible:
            C0 = 0.5 * (C00 + C11)
            C1 = 0.5 * (C01 + C10)

            # compute eigenpairs
            eigvals, eigvecs = linalg.eigh(C0, C1, eigvals_only=False)
            idx = eigvals.argsort()[::-1]

            # store the eigenpairs
            self._eigenvals = eigvals[idx]
            self._eigenvecs = eigvecs[:, idx]
        else:
            C00_sqrt_inv = linalg.fractional_matrix_power(C00, -0.5)
            C11_sqrt_inv = linalg.fractional_matrix_power(C11, -0.5)

            P = C00_sqrt_inv @ C01 @ C11_sqrt_inv
            Up, S, _ = linalg.svd(P)
            U = C00_sqrt_inv @ Up

            idx = S.argsort()[::-1]
            self._eigenvals = S[idx]
            self._eigenvecs = U[:, idx]
        
        # store the norm
        z = (z - self._means) @ self._eigenvecs
        self._norms = np.sqrt((z*z).mean(0))


    def _vac_encoder(self, z: torch.Tensor) -> torch.Tensor:
        z -= self._means
        z = z @ self._eigenvecs
        z /= self._norms
        return z
        