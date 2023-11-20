from typing import Optional, Union
import torch
import torch.nn as nn

class VAMPLoss(nn.Module):
    def __init__(self, r: int = 2, mode: str = "trunc", epsilon : float = 1e-7, rank: Optional[int] = None, reversible: bool = False):
        super(VAMPLoss, self).__init__()
        self.r = r
        self.mode = mode
        self.epsilon = epsilon
        self.rank = rank
        self.reversible = reversible

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X [batch_size, dim]
            Y [batch_size, dim]
        """
        if not self.reversible:
            # compute the vamp score
            vamp_matrix = self._compute_Koopman_matrix(X, Y) #(dim,dim)

            # compute the vamp score
            vamp_score = self._compute_vamp_score(vamp_matrix)
        else:
            # compute the vamp score
            vamp_score = self._compute_reversible_vamp_score(X, Y)

        return - vamp_score
    

    def _compute_Koopman_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        X = X.T; Y = Y.T

        # compute covariance matrix
        C01 = X @ Y.T / (batch_size - 1) #(dim,dim)
        C00 = X @ X.T / (batch_size - 1) #(dim,dim)
        C11 = Y @ Y.T / (batch_size - 1) #(dim,dim)

        #  compute the inverse matrices
        C00_sqrt_inv = self._inv(C00, ret_sqrt=True) #(dim,dim)
        C11_sqrt_inv = self._inv(C11, ret_sqrt=True) #(dim,dim)

        # compute the vamp score
        vamp_matrix = C00_sqrt_inv @ C01 @ C11_sqrt_inv #(dim,dim)

        return vamp_matrix
    

    def _compute_vamp_score(self, vamp_matrix: torch.Tensor) -> torch.Tensor:
        if self.rank is not None:
            eigvals = torch.linalg.svdvals(vamp_matrix).sort(descending=True).values #(dim,)
            eigvals = eigvals[:self.rank]
            vamp_score = torch.sum(torch.abs(eigvals)**self.r)
        if self.r == 1:
            vamp_score = torch.norm(vamp_matrix, p="nuc")
        elif self.r == 2:
            vamp_score = torch.norm(vamp_matrix, p="fro")**2
        elif self.r > 0:
            eigvals = torch.linalg.eigvalsh(vamp_matrix) #(dim,)
            vamp_score = torch.sum(torch.abs(eigvals)**self.r)
        else:
            raise ValueError("r must be positive")
        
        return vamp_score
    

    def _compute_reversible_vamp_score(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        X = X.T; Y = Y.T

        # compute covariance matrix
        C01 = X @ Y.T / (batch_size - 1) #(dim,dim)
        C00 = X @ X.T / (batch_size - 1) #(dim,dim)
        C11 = Y @ Y.T / (batch_size - 1) #(dim,dim)

        C10 = Y @ X.T / (batch_size - 1) #(dim,dim)
        C0 = 0.5 * (C00 + C11) #(dim,dim)
        C1 = 0.5 * (C01 + C10) #(dim,dim)
        
        # cholensky decomposition
        L = torch.linalg.cholesky(C0)
        L_inv = torch.linalg.inv(L)

        # compute the self-adjoint matrix
        M = L_inv @ C1 @ L_inv.T

        # compute the eigenvalues
        eigvals = torch.linalg.eigvalsh(M)

        # compute the vamp score
        vamp_score = torch.sum(torch.abs(eigvals)**self.r)

        return vamp_score


    def _inv(self, x: torch.Tensor, ret_sqrt: bool = False):
        '''Utility function that returns the inverse of a matrix, with the
        option to return the square root of the inverse matrix.
        Parameters
        ----------
        x: torch tensor with shape [dx,dx]
            matrix to be inverted
            
        ret_sqrt: bool, optional, default = False
            if True, the square root of the inverse matrix is returned instead
        Returns
        -------
        x_inv: numpy array with shape [m,m]
            inverse of the original matrix
        '''
        # print(torch.linalg.svdvals(x))
        if self.mode=="regularize":
            eigval, eigvec = torch.linalg.eigh(x + self.epsilon*torch.eye(x.shape[0], device=x.device, dtype=x.dtype)) #(dx),(dx,dx)
            eigval = torch.abs(eigval)
        else:
            eigval, eigvec = torch.linalg.eigh(x) #(dx),(dx,dx)	

        if self.mode=="trunc":
            # Filter out eigvalues below threshold and corresponding eigvectors
            index_eig = eigval > self.epsilon
            eigval = eigval[index_eig]
            eigvec = eigvec[:,index_eig]

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        diag = torch.diag(torch.sqrt(1/eigval)) if ret_sqrt else torch.diag(1/eigval)

        # Rebuild the square root of the inverse matrix
        x_inv = eigvec @ diag @ eigvec.T
        return x_inv