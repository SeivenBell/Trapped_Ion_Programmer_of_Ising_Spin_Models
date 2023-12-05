import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

import trical

########################################################################################


class SpinDecoder(nn.Module):
    """
    A neural network module that decodes spin interactions in a quantum system.

    Attributes:
    - N (int): The size of the spin system.
    - hbar, m, w, b, mu, deltak, eta, nu, w_mult_nu: Physical parameters and tensors representing the quantum system.
    """
    def __init__(self, ti, mu, deltak=torch.Tensor([1, 0, 0]) * 4 * torch.pi / 355e-9):
        super().__init__()
        # Define the constants and parameters directly as class attributes
        self.N = ti.N

        self.hbar = torch.Tensor([trical.misc.constants.hbar]).to(torch.float32)
        self.m = torch.Tensor([ti.m]).to(torch.float32)
        self.w = torch.from_numpy(ti.w).to(torch.float32)
        self.b = torch.from_numpy(ti.b).to(torch.float32).reshape(3, self.N, 3 * self.N)
        self.mu = mu.to(torch.float32)
        self.deltak = deltak.to(torch.float32)
        
        # Compute additional parameters
        self.eta = torch.einsum("kim,k,m->im", self.b, self.deltak, torch.sqrt(self.hbar / (2 * self.m * self.w)))
        self.nu = 1 / ((self.mu**2)[:, None] - (self.w**2)[None, :])
        self.w_mult_nu = self.w * self.nu

        pass

    
    def vectorize_J(self, x):
        """
        Converts a matrix J to a vector by extracting the upper triangular part.

        Args:
        - x: The matrix to be vectorized.

        Returns:
        - Tensor: The vectorized form of the matrix.
        """
        N = x.size(-1)
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        return x[..., mask]

    def matrixify_J(self, x):
        """
        Converts a vector back to a matrix J.

        Args:
        - x: The vector to be converted into a matrix.

        Returns:
        - Tensor: The matrix form of the vector.
        """
        N = self.N
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        J = torch.zeros((*x.shape[:-1], N, N), device=x.device)
        J[..., mask] = x
        J = J + J.transpose(-2, -1)  # Make the matrix symmetric
        return J

    def forward(self, x):
        y = torch.einsum("...im,in->...imn", x, self.eta)
        y = torch.einsum("...imn,...jmn->...ij", y, y * self.w_mult_nu)
        y = self.vectorize_J(y)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        return y

    pass


########################################################################################
class RabiEncoder(nn.Module):
    """
    Encodes Rabi frequencies for a quantum system using a neural network.

    Attributes:
    - N (int): Size of the spin system.
    - N_h (int): Number of hidden units in the linear layers.
    - activation (torch.nn.Module): Activation function.
    - layers (torch.nn.Sequential): Sequential container of layers.
    """
    def __init__(self, N, N_h, activation=None):
        """
        Initializes the RabiEncoder with specified parameters.

        Args:
        - N (int): Size of the spin system.
        - N_h (int): Number of hidden units.
        - activation (torch.nn.Module): Activation function. Defaults to Identity if None.
        """
        super(RabiEncoder, self).__init__()
        self.N = N
        self.N_h = N_h # optionally use as the number of hidden units
        self.activation = nn.ReLU() if activation is None else activation

        # Feedforward network
        self.layers = nn.Sequential(
            nn.Linear(self.N * (self.N - 1) // 2, self.N_h),
            self.activation,
            nn.Linear(self.N_h, self.N * self.N)
        )

    def forward(self, x):
        y = self.layers(x)
        y = y.reshape(-1, self.N, self.N)

        return y
        
    def forwardOmega(self, x):
        """
        Normalizes the output of the forward pass.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Normalized tensor after encoding.
        """
        y = self(x)
        return y / torch.norm(y, dim=(-1, -2), keepdim=True)


########################################################################################

class PrISM(nn.Module):
    """
    PrISM model integrating an encoder and a decoder for processing quantum system interactions.

    Attributes:
    - encoder (torch.nn.Module): Encoder module.
    - decoder (torch.nn.Module): Decoder module.
    """
    def __init__(self, encoder, decoder):
        """
        Initializes the PrISM model with an encoder and a decoder.

        Args:
        - encoder (torch.nn.Module): An instance of the encoder.
        - decoder (torch.nn.Module): An instance of the decoder.
        """
        super(PrISM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset parameters using Xavier uniform initialization. """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass through the PrISM model.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: The output tensor after encoding and decoding.
        """
        y = self.encoder(x)
        return self.decoder(y)


    def reconstruction_loss(self, x):
        """
        Compute the reconstruction loss for the model.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: The reconstruction loss.
        """
        return 1 - torch.einsum("...i,...i->...", self(x), x).mean()