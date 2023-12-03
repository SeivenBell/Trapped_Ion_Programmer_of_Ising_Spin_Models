import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
import trical
import numpy as np

torch.autograd.set_detect_anomaly(True)

########################################################################################

class SpinDecoder(nn.Module):
    """
    A neural network module that decodes spin interactions in a quantum system.

    Attributes:
    - N (int): The size of the spin system.
    - hbar, m, w, b, mu, deltak, eta, nu, w_mult_nu: Physical parameters and tensors representing the quantum system.
    """
    def __init__(self, ti, mu, deltak=torch.tensor([1, 0, 0]) * 4 * torch.pi / 355e-9):
        super(SpinDecoder, self).__init__()
        self.N = ti.N

        # Define the constants and parameters directly as class attributes
        self.hbar = torch.tensor(trical.misc.constants.hbar, dtype=torch.float32)
        self.m = torch.tensor(ti.m, dtype=torch.float32)
        self.w = torch.tensor(ti.w, dtype=torch.float32)
        self.b = ti.b.reshape(3, self.N, 3 * self.N).to(torch.float32)
        self.mu = mu.to(torch.float32)
        self.deltak = deltak.to(torch.float32)

        # Compute additional parameters
        sqrt_term = torch.sqrt(self.hbar / (2 * self.m * self.w))
        self.eta = torch.matmul(self.b, self.deltak.view(-1, 1)).view(self.N, 3) * sqrt_term
        self.nu = 1 / ((self.mu**2)[:, None] - (self.w**2)[None, :])
        self.w_mult_nu = self.w * self.nu

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
        # Replace einsum for more intuitive tensor operations
        y = torch.matmul(x.unsqueeze(-1), self.eta.unsqueeze(0)).squeeze(-1)
        y = torch.matmul(y, (y * self.w_mult_nu).transpose(-1, -2))
        
        vectorized_y = self.vectorize_J(y)
        normalized_y = vectorized_y / torch.norm(vectorized_y, dim=-1, keepdim=True)
        return normalized_y

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
            nn.Linear(self.N_h, self.N_h),
            self.activation,
            nn.Linear(self.N_h, self.N * self.N)
            
        )
        
    def forwardOmega(self, x):
        """
        Normalizes the output of the forward pass.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Normalized tensor after encoding.
        """
        y = self.forward(x)
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

    # TODO: understand how these loss functions compares target and predicted configuration matrices

    def reconstruction_loss(self, x):
        """
        Compute the reconstruction loss for the model.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: The reconstruction loss.
        """
        return 1 - torch.einsum("...i,...i->...", self(x), x).mean()
