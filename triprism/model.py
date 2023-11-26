import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

import trical

########################################################################################


class SpinDecoder(nn.Module):
    def __init__(self, ti, mu, deltak=torch.Tensor([1, 0, 0]) * 4 * torch.pi / 355e-9):
            """
            A PyTorch Module for decoding quantum states in a system of trapped ions. 
            It takes into account the physical properties of the ions and their interactions, 
            transforming encoded quantum states into a matrix form representing spin interactions.

            Parameters:
            - ti (object): An object representing trapped ions in a quantum system. This should 
                contain attributes like the number of ions (N), mass (m), frequencies (w), and 
                coupling coefficients (b).
            - mu (torch.Tensor): A tensor representing specific physical parameters of the system,
                possibly related to the frequencies or other characteristics relevant to the 
                quantum interactions.
            - deltak (torch.Tensor, optional): A tensor representing the change in the wave 
                vector, related to the wavelength of the light used in ion trapping.
                Default is set to correspond to a 355 nm wavelength.

            The class registers several buffers that are constants during the training:
            - hbar: Reduced Planck's constant, obtained from trical.misc.constants.
            - m: Mass of the ions, derived from the 'ti' object.
            - w: Frequencies associated with the ions, derived from 'ti'.
            - b: Coupling matrix reshaped for computation, derived from 'ti'.
            - mu: Physical parameter tensor passed during initialization.
            - deltak: Change in wave vector tensor.
            - eta: Tensor representing the coupling strength between the ions and the external field.
            - nu: Tensor representing the inverse of the difference between the square of mu and the square of frequencies.
            - w_mult_nu: Element-wise product of w and nu.

            Methods:
            - vectorize_J(x): Converts a matrix of interactions (J) into a vector form.
            - matrixify_J(x): Converts a vectorized interaction matrix back into its matrix form.
            - forward(x): Performs the forward pass of the network. It computes the interaction matrix 
                from the input, vectorizes it, normalizes it, and then returns the result.

            """
            pass

    def vectorize_J(self, x):
            # Converts a matrix of interactions (J) into a vector form. 
            # This is useful for processing in neural networks.
            idcs = torch.triu_indices(self.N, self.N, 1)
            y = x[..., idcs[0], idcs[1]]
            return y

    def matrixify_J(self, x):
        # The inverse of vectorize_J; it converts the vectorized interactions back into a matrix form.
        
        return J

    def forward(self, x):
        # x: A tensor of interactions (J)
        # y: A tensor of ion spins (S)
        # computes the interaction matrix from the input 
        # vectorizes it, normalizes it, and then returns the result.
        # einsum("...im,in->...imn", x, self.eta)
        # einsum("...imn,...jmn->...ij", y, y * self.w_mult_nu)
        
        return y

    pass


########################################################################################


class RabiEncoder(nn.Module):
    def __init__(self, N, N_h, activation=torch.nn.Identity()):
        # N: The size of the input.
        # N_h: The number of hidden units.
        # activation: An activation function (default is identity).
        
        # self.linear and self.linear2) for transforming the input data.

        pass

    def forward(self, x):
       # Processes the input through the linear layers and an activation function
       # (ReLU followed by the specified activation function), and reshapes the output to 
       # a specific format.

        return y

    def forwardOmega(self, x):
       # pass that normalizes the output of the forward method.
        
        return y

    pass


class kRankRabiEncoder(nn.Module):
    #k: the rank for the approximation.
    def __init__(self, k, N, N_h, activation=torch.nn.Identity()):
        super().__init__()
         #it adds another linear layer (self.linear3)
        

        

        pass

    def forward(self, x):
        # Similar to RabiEncoder, but processes the input into two separate paths (y1 and y2) 
        # 2 reshapes ruquired (y1 and y2)
        y = (y1, y2)

        return y

    def forwardOmega(self, x):
        
        return y

    pass


########################################################################################


class PrISM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self._reset_parameters()

        pass

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        pass

    def forward(self, x):
        y = self.encoder.forwardOmega(x)
        y = self.decoder(y)

        return y

    def smoothed_reconstruction_loss(self, x, smoothing_factor):

        y = self.decoder(y)

        return 1 - torch.einsum("...i,...i->...", y, x).mean()

    def reconstruction_loss(self, x):
        # einsum needed torch.einsum("...i,...i->...",
        

    def dynamic_range_penalty(self, x):
        #Computes a penalty based on the dynamic range of the encoded output, 
        # encouraging model stability.
    
        return (Omega.std() / Omega.mean()) ** 2

    pass
