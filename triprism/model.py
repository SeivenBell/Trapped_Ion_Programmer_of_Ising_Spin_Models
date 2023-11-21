import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

import trical

########################################################################################


class SpinDecoder(nn.Module):
    def __init__(self, ti, mu, deltak=torch.Tensor([1, 0, 0]) * 4 * torch.pi / 355e-9):
        

        pass

    def vectorize_J(self, x):
        
        return y

    def matrixify_J(self, x):
        
        return J

    def forward(self, x):
        
        return y

    pass


########################################################################################


class RabiEncoder(nn.Module):
    def __init__(self, N, N_h, activation=torch.nn.Identity()):
        

        pass

    def forward(self, x):
       

        return y

    def forwardOmega(self, x):
        
        return y

    pass


class kRankRabiEncoder(nn.Module):
    def __init__(self, k, N, N_h, activation=torch.nn.Identity()):
        super().__init__()

        

        pass

    def forward(self, x):
        
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
        

    def dynamic_range_penalty(self, x):
    
        return (Omega.std() / Omega.mean()) ** 2

    pass
