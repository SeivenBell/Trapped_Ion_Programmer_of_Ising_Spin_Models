import torch
from .model import SpinDecoder

########################################################################################


def generate_feasible_interactions(
    N: int, batch_size: int, spin_decoder: SpinDecoder, device: torch.device
):
    Omega = torch.rand((2, batch_size, N))
    Omega = Omega[0][..., :, None] * Omega[1][..., None, :]
    Omega = Omega.to(device)

    J = spin_decoder.to(device)(Omega)
    return J


def generate_random_interactions(N: int, batch_size: int, device: torch.device):
    J = torch.randn((batch_size, int(N * (N - 1) / 2)))
    J = J / torch.norm(J, dim=-1, keepdim=True)
    J = J.to(device=device)
    return J
